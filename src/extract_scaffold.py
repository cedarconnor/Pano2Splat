"""Extract scaffold Gaussians from One2Scene into standard 3DGS PLY format.

One2Scene's model outputs Gaussians with covariance matrices (means, covariances,
harmonics, opacities). This script decomposes covariances back into scales and
rotations (quaternion wxyz) and saves as a standard 3DGS .ply file compatible
with SuperSplat and gsplat.

Also extracts depth maps and anchor camera parameters from the data.pth file
produced by Stage A.

Usage:
    python -m src.extract_scaffold \
        --data_pth pipeline_output/stage_a/data.pth \
        --output_dir pipeline_output/stage_a/
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from src.utils.ply_io import save_ply


def decompose_covariance(covariances: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose covariance matrices into scales and rotation quaternions.

    Covariance = R @ S @ S^T @ R^T, where S = diag(scales).
    We recover R and scales via eigendecomposition: C = V @ diag(eigenvalues) @ V^T
    Then scales = sqrt(eigenvalues), R = V (with handedness correction).

    Args:
        covariances: (N, 3, 3) covariance matrices.

    Returns:
        scales: (N, 3) in log-space (log of actual scales).
        rotations: (N, 4) quaternions in wxyz convention.
    """
    # Eigendecomposition: covariances = V @ diag(eigenvalues) @ V^T
    eigenvalues, eigenvectors = torch.linalg.eigh(covariances)

    # Clamp eigenvalues to be positive (numerical stability)
    eigenvalues = eigenvalues.clamp(min=1e-8)

    # Scales = sqrt(eigenvalues), stored in log-space
    scales = torch.sqrt(eigenvalues)
    log_scales = torch.log(scales)

    # Rotation matrix = eigenvectors (columns are principal axes)
    # Ensure proper rotation (det = +1)
    det = torch.linalg.det(eigenvectors)
    # Flip last column if det is negative
    flip = (det < 0).float().unsqueeze(-1).unsqueeze(-1)
    correction = torch.ones_like(eigenvectors)
    correction[..., :, -1] = 1 - 2 * flip.squeeze(-1)
    eigenvectors = eigenvectors * correction

    # Convert rotation matrices to quaternions (wxyz)
    rotations = matrix_to_quaternion_wxyz(eigenvectors)

    return log_scales, rotations


def matrix_to_quaternion_wxyz(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to quaternions in wxyz convention.

    Args:
        R: (N, 3, 3) rotation matrices.

    Returns:
        (N, 4) quaternions [w, x, y, z].
    """
    N = R.shape[0]
    q = torch.zeros(N, 4, device=R.device, dtype=R.dtype)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # Case 1: trace > 0
    s = torch.sqrt((trace + 1.0).clamp(min=1e-8)) * 2  # s = 4*w
    mask1 = trace > 0
    q[mask1, 0] = 0.25 * s[mask1]
    q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s[mask1]
    q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s[mask1]
    q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s[mask1]

    # Case 2: R[0,0] is largest diagonal
    mask2 = ~mask1 & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    s2 = torch.sqrt((1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]).clamp(min=1e-8)) * 2
    q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s2[mask2]
    q[mask2, 1] = 0.25 * s2[mask2]
    q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s2[mask2]
    q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s2[mask2]

    # Case 3: R[1,1] is largest diagonal
    mask3 = ~mask1 & ~mask2 & (R[:, 1, 1] > R[:, 2, 2])
    s3 = torch.sqrt((1.0 + R[:, 1, 1] - R[:, 0, 0] - R[:, 2, 2]).clamp(min=1e-8)) * 2
    q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s3[mask3]
    q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s3[mask3]
    q[mask3, 2] = 0.25 * s3[mask3]
    q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s3[mask3]

    # Case 4: R[2,2] is largest diagonal
    mask4 = ~mask1 & ~mask2 & ~mask3
    s4 = torch.sqrt((1.0 + R[:, 2, 2] - R[:, 0, 0] - R[:, 1, 1]).clamp(min=1e-8)) * 2
    q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s4[mask4]
    q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s4[mask4]
    q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s4[mask4]
    q[mask4, 3] = 0.25 * s4[mask4]

    # Normalize
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    return q


def extract_from_data_pth(data_pth_path: str, output_dir: str):
    """Extract anchor cameras and images from data.pth.

    data.pth contains:
        'cuda_images': (N_total, 3, H, W) - context + target rendered images
        'conda_mask_1': (N_total,) - mask for first context view
        'conda_mask_6': (N_total,) - mask for 6 context views
        'intrinsics': (N_total, 3, 3)
        'extrinsics': (N_total, 4, 4) - world-to-camera
        'seva_c2w': (N_total, 4, 4) - camera-to-world for SEVA
        'scene': scene name
    """
    print(f"Loading data.pth from {data_pth_path}")
    data = torch.load(data_pth_path, map_location="cpu", weights_only=False)

    images = data["cuda_images"]  # (N, 3, H, W)
    intrinsics = data["intrinsics"]  # (N, 3, 3)
    extrinsics = data["extrinsics"]  # (N, 4, 4)
    conda_mask_6 = data["conda_mask_6"]  # (N,)

    n_total = images.shape[0]
    _, _, H, W = images.shape

    print(f"  Total frames: {n_total}, Resolution: {W}x{H}")
    print(f"  Context views (anchor): {int(conda_mask_6.sum())}")
    print(f"  Target views: {n_total - int(conda_mask_6.sum())}")

    # Save anchor views (context images, first 6)
    anchor_dir = os.path.join(output_dir, "anchor_views")
    os.makedirs(anchor_dir, exist_ok=True)

    anchor_indices = torch.where(conda_mask_6 > 0)[0]
    for i, idx in enumerate(anchor_indices):
        img = images[idx].permute(1, 2, 0).clamp(0, 1).numpy()
        img_uint8 = (img * 255).astype(np.uint8)
        from PIL import Image
        Image.fromarray(img_uint8).save(os.path.join(anchor_dir, f"view_{i:03d}.png"))

    print(f"  Saved {len(anchor_indices)} anchor views to {anchor_dir}")

    # Save anchor cameras
    cam_dir = os.path.join(output_dir, "anchor_cameras")
    os.makedirs(cam_dir, exist_ok=True)

    cameras = []
    for i, idx in enumerate(anchor_indices):
        cameras.append({
            "id": i,
            "extrinsic": extrinsics[idx].numpy().tolist(),
            "intrinsic": intrinsics[idx].numpy().tolist(),
            "width": W,
            "height": H,
        })

    with open(os.path.join(cam_dir, "cameras.json"), "w") as f:
        json.dump({"poses": cameras}, f, indent=2)

    print(f"  Saved anchor cameras to {cam_dir}/cameras.json")

    # Save all target views as scaffold renders (for Stage C input)
    target_indices = torch.where(conda_mask_6 == 0)[0]
    scaffold_render_dir = os.path.join(output_dir, "scaffold_renders")
    os.makedirs(scaffold_render_dir, exist_ok=True)

    for i, idx in enumerate(target_indices):
        img = images[idx].permute(1, 2, 0).clamp(0, 1).numpy()
        img_uint8 = (img * 255).astype(np.uint8)
        from PIL import Image
        Image.fromarray(img_uint8).save(
            os.path.join(scaffold_render_dir, f"render_{i:03d}.png")
        )

    print(f"  Saved {len(target_indices)} scaffold renders to {scaffold_render_dir}")

    # Save all extrinsics/intrinsics for target views
    target_cameras = []
    for i, idx in enumerate(target_indices):
        target_cameras.append({
            "id": i,
            "extrinsic": extrinsics[idx].numpy().tolist(),
            "intrinsic": intrinsics[idx].numpy().tolist(),
            "width": W,
            "height": H,
        })

    with open(os.path.join(output_dir, "target_cameras.json"), "w") as f:
        json.dump({"poses": target_cameras}, f, indent=2)

    print(f"  Saved {len(target_cameras)} target cameras")

    return {
        "n_anchor": len(anchor_indices),
        "n_target": len(target_indices),
        "resolution": (W, H),
    }


def extract_real_gaussians(data_pth_path: str, output_ply_path: str) -> int | None:
    """Extract real Gaussian parameters from data.pth if available.

    One2Scene computes real covariances, harmonics, and opacities but storePly()
    only saves XYZ+RGB. When the patched model_wrapper saves full Gaussian state,
    this function uses decompose_covariance() to recover scales and rotations,
    producing a much better scaffold initialization.

    Returns number of Gaussians saved, or None if real params not available.
    """
    data = torch.load(data_pth_path, map_location="cpu", weights_only=False)

    # Check if real Gaussian params were saved
    if "gaussian_covariances" not in data:
        return None

    print("Extracting REAL Gaussian parameters from data.pth...")

    means = data["gaussian_means"]          # (N, 3)
    covariances = data["gaussian_covariances"]  # (N, 3, 3)
    harmonics = data["gaussian_harmonics"]  # (N, 3, d_sh)
    opacities_raw = data["gaussian_opacities"]  # (N,)

    n = means.shape[0]
    print(f"  Found {n:,} Gaussians with real parameters")

    # 1. Positions
    positions = means.numpy().astype(np.float32)

    # 2. Decompose covariances -> log-scales + quaternions
    log_scales, rotations = decompose_covariance(covariances)
    log_scales = log_scales.numpy().astype(np.float32)
    rotations = rotations.numpy().astype(np.float32)

    # Report scale statistics
    actual_scales = np.exp(log_scales)
    print(f"  Scales: min={actual_scales.min():.6f}, median={np.median(actual_scales):.6f}, "
          f"max={actual_scales.max():.6f}")

    # 3. SH coefficients from harmonics (N, 3, d_sh)
    # harmonics[:, :, 0] is DC component -> sh_dc (N, 3)
    # harmonics[:, :, 1:] are higher-order -> sh_rest
    C0 = 0.28209479177387814
    d_sh = harmonics.shape[2]

    # One2Scene harmonics are in a different convention than 3DGS PLY storage.
    # The DC component (band 0) maps to color as: color = h[:, :, 0] * C0 + 0.5
    # 3DGS PLY stores sh_dc as (color - 0.5) / C0 = h[:, :, 0]
    sh_dc = harmonics[:, :, 0].numpy().astype(np.float32)  # (N, 3)

    if d_sh > 1:
        # Higher-order SH: harmonics is (N, 3, d_sh), rest is (N, 3, d_sh-1)
        # PLY stores sh_rest interleaved as (N, (d_sh-1)*3)
        sh_higher = harmonics[:, :, 1:]  # (N, 3, d_sh-1)
        n_rest_per_channel = d_sh - 1
        # Interleave: for each SH index, store R, G, B
        sh_rest = sh_higher.permute(0, 2, 1).reshape(n, n_rest_per_channel * 3)
        sh_rest = sh_rest.numpy().astype(np.float32)
        print(f"  SH degree: {int(np.sqrt(d_sh)) - 1} ({d_sh} coefficients, "
              f"{n_rest_per_channel} higher-order per channel)")
    else:
        sh_rest = np.zeros((n, 0), dtype=np.float32)
        print(f"  SH degree: 0 (DC only)")

    # 4. Opacities: convert to logit space (inverse sigmoid)
    # Clamp to avoid inf in logit
    opacities_clamped = opacities_raw.clamp(1e-4, 1.0 - 1e-4)
    opacities_logit = torch.log(opacities_clamped / (1.0 - opacities_clamped))
    opacities_np = opacities_logit.numpy().astype(np.float32).reshape(-1, 1)

    # Report opacity statistics
    opacities_activated = torch.sigmoid(opacities_logit)
    print(f"  Opacities: min={opacities_activated.min():.4f}, "
          f"median={opacities_activated.median():.4f}, "
          f"max={opacities_activated.max():.4f}, "
          f">{0.5:.0%}: {(opacities_activated > 0.5).float().mean():.1%}")

    # Report rotation statistics
    rot_norms = np.linalg.norm(rotations, axis=1)
    print(f"  Quaternion norms: min={rot_norms.min():.4f}, "
          f"mean={rot_norms.mean():.4f}, max={rot_norms.max():.4f}")

    gaussians = {
        "positions": positions,
        "scales": log_scales,
        "rotations": rotations,
        "opacities": opacities_np,
        "sh_dc": sh_dc,
        "sh_rest": sh_rest,
    }

    save_ply(output_ply_path, gaussians)
    print(f"  Saved {n:,} real Gaussians to {output_ply_path}")
    return n


def _compute_knn_scales(positions: np.ndarray, k: int = 3) -> np.ndarray:
    """Compute initial scales from K-nearest-neighbor distances.

    This is the standard 3DGS initialization: each Gaussian's scale is set to
    the mean distance to its K nearest neighbors. This gives geometrically
    plausible starting sizes instead of arbitrary uniform values.

    Args:
        positions: (N, 3) point positions.
        k: Number of nearest neighbors.

    Returns:
        (N, 3) log-space scales (isotropic, same value on all 3 axes).
    """
    from scipy.spatial import KDTree

    print(f"  Computing K-nearest-neighbor scales (k={k}) on {len(positions):,} points...")
    tree = KDTree(positions)
    dists, _ = tree.query(positions, k=k + 1)  # +1 because first neighbor is self
    nn_dists = dists[:, 1:k + 1].mean(axis=1)  # (N,) mean distance to k neighbors

    # Clamp to reasonable range
    nn_dists = np.clip(nn_dists, 1e-6, 10.0).astype(np.float32)
    log_scales = np.log(nn_dists)

    print(f"    Scale range: [{np.exp(log_scales.min()):.6f}, {np.exp(log_scales.max()):.6f}]")
    print(f"    Scale median: {np.exp(np.median(log_scales)):.6f}")

    return np.stack([log_scales] * 3, axis=-1)  # (N, 3) isotropic


def convert_fused_ply_to_3dgs(fused_ply_path: str, output_ply_path: str):
    """Convert One2Scene's fused.ply (XYZ+RGB) to a minimal 3DGS PLY.

    Fallback path when real Gaussian parameters are not available in data.pth.
    Uses KNN-based scale initialization (standard 3DGS approach) instead of
    arbitrary uniform scales.
    """
    from plyfile import PlyData

    print(f"Converting fused.ply to 3DGS format...")
    plydata = PlyData.read(fused_ply_path)
    vertex = plydata["vertex"]
    n = vertex.count

    positions = np.stack(
        [vertex["x"], vertex["y"], vertex["z"]], axis=-1
    ).astype(np.float32)

    # Extract RGB and convert to SH DC (c0 = (color - 0.5) / 0.2820948)
    # The SH0 -> color mapping is: color = SH0 * 0.2820948 + 0.5
    C0 = 0.28209479177387814
    rgb = np.stack(
        [vertex["red"], vertex["green"], vertex["blue"]], axis=-1
    ).astype(np.float32) / 255.0
    sh_dc = (rgb - 0.5) / C0

    # Scales from K-nearest-neighbor distances (standard 3DGS initialization)
    log_scale = _compute_knn_scales(positions, k=3)

    # Default rotations (identity quaternion wxyz)
    rotations = np.zeros((n, 4), dtype=np.float32)
    rotations[:, 0] = 1.0  # w = 1

    # Default opacities (inverse_sigmoid(0.5) = 0.0)
    opacities = np.zeros((n, 1), dtype=np.float32)

    # No higher-order SH
    sh_rest = np.zeros((n, 0), dtype=np.float32)

    gaussians = {
        "positions": positions,
        "scales": log_scale,
        "rotations": rotations,
        "opacities": opacities,
        "sh_dc": sh_dc,
        "sh_rest": sh_rest,
    }

    save_ply(output_ply_path, gaussians)
    print(f"  Saved {n:,} Gaussians to {output_ply_path}")
    return n


def render_depths_from_scaffold(
    scaffold_ply_path: str,
    cameras: list[dict],
    output_dir: str,
) -> int:
    """Render depth maps from scaffold PLY at anchor camera positions.

    Uses gsplat to rasterize the scaffold Gaussians in depth mode at each
    anchor camera.  Even with fabricated parameters (isotropic scales),
    the positions define scene geometry well enough for depth regularization.

    Returns the number of depth maps rendered (0 if gsplat unavailable).
    """
    try:
        import torch
        import torch.nn.functional as F
        from gsplat import rasterization
    except ImportError as e:
        print(f"  Skipping depth rendering ({e})")
        return 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("  Skipping depth rendering (no CUDA)")
        return 0

    from src.utils.ply_io import load_ply

    data = load_ply(scaffold_ply_path)

    # Prepare activated tensors
    positions = torch.tensor(data["positions"], dtype=torch.float32, device=device)
    rotations = F.normalize(
        torch.tensor(data["rotations"], dtype=torch.float32, device=device),
        p=2, dim=-1,
    )
    scales = torch.exp(
        torch.tensor(data["scales"], dtype=torch.float32, device=device)
    )
    opacities = torch.sigmoid(
        torch.tensor(data["opacities"], dtype=torch.float32, device=device).squeeze(-1)
    )
    # SH: DC only for depth rendering
    sh_dc = torch.tensor(
        data["sh_dc"], dtype=torch.float32, device=device
    ).unsqueeze(1)  # (N, 1, 3)

    depth_dir = os.path.join(output_dir, "anchor_depths")
    os.makedirs(depth_dir, exist_ok=True)

    n_rendered = 0
    with torch.no_grad():
        for cam in cameras:
            cam_id = cam["id"]
            viewmat = torch.tensor(
                cam["extrinsic"], dtype=torch.float32, device=device
            )
            K = torch.tensor(
                cam["intrinsic"], dtype=torch.float32, device=device
            )
            w, h = cam["width"], cam["height"]

            # Convert normalized intrinsics to pixel space
            if K[0, 0].item() < 10.0:
                K = K.clone()
                K[0, 0] *= w
                K[0, 2] *= w
                K[1, 1] *= h
                K[1, 2] *= h

            renders, _, _ = rasterization(
                means=positions,
                quats=rotations,
                scales=scales,
                opacities=opacities,
                colors=sh_dc,
                viewmats=viewmat.unsqueeze(0),
                Ks=K.unsqueeze(0),
                width=w,
                height=h,
                sh_degree=0,
                render_mode="RGB+ED",
            )

            depth = renders[0, :, :, 3].cpu().numpy()  # (H, W)
            np.save(
                os.path.join(depth_dir, f"depth_{cam_id:03d}.npy"), depth
            )
            n_rendered += 1

    print(f"  Rendered {n_rendered} scaffold depth maps to {depth_dir}")
    return n_rendered


def main():
    parser = argparse.ArgumentParser(
        description="Extract scaffold data from One2Scene output."
    )
    parser.add_argument(
        "--data_pth", type=str, required=True,
        help="Path to data.pth from Stage A output",
    )
    parser.add_argument(
        "--fused_ply", type=str, default=None,
        help="Path to fused.ply from Stage A (optional, auto-detected from data_pth dir)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="pipeline_output/stage_a",
        help="Output directory",
    )
    parser.add_argument(
        "--force_fabricated", action="store_true",
        help="Force fabricated params even if real Gaussian params exist in data.pth",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Extract cameras and images from data.pth
    info = extract_from_data_pth(args.data_pth, args.output_dir)

    # Convert to standard 3DGS format.
    # Prefer real Gaussian params from data.pth, fall back to fused.ply.
    scaffold_ply_path = os.path.join(args.output_dir, "scaffold_gaussians.ply")
    if args.force_fabricated:
        print("--force_fabricated: skipping real Gaussian extraction")
        n = None
    else:
        n = extract_real_gaussians(args.data_pth, scaffold_ply_path)

    if n is not None:
        info["n_gaussians"] = n
        info["scaffold_source"] = "real_params"
    else:
        # Fallback: fused.ply with fabricated params
        fused_ply = args.fused_ply
        if fused_ply is None:
            candidate = os.path.join(os.path.dirname(args.data_pth), "fused.ply")
            if os.path.exists(candidate):
                fused_ply = candidate

        if fused_ply and os.path.exists(fused_ply):
            n = convert_fused_ply_to_3dgs(fused_ply, scaffold_ply_path)
            info["n_gaussians"] = n
            info["scaffold_source"] = "fabricated"
        else:
            print(f"WARNING: No fused.ply found. Scaffold PLY not created.")
            print(f"  Checked: {fused_ply or 'auto-detect from data_pth dir'}")

    # Render depth maps from scaffold at anchor cameras
    if os.path.exists(scaffold_ply_path):
        anchor_cameras_path = os.path.join(
            args.output_dir, "anchor_cameras", "cameras.json"
        )
        if os.path.exists(anchor_cameras_path):
            print("Rendering scaffold depth maps at anchor cameras...")
            with open(anchor_cameras_path) as f:
                cam_data = json.load(f)
            n_depths = render_depths_from_scaffold(
                scaffold_ply_path,
                cam_data.get("poses", cam_data),
                args.output_dir,
            )
            info["n_depth_maps"] = n_depths

    # Save extraction info
    info_path = os.path.join(args.output_dir, "extraction_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nExtraction complete. Info saved to {info_path}")
    print(f"  Resolution: {info['resolution'][0]}x{info['resolution'][1]}")
    print(f"  Anchor views: {info['n_anchor']}")
    print(f"  Target views: {info['n_target']}")


if __name__ == "__main__":
    main()
