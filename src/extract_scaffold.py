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


def convert_fused_ply_to_3dgs(fused_ply_path: str, output_ply_path: str):
    """Convert One2Scene's fused.ply (XYZ+RGB) to a minimal 3DGS PLY.

    Since fused.ply only has positions and colors (no scales, rotations, or
    SH coefficients), we initialize with default values suitable for scaffold
    training initialization.
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

    # Default scales (small, log-space)
    log_scale = np.log(np.full((n, 3), 0.01, dtype=np.float32))

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
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Extract cameras and images from data.pth
    info = extract_from_data_pth(args.data_pth, args.output_dir)

    # Convert fused.ply to standard 3DGS format
    fused_ply = args.fused_ply
    if fused_ply is None:
        # Auto-detect: fused.ply is in same directory as data.pth
        candidate = os.path.join(os.path.dirname(args.data_pth), "fused.ply")
        if os.path.exists(candidate):
            fused_ply = candidate

    if fused_ply and os.path.exists(fused_ply):
        scaffold_ply_path = os.path.join(args.output_dir, "scaffold_gaussians.ply")
        n = convert_fused_ply_to_3dgs(fused_ply, scaffold_ply_path)
        info["n_gaussians"] = n

        # Render depth maps from scaffold at anchor cameras
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
    else:
        print(f"WARNING: No fused.ply found. Scaffold PLY not created.")
        print(f"  Checked: {fused_ply or 'auto-detect from data_pth dir'}")

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
