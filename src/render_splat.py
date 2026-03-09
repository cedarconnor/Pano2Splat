"""Render all views from a trained Gaussian splat.

Used by Stage E (upscale refinement) to produce training-view renders
that are then super-resolved and fed back into refinement training.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm

from src.utils.camera import load_cameras_json
from src.utils.ply_io import load_ply


def render_gaussians(
    gaussians: dict[str, torch.Tensor],
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    sh_degree: int = 3,
) -> torch.Tensor:
    """Render a single view of the Gaussian splat.

    Args:
        gaussians: Dict of tensors with keys: positions, rotations,
                   scales, opacities, sh_coefficients.
        viewmat: (4, 4) world-to-camera extrinsic matrix.
        K: (3, 3) camera intrinsic matrix.
        width: Image width in pixels.
        height: Image height in pixels.
        sh_degree: Spherical harmonics degree.

    Returns:
        (H, W, 3) float32 tensor in [0, 1].
    """
    from gsplat import rasterization

    # Detect and convert normalized intrinsics to pixel space.
    if K[0, 0].item() < 10.0:
        K = K.clone()
        K[0, 0] *= width
        K[0, 2] *= width
        K[1, 1] *= height
        K[1, 2] *= height

    # Apply activation functions: PLY stores log-scales, opacity logits,
    # and unnormalized quaternions. gsplat expects actual values.
    quats = F.normalize(gaussians["rotations"], p=2, dim=-1)
    scales = torch.exp(gaussians["scales"])
    opacities = torch.sigmoid(gaussians["opacities"])

    renders, alphas, info = rasterization(
        means=gaussians["positions"],
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=gaussians["sh_coefficients"],
        viewmats=viewmat.unsqueeze(0),
        Ks=K.unsqueeze(0),
        width=width,
        height=height,
        sh_degree=sh_degree,
    )
    # renders shape: (1, H, W, 3)
    return renders[0].clamp(0.0, 1.0)


def load_gaussians_to_device(ply_path: str, device: str) -> dict[str, torch.Tensor]:
    """Load .ply and convert all arrays to GPU tensors.

    Combines sh_dc and sh_rest into a single sh_coefficients tensor
    shaped (N, K, 3) as expected by gsplat.

    Returns:
        Dict with keys: positions, rotations, scales, opacities,
        sh_coefficients.
    """
    data = load_ply(ply_path)

    # SH coefficients: combine DC (N, 3) and rest (N, R) into (N, K, 3)
    sh_dc = torch.from_numpy(data["sh_dc"]).to(device)       # (N, 3)
    sh_rest = torch.from_numpy(data["sh_rest"]).to(device)    # (N, R)
    n = sh_dc.shape[0]

    # DC term as (N, 1, 3)
    sh_dc = sh_dc.unsqueeze(1)

    # Rest coefficients: reshape from (N, R) to (N, R//3, 3)
    n_rest = sh_rest.shape[1]
    if n_rest > 0:
        sh_rest = sh_rest.reshape(n, n_rest // 3, 3)
        sh_coefficients = torch.cat([sh_dc, sh_rest], dim=1)
    else:
        sh_coefficients = sh_dc

    return {
        "positions": torch.from_numpy(data["positions"]).to(device),
        "rotations": torch.from_numpy(data["rotations"]).to(device),
        "scales": torch.from_numpy(data["scales"]).to(device),
        "opacities": torch.from_numpy(data["opacities"]).squeeze(-1).to(device),
        "sh_coefficients": sh_coefficients,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render all views from a trained Gaussian splat."
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to trained .ply (e.g. pipeline_output/stage_d/final.ply)",
    )
    parser.add_argument(
        "--cameras", type=str, required=True,
        help="Path to cameras JSON (trajectory.json or training_cameras.json)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save rendered PNGs",
    )
    parser.add_argument("--sh_degree", type=int, default=3, help="SH degree")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and cameras
    print(f"Loading Gaussians from {args.model_path}")
    gaussians = load_gaussians_to_device(args.model_path, args.device)
    n_gaussians = gaussians["positions"].shape[0]
    # Auto-detect SH degree from data
    n_sh_coeffs = gaussians["sh_coefficients"].shape[1]  # (N, K, 3)
    max_sh_degree = int(n_sh_coeffs ** 0.5) - 1
    sh_degree = min(args.sh_degree, max_sh_degree)
    print(f"  {n_gaussians:,} Gaussians loaded (SH degree: {sh_degree})")

    cameras = load_cameras_json(args.cameras)
    print(f"Loaded {len(cameras)} cameras from {args.cameras}")

    # Render all views
    with torch.no_grad():
        for idx, cam in enumerate(tqdm(cameras, desc="Rendering")):
            viewmat = torch.tensor(
                cam["extrinsic"], dtype=torch.float32, device=args.device
            )
            K = torch.tensor(
                cam["intrinsic"], dtype=torch.float32, device=args.device
            )
            width = cam["width"]
            height = cam["height"]

            rendered = render_gaussians(
                gaussians, viewmat, K, width, height, sh_degree=sh_degree
            )

            # Convert to uint8 PNG
            img_np = (rendered.cpu().numpy() * 255.0).astype(np.uint8)
            img = Image.fromarray(img_np)
            img.save(output_dir / f"render_{idx:03d}.png")

    print(f"\nRendered {len(cameras)} views to {output_dir}")


if __name__ == "__main__":
    main()
