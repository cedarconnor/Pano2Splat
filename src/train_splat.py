"""Stage D: Gaussian Splat Optimization.

Trains a high-quality Gaussian splat using scaffold initialization from Stage A
and denoised view supervision from Stage C, using the gsplat library for
rasterization.

Also supports Stage E fine-tuning mode via --init_ply and --lr_scale flags.

Usage (Stage D):
    python -m src.train_splat \
        --scaffold_ply pipeline_output/stage_a/scaffold_gaussians.ply \
        --images pipeline_output/stage_c/denoised_frames/ \
        --cameras pipeline_output/stage_b/trajectory.json \
        --scaffold_depths pipeline_output/stage_a/anchor_depths/ \
        --anchor_cameras pipeline_output/stage_a/anchor_cameras/cameras.json \
        --output pipeline_output/stage_d/

Usage (Stage E fine-tune):
    python -m src.train_splat \
        --init_ply pipeline_output/stage_d/final.ply \
        --images pipeline_output/stage_e/renders_upscaled/ \
        --cameras pipeline_output/stage_d/training_cameras.json \
        --scaffold_depths pipeline_output/stage_a/anchor_depths/ \
        --anchor_cameras pipeline_output/stage_a/anchor_cameras/cameras.json \
        --output pipeline_output/stage_e/ \
        --iterations 10000 --lr_scale 0.1
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gsplat import rasterization
from pytorch_msssim import ssim as ssim_fn

from src.utils.camera import load_cameras_json, save_cameras_json
from src.utils.metrics import compute_all_metrics
from src.utils.ply_io import load_ply, save_ply


# ---------------------------------------------------------------------------
# Gaussian Model
# ---------------------------------------------------------------------------

class GaussianModel(nn.Module):
    """Container for all Gaussian splat parameters."""

    def __init__(
        self,
        positions: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
        sh_dc: torch.Tensor,
        sh_rest: torch.Tensor,
    ):
        super().__init__()
        self.positions = nn.Parameter(positions)    # (N, 3)
        self.scales = nn.Parameter(scales)          # (N, 3) log-space
        self.rotations = nn.Parameter(rotations)    # (N, 4) quaternion wxyz
        self.opacities = nn.Parameter(opacities)    # (N, 1) logit-space
        self.sh_dc = nn.Parameter(sh_dc)            # (N, 3)
        self.sh_rest = nn.Parameter(sh_rest)        # (N, K)

    # -- Construction helpers ------------------------------------------------

    @classmethod
    def from_ply(
        cls, path: str, device: str = "cuda", sh_degree: int = 0,
        max_init_points: int | None = None,
    ) -> "GaussianModel":
        """Load Gaussians from a standard 3DGS .ply file.

        If the PLY has fewer SH coefficients than needed for sh_degree,
        the extra coefficients are zero-initialized.

        If max_init_points is set and the PLY has more points, a random
        subset is used.  This prevents training collapse when starting from
        a very dense point cloud with fabricated parameters.
        """
        data = load_ply(path)
        n = data["positions"].shape[0]

        if max_init_points is not None and n > max_init_points:
            rng = np.random.default_rng(42)
            indices = rng.choice(n, max_init_points, replace=False)
            data = {k: v[indices] for k, v in data.items()}
            print(f"  Subsampled to {max_init_points:,} points (from {n:,})")
            n = max_init_points
        sh_rest = torch.tensor(data["sh_rest"], dtype=torch.float32, device=device)

        # Expand SH rest if we need higher degree than what's in the PLY
        n_rest_needed = ((sh_degree + 1) ** 2 - 1) * 3
        n_rest_have = sh_rest.shape[1]
        if n_rest_needed > n_rest_have:
            padding = torch.zeros(n, n_rest_needed - n_rest_have, dtype=torch.float32, device=device)
            sh_rest = torch.cat([sh_rest, padding], dim=1)

        return cls(
            positions=torch.tensor(data["positions"], dtype=torch.float32, device=device),
            scales=torch.tensor(data["scales"], dtype=torch.float32, device=device),
            rotations=torch.tensor(data["rotations"], dtype=torch.float32, device=device),
            opacities=torch.tensor(data["opacities"], dtype=torch.float32, device=device),
            sh_dc=torch.tensor(data["sh_dc"], dtype=torch.float32, device=device),
            sh_rest=sh_rest,
        )

    def to_ply(self, path: str) -> None:
        """Save current Gaussians to a standard 3DGS .ply file."""
        save_ply(path, {
            "positions": self.positions.detach().cpu().numpy(),
            "scales": self.scales.detach().cpu().numpy(),
            "rotations": self.rotations.detach().cpu().numpy(),
            "opacities": self.opacities.detach().cpu().numpy(),
            "sh_dc": self.sh_dc.detach().cpu().numpy(),
            "sh_rest": self.sh_rest.detach().cpu().numpy(),
        })

    # -- Properties ----------------------------------------------------------

    @property
    def n_gaussians(self) -> int:
        return self.positions.shape[0]

    def get_activated_opacities(self) -> torch.Tensor:
        """Return opacities in [0, 1] via sigmoid."""
        return torch.sigmoid(self.opacities)

    def get_activated_scales(self) -> torch.Tensor:
        """Return positive scales via exp."""
        return torch.exp(self.scales)

    # -- Densification & Pruning ---------------------------------------------

    def densify_and_prune(
        self,
        grad_accum: torch.Tensor,
        grad_count: torch.Tensor,
        grad_threshold: float,
        min_opacity: float,
        max_screen_size: float = 20.0,
    ) -> None:
        """Standard 3DGS densification: clone small, split large, prune dim.

        Args:
            grad_accum: Accumulated viewspace position gradients (N, 2).
            grad_count: Number of observations per Gaussian (N,).
            grad_threshold: Gradient magnitude threshold for densification.
            min_opacity: Prune Gaussians with activated opacity below this.
            max_screen_size: (unused, reserved for screen-space pruning).
        """
        device = self.positions.device
        avg_grad = grad_accum / grad_count.clamp(min=1).unsqueeze(-1)
        grad_norm = avg_grad.norm(dim=-1)  # (N,)

        # Identify candidates for densification
        large_grad_mask = grad_norm >= grad_threshold

        # Small Gaussians: clone.  Large Gaussians: split.
        # "Small" = scale norm below the scene median.
        scale_norm = self.get_activated_scales().norm(dim=-1)
        scale_threshold = scale_norm.median().item()

        clone_mask = large_grad_mask & (scale_norm <= scale_threshold)
        split_mask = large_grad_mask & (scale_norm > scale_threshold)

        # --- Clone: duplicate with same parameters ---
        new_positions = [self.positions.data]
        new_scales = [self.scales.data]
        new_rotations = [self.rotations.data]
        new_opacities = [self.opacities.data]
        new_sh_dc = [self.sh_dc.data]
        new_sh_rest = [self.sh_rest.data]

        if clone_mask.any():
            new_positions.append(self.positions.data[clone_mask])
            new_scales.append(self.scales.data[clone_mask])
            new_rotations.append(self.rotations.data[clone_mask])
            new_opacities.append(self.opacities.data[clone_mask])
            new_sh_dc.append(self.sh_dc.data[clone_mask])
            new_sh_rest.append(self.sh_rest.data[clone_mask])

        # --- Split: create 2 children offset along principal axis, halve scale ---
        if split_mask.any():
            n_split = split_mask.sum().item()
            # Two copies per split Gaussian
            for _ in range(2):
                stds = self.get_activated_scales()[split_mask]  # (M, 3)
                offsets = torch.randn(n_split, 3, device=device) * stds
                new_positions.append(self.positions.data[split_mask] + offsets)
                # Halve scale in log-space => subtract ln(2)
                new_scales.append(self.scales.data[split_mask] - math.log(2.0))
                new_rotations.append(self.rotations.data[split_mask])
                new_opacities.append(self.opacities.data[split_mask])
                new_sh_dc.append(self.sh_dc.data[split_mask])
                new_sh_rest.append(self.sh_rest.data[split_mask])

        # --- Prune: remove low-opacity and original split Gaussians ---
        prune_mask = (self.get_activated_opacities().squeeze(-1) < min_opacity) | split_mask

        # Rebuild parameters: keep non-pruned originals + new Gaussians
        keep_mask = ~prune_mask
        kept = [
            self.positions.data[keep_mask],
            self.scales.data[keep_mask],
            self.rotations.data[keep_mask],
            self.opacities.data[keep_mask],
            self.sh_dc.data[keep_mask],
            self.sh_rest.data[keep_mask],
        ]

        # Concatenate kept originals + cloned + split children
        # (skip index 0 from new_* lists since originals are handled above)
        extra_positions = new_positions[1:] if len(new_positions) > 1 else []
        extra_scales = new_scales[1:] if len(new_scales) > 1 else []
        extra_rotations = new_rotations[1:] if len(new_rotations) > 1 else []
        extra_opacities = new_opacities[1:] if len(new_opacities) > 1 else []
        extra_sh_dc = new_sh_dc[1:] if len(new_sh_dc) > 1 else []
        extra_sh_rest = new_sh_rest[1:] if len(new_sh_rest) > 1 else []

        final_positions = torch.cat([kept[0]] + extra_positions, dim=0)
        final_scales = torch.cat([kept[1]] + extra_scales, dim=0)
        final_rotations = torch.cat([kept[2]] + extra_rotations, dim=0)
        final_opacities = torch.cat([kept[3]] + extra_opacities, dim=0)
        final_sh_dc = torch.cat([kept[4]] + extra_sh_dc, dim=0)
        final_sh_rest = torch.cat([kept[5]] + extra_sh_rest, dim=0)

        # Update in-place
        self.positions = nn.Parameter(final_positions)
        self.scales = nn.Parameter(final_scales)
        self.rotations = nn.Parameter(final_rotations)
        self.opacities = nn.Parameter(final_opacities)
        self.sh_dc = nn.Parameter(final_sh_dc)
        self.sh_rest = nn.Parameter(final_sh_rest)

    def reset_opacities(self, value: float = 0.01) -> None:
        """Reset all opacities to inverse_sigmoid(value)."""
        logit_val = math.log(value / (1.0 - value))
        self.opacities.data.fill_(logit_val)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_gaussians(
    model: GaussianModel,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    sh_degree: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Render Gaussians to an image using gsplat.

    Args:
        model: GaussianModel instance.
        viewmat: (4, 4) world-to-camera matrix.
        K: (3, 3) camera intrinsic matrix.
        width: Image width in pixels.
        height: Image height in pixels.
        sh_degree: Spherical harmonics degree.

    Returns:
        rendered_image: (3, H, W) tensor in [0, 1].
        rendered_depth: (1, H, W) tensor.
        info: dict with 'means2d' for gradient accumulation.
    """
    device = model.positions.device

    # Detect and convert normalized intrinsics to pixel space.
    # One2Scene stores intrinsics normalized by image dimensions (fx/W, cx/W, etc.)
    # gsplat expects pixel-space intrinsics.
    if K[0, 0].item() < 10.0:  # clearly normalized if fx < 10 pixels
        K = K.clone()
        K[0, 0] *= width
        K[0, 2] *= width
        K[1, 1] *= height
        K[1, 2] *= height

    # Prepare SH coefficients: gsplat expects (N, K, 3) where K = (degree+1)^2
    # sh_dc is (N, 3) -> (N, 1, 3); sh_rest is (N, M) where M = (K-1)*3
    n = model.n_gaussians
    n_sh_rest_total = model.sh_rest.shape[1]
    n_sh_rest_per_channel = n_sh_rest_total // 3 if n_sh_rest_total > 0 else 0
    n_sh_coeffs = 1 + n_sh_rest_per_channel  # total per channel

    # Clamp sh_degree to what we actually have
    max_degree_from_data = int(math.sqrt(n_sh_coeffs)) - 1
    effective_degree = min(sh_degree, max_degree_from_data)

    # Build colors tensor: (N, n_sh_coeffs, 3)
    sh_dc_expanded = model.sh_dc.unsqueeze(1)  # (N, 1, 3)
    if n_sh_rest_total > 0:
        # sh_rest is stored flat: (N, n_rest_per_channel * 3)
        # Reshape to (N, n_rest_per_channel, 3)
        sh_rest_3d = model.sh_rest.reshape(n, n_sh_rest_per_channel, 3)
        colors = torch.cat([sh_dc_expanded, sh_rest_3d], dim=1)  # (N, K, 3)
    else:
        colors = sh_dc_expanded  # (N, 1, 3)

    # Activated values
    opacities = model.get_activated_opacities().squeeze(-1)  # (N,)
    scales = model.get_activated_scales()  # (N, 3)

    # Normalize quaternions
    quats = F.normalize(model.rotations, p=2, dim=-1)  # (N, 4)

    renders, alphas, info = rasterization(
        means=model.positions,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat.unsqueeze(0),  # (1, 4, 4)
        Ks=K.unsqueeze(0),              # (1, 3, 3)
        width=width,
        height=height,
        sh_degree=effective_degree,
        render_mode="RGB+ED",
    )
    # renders: (1, H, W, 4) — RGB + expected depth
    # alphas: (1, H, W, 1)

    rendered = renders[0]  # (H, W, 4)
    rgb = rendered[..., :3].permute(2, 0, 1).clamp(0.0, 1.0)  # (3, H, W)
    depth = rendered[..., 3:4].permute(2, 0, 1)                # (1, H, W)

    return rgb, depth, info


# ---------------------------------------------------------------------------
# Depth Loss & Scale Computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_global_scale(
    model: GaussianModel,
    anchor_cameras: list[dict],
    scaffold_depths: dict[int, np.ndarray],
    sh_degree: int,
    device: str,
) -> float:
    """Compute a single median-ratio scale factor across all anchor views.

    Renders depth at each anchor camera from the current model and compares
    to the scaffold depth maps.

    Returns:
        Global scale factor (float). Multiply rendered depth by this to
        match scaffold depth scale.
    """
    all_rendered = []
    all_scaffold = []

    for cam in anchor_cameras:
        cam_id = cam["id"]
        if cam_id not in scaffold_depths:
            continue

        viewmat = torch.tensor(cam["extrinsic"], dtype=torch.float32, device=device)
        K = torch.tensor(cam["intrinsic"], dtype=torch.float32, device=device)
        w, h = cam["width"], cam["height"]

        _, depth, _ = render_gaussians(model, viewmat, K, w, h, sh_degree)
        rendered_depth = depth.squeeze(0).squeeze(0)  # (H, W)

        scaffold_depth = torch.tensor(
            scaffold_depths[cam_id], dtype=torch.float32, device=device
        )

        # Resize if shapes don't match
        if scaffold_depth.shape != rendered_depth.shape:
            scaffold_depth = F.interpolate(
                scaffold_depth.unsqueeze(0).unsqueeze(0),
                size=rendered_depth.shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

        valid = (scaffold_depth > 0) & (rendered_depth > 0)
        if valid.sum() < 100:
            continue

        all_rendered.append(rendered_depth[valid])
        all_scaffold.append(scaffold_depth[valid])

    if not all_rendered:
        print("WARNING: No valid depth pairs found. Using global_scale=1.0")
        return 1.0

    all_rendered_cat = torch.cat(all_rendered)
    all_scaffold_cat = torch.cat(all_scaffold)

    scale = (torch.median(all_scaffold_cat) / torch.median(all_rendered_cat)).item()
    return scale


def depth_loss_fn(
    rendered_depth: torch.Tensor,
    scaffold_depth: torch.Tensor,
    global_scale: float,
) -> torch.Tensor:
    """L1 depth loss with global scale alignment.

    Args:
        rendered_depth: (1, H, W) rendered depth.
        scaffold_depth: (H, W) scaffold depth (numpy-loaded, on same device).
        global_scale: Precomputed global scale factor.

    Returns:
        Scalar loss tensor.
    """
    rd = rendered_depth.squeeze(0)  # (H, W)

    # Resize scaffold to match rendered if needed
    if scaffold_depth.shape != rd.shape:
        scaffold_depth = F.interpolate(
            scaffold_depth.unsqueeze(0).unsqueeze(0),
            size=rd.shape,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)

    valid = scaffold_depth > 0
    if valid.sum() < 10:
        return torch.tensor(0.0, device=rd.device)

    aligned = rd * global_scale
    return F.l1_loss(aligned[valid], scaffold_depth[valid])


# ---------------------------------------------------------------------------
# Learning Rate Scheduling
# ---------------------------------------------------------------------------

def get_position_lr(
    iteration: int,
    lr_init: float,
    lr_final: float,
    lr_delay_mult: float,
    max_steps: int,
) -> float:
    """Exponential decay learning rate for position parameters."""
    if lr_init == 0.0 or lr_final == 0.0:
        return 0.0
    if lr_delay_mult > 0:
        delay_rate = lr_delay_mult + (1 - lr_delay_mult) * min(
            iteration / max(max_steps, 1), 1.0
        )
    else:
        delay_rate = 1.0
    t = min(iteration / max(max_steps, 1), 1.0)
    log_lerp = math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)
    return delay_rate * log_lerp


def update_learning_rate(optimizer: torch.optim.Optimizer, iteration: int, args) -> None:
    """Update the position learning rate via exponential decay schedule."""
    new_lr = get_position_lr(
        iteration,
        args.position_lr_init * args.lr_scale,
        args.position_lr_final * args.lr_scale,
        args.position_lr_delay_mult,
        args.position_lr_max_steps,
    )
    for param_group in optimizer.param_groups:
        if param_group["name"] == "positions":
            param_group["lr"] = new_lr


# ---------------------------------------------------------------------------
# Optimizer Construction
# ---------------------------------------------------------------------------

def build_optimizer(model: GaussianModel, args) -> torch.optim.Adam:
    """Create the Adam optimizer with per-parameter-group learning rates."""
    lr_scale = args.lr_scale
    return torch.optim.Adam(
        [
            {"params": [model.positions], "lr": args.position_lr_init * lr_scale, "name": "positions"},
            {"params": [model.sh_dc], "lr": args.feature_lr * lr_scale, "name": "sh_dc"},
            {"params": [model.sh_rest], "lr": args.feature_lr * lr_scale / 20.0, "name": "sh_rest"},
            {"params": [model.opacities], "lr": args.opacity_lr * lr_scale, "name": "opacities"},
            {"params": [model.scales], "lr": args.scaling_lr * lr_scale, "name": "scales"},
            {"params": [model.rotations], "lr": args.rotation_lr * lr_scale, "name": "rotations"},
        ],
        eps=1e-15,
    )


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_images(image_dir: str, device: str) -> list[torch.Tensor]:
    """Load all images from a directory as (C, H, W) float tensors in [0, 1].

    Images are sorted by filename to ensure consistent ordering with cameras.
    """
    from PIL import Image
    from torchvision import transforms

    image_dir = Path(image_dir)
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in extensions
    )

    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    to_tensor = transforms.ToTensor()  # converts to (C, H, W) in [0, 1]
    images = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        images.append(to_tensor(img).to(device))

    print(f"Loaded {len(images)} images from {image_dir} "
          f"(resolution: {images[0].shape[1]}x{images[0].shape[2]})")
    return images


def load_scaffold_depths(depth_dir: str) -> dict[int, np.ndarray]:
    """Load scaffold depth maps from a directory of .npy files.

    Returns dict mapping view index -> depth array (H, W).
    """
    depth_dir = Path(depth_dir)
    if not depth_dir.exists():
        print(f"WARNING: Scaffold depth directory not found: {depth_dir}")
        return {}

    depths = {}
    for p in sorted(depth_dir.glob("*.npy")):
        # Extract index from filename, e.g. depth_000.npy -> 0
        stem = p.stem
        # Try to parse trailing digits
        digits = "".join(c for c in stem if c.isdigit())
        if digits:
            idx = int(digits)
        else:
            idx = len(depths)
        depths[idx] = np.load(p).astype(np.float32)

    print(f"Loaded {len(depths)} scaffold depth maps from {depth_dir}")
    return depths


def find_depth_for_view(
    view_idx: int,
    anchor_cameras: list[dict],
    training_cameras: list[dict],
    scaffold_depths: dict[int, np.ndarray],
    cos_threshold: float = 0.95,
) -> int | None:
    """Find the anchor depth map closest to a training view.

    Matches by comparing camera forward directions. Returns the anchor camera
    ID if the cosine similarity exceeds cos_threshold, else None.
    """
    if not scaffold_depths or not anchor_cameras:
        return None

    train_cam = training_cameras[view_idx]
    train_ext = np.array(train_cam["extrinsic"])
    # Camera forward direction is the third row of rotation (negated for OpenGL)
    train_fwd = -train_ext[2, :3]
    train_fwd = train_fwd / np.linalg.norm(train_fwd)

    best_id = None
    best_cos = -1.0

    for acam in anchor_cameras:
        if acam["id"] not in scaffold_depths:
            continue
        a_ext = np.array(acam["extrinsic"])
        a_fwd = -a_ext[2, :3]
        a_fwd = a_fwd / np.linalg.norm(a_fwd)
        cos_sim = float(np.dot(train_fwd, a_fwd))
        if cos_sim > best_cos:
            best_cos = cos_sim
            best_id = acam["id"]

    if best_cos >= cos_threshold and best_id is not None:
        return best_id
    return None


# ---------------------------------------------------------------------------
# Metrics Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_metrics(
    model: GaussianModel,
    images: list[torch.Tensor],
    cameras: list[dict],
    sh_degree: int,
    device: str,
) -> dict:
    """Compute average PSNR, SSIM, LPIPS across all training views."""
    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0

    for idx in range(len(images)):
        gt = images[idx]
        cam = cameras[idx]
        viewmat = torch.tensor(cam["extrinsic"], dtype=torch.float32, device=device)
        K = torch.tensor(cam["intrinsic"], dtype=torch.float32, device=device)
        w, h = cam["width"], cam["height"]

        rendered, _, _ = render_gaussians(model, viewmat, K, w, h, sh_degree)
        metrics = compute_all_metrics(rendered, gt)
        psnr_sum += metrics["psnr"]
        ssim_sum += metrics["ssim"]
        lpips_sum += metrics["lpips"]

    n = len(images)
    return {
        "psnr": psnr_sum / n,
        "ssim": ssim_sum / n,
        "lpips": lpips_sum / n,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    """Main training loop."""
    device = args.device
    os.makedirs(args.output, exist_ok=True)

    # ---- Load model -------------------------------------------------------
    max_init = args.max_init_points if args.max_init_points > 0 else None
    if args.init_ply:
        print(f"Loading pre-trained Gaussians from {args.init_ply} (fine-tune mode)")
        model = GaussianModel.from_ply(args.init_ply, device=device, sh_degree=args.sh_degree)
    else:
        print(f"Loading scaffold Gaussians from {args.scaffold_ply}")
        model = GaussianModel.from_ply(
            args.scaffold_ply, device=device, sh_degree=args.sh_degree,
            max_init_points=max_init,
        )

    print(f"Initial Gaussian count: {model.n_gaussians:,}")

    # ---- Load training data -----------------------------------------------
    images = load_images(args.images, device=device)
    cameras = load_cameras_json(args.cameras)

    if len(images) != len(cameras):
        raise ValueError(
            f"Image count ({len(images)}) does not match camera count "
            f"({len(cameras)}). Check --images and --cameras paths."
        )

    # ---- Load scaffold depths for regularization --------------------------
    scaffold_depths = {}
    anchor_cameras = []
    if args.scaffold_depths:
        scaffold_depths = load_scaffold_depths(args.scaffold_depths)
    if args.anchor_cameras and os.path.exists(args.anchor_cameras):
        anchor_cameras = load_cameras_json(args.anchor_cameras)
        print(f"Loaded {len(anchor_cameras)} anchor cameras from {args.anchor_cameras}")

    # Precompute depth-to-view mapping for training views
    view_depth_map: dict[int, int] = {}
    for i in range(len(cameras)):
        anchor_id = find_depth_for_view(i, anchor_cameras, cameras, scaffold_depths)
        if anchor_id is not None:
            view_depth_map[i] = anchor_id

    print(f"Depth regularization available for {len(view_depth_map)}/{len(cameras)} training views")

    # ---- Compute global depth scale ---------------------------------------
    global_scale = 1.0
    if scaffold_depths and anchor_cameras:
        print("Computing global depth scale factor...")
        global_scale = compute_global_scale(
            model, anchor_cameras, scaffold_depths, args.sh_degree, device
        )
        print(f"Global depth scale: {global_scale:.4f}")

    # ---- Build optimizer ---------------------------------------------------
    optimizer = build_optimizer(model, args)

    # ---- Gradient accumulation buffers for densification -------------------
    grad_accum = torch.zeros(model.n_gaussians, 2, device=device)
    grad_count = torch.zeros(model.n_gaussians, device=device)

    # ---- Move scaffold depths to device ------------------------------------
    scaffold_depths_gpu: dict[int, torch.Tensor] = {}
    for k, v in scaffold_depths.items():
        scaffold_depths_gpu[k] = torch.tensor(v, dtype=torch.float32, device=device)

    # ---- Training loop -----------------------------------------------------
    print(f"\nStarting training: {args.iterations} iterations")
    print(f"  L1 weight: {args.lambda_l1}, SSIM weight: {args.lambda_ssim}")
    print(f"  Depth weight: {args.lambda_depth} (anneal {args.depth_anneal_start}-{args.depth_anneal_end})")
    print(f"  Densify: iter {args.densify_from_iter}-{args.densify_until_iter}, "
          f"threshold={args.densify_grad_threshold}, interval={args.densification_interval}")
    print(f"  LR scale: {args.lr_scale}")
    print()

    start_time = time.time()
    best_loss = float("inf")

    for iteration in range(1, args.iterations + 1):
        # 1. Sample random training view
        idx = random.randint(0, len(images) - 1)
        gt_image = images[idx]  # (3, H, W)
        cam = cameras[idx]

        viewmat = torch.tensor(cam["extrinsic"], dtype=torch.float32, device=device)
        K = torch.tensor(cam["intrinsic"], dtype=torch.float32, device=device)
        w, h = cam["width"], cam["height"]

        # 2. Render
        rendered_image, rendered_depth, info = render_gaussians(
            model, viewmat, K, w, h, args.sh_degree
        )

        # 3. Compute loss
        l1_loss = F.l1_loss(rendered_image, gt_image)
        ssim_val = ssim_fn(
            rendered_image.unsqueeze(0),
            gt_image.unsqueeze(0),
            data_range=1.0,
            size_average=True,
        )
        ssim_loss = 1.0 - ssim_val
        loss = args.lambda_l1 * l1_loss + args.lambda_ssim * ssim_loss

        # Depth regularization (annealed)
        if (
            idx in view_depth_map
            and iteration < args.depth_anneal_end
            and args.lambda_depth > 0
        ):
            anchor_id = view_depth_map[idx]
            sd = scaffold_depths_gpu[anchor_id]

            anneal_range = args.depth_anneal_end - args.depth_anneal_start
            if anneal_range > 0:
                progress = max(0, iteration - args.depth_anneal_start) / anneal_range
                anneal_factor = 1.0 - min(progress, 1.0)
            else:
                anneal_factor = 1.0

            d_loss = depth_loss_fn(rendered_depth, sd, global_scale)
            loss = loss + args.lambda_depth * anneal_factor * d_loss

        # 4. Retain grad on means2d for densification, then backward
        in_densify_range = (
            args.densify_from_iter <= iteration <= args.densify_until_iter
        )
        if in_densify_range and "means2d" in info:
            info["means2d"].retain_grad()
        loss.backward()

        # 5. Accumulate viewspace gradients for densification
        if in_densify_range and "means2d" in info:
            means2d = info["means2d"]  # (1, N, 2) or (N, 2)
            if means2d.grad is not None:
                g = means2d.grad
                if g.ndim == 3:
                    g = g.squeeze(0)  # (N, 2)
                # Handle size mismatch after previous densification
                n_current = model.n_gaussians
                if g.shape[0] == n_current:
                    grad_accum[:n_current] += g.detach()
                    grad_count[:n_current] += 1

        # 6. Optimizer step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # 7. Densification
        if in_densify_range and iteration % args.densification_interval == 0:
            model.densify_and_prune(
                grad_accum,
                grad_count,
                args.densify_grad_threshold,
                args.min_opacity,
            )
            # Reset buffers and rebuild optimizer for new parameters
            n_new = model.n_gaussians
            grad_accum = torch.zeros(n_new, 2, device=device)
            grad_count = torch.zeros(n_new, device=device)
            optimizer = build_optimizer(model, args)
            # Re-apply current LR schedule for position
            update_learning_rate(optimizer, iteration, args)

        # 8. Opacity reset
        if iteration % args.opacity_reset_interval == 0:
            model.reset_opacities()

        # 9. Learning rate scheduling
        update_learning_rate(optimizer, iteration, args)

        # 10. Logging
        cur_loss = loss.item()
        if cur_loss < best_loss:
            best_loss = cur_loss

        if iteration % 500 == 0 or iteration == 1:
            elapsed = time.time() - start_time
            print(
                f"[Iter {iteration:>6d}/{args.iterations}] "
                f"Loss: {cur_loss:.4f}  L1: {l1_loss.item():.4f}  "
                f"SSIM: {ssim_loss.item():.4f}  "
                f"Gaussians: {model.n_gaussians:,}  "
                f"Time: {elapsed:.0f}s"
            )

        # 11. Gaussian count warning
        if model.n_gaussians > 3_000_000 and iteration < 20000:
            if iteration % 1000 == 0:
                print(
                    f"WARNING: Gaussian count {model.n_gaussians:,} exceeds 3M "
                    f"before iter 20K. Consider raising densify_grad_threshold."
                )

        # 12. Checkpointing
        if iteration % args.checkpoint_interval == 0:
            ckpt_path = os.path.join(args.output, f"checkpoint_{iteration}.ply")
            model.to_ply(ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # ---- Finalize ----------------------------------------------------------
    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"Final Gaussian count: {model.n_gaussians:,}")

    # Save final model
    final_path = os.path.join(args.output, "final.ply")
    model.to_ply(final_path)
    print(f"Saved final model: {final_path}")

    # Copy cameras to output directory
    cameras_out = os.path.join(args.output, "training_cameras.json")
    save_cameras_json(cameras, cameras_out)
    print(f"Saved cameras: {cameras_out}")

    # Compute final metrics
    print("Computing final metrics on all training views...")
    metrics = evaluate_metrics(model, images, cameras, args.sh_degree, device)
    metrics["n_gaussians"] = model.n_gaussians
    metrics["training_time_sec"] = total_time
    metrics["iterations"] = args.iterations

    metrics_path = os.path.join(args.output, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nFinal Metrics:")
    print(f"  PSNR:  {metrics['psnr']:.2f} dB")
    print(f"  SSIM:  {metrics['ssim']:.4f}")
    print(f"  LPIPS: {metrics['lpips']:.4f}")
    print(f"Saved metrics: {metrics_path}")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage D: Train Gaussian splat from scaffold + denoised views."
    )

    # --- Paths ---
    parser.add_argument("--scaffold_ply", type=str, default=None,
                        help="Path to Stage A scaffold .ply")
    parser.add_argument("--images", type=str, required=True,
                        help="Directory of denoised frames (Stage C)")
    parser.add_argument("--cameras", type=str, required=True,
                        help="Path to trajectory.json (Stage B cameras)")
    parser.add_argument("--scaffold_depths", type=str, default=None,
                        help="Directory of anchor depth .npy files (Stage A)")
    parser.add_argument("--anchor_cameras", type=str, default=None,
                        help="Path to anchor cameras.json (Stage A)")
    parser.add_argument("--output", type=str, default="pipeline_output/stage_d",
                        help="Output directory")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to pipeline.yaml (optional, for default params)")

    # --- Training params ---
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--position_lr_init", type=float, default=0.00016)
    parser.add_argument("--position_lr_final", type=float, default=0.0000016)
    parser.add_argument("--position_lr_delay_mult", type=float, default=0.01)
    parser.add_argument("--position_lr_max_steps", type=int, default=30000)
    parser.add_argument("--feature_lr", type=float, default=0.0025)
    parser.add_argument("--opacity_lr", type=float, default=0.05)
    parser.add_argument("--scaling_lr", type=float, default=0.005)
    parser.add_argument("--rotation_lr", type=float, default=0.001)

    # --- Densification ---
    parser.add_argument("--densify_from_iter", type=int, default=500)
    parser.add_argument("--densify_until_iter", type=int, default=20000)
    parser.add_argument("--densify_grad_threshold", type=float, default=0.00005)
    parser.add_argument("--densification_interval", type=int, default=100)

    # --- Opacity ---
    parser.add_argument("--opacity_reset_interval", type=int, default=100000,
                        help="Reset all opacities every N iter (default: disabled for scaffold init)")
    parser.add_argument("--min_opacity", type=float, default=0.005)

    # --- Loss ---
    parser.add_argument("--lambda_l1", type=float, default=0.8)
    parser.add_argument("--lambda_ssim", type=float, default=0.2)
    parser.add_argument("--lambda_depth", type=float, default=0.1)
    parser.add_argument("--depth_anneal_start", type=int, default=0)
    parser.add_argument("--depth_anneal_end", type=int, default=15000)

    # --- Fine-tune mode (Stage E) ---
    parser.add_argument("--init_ply", type=str, default=None,
                        help="Load this .ply instead of scaffold (for Stage E fine-tuning)")
    parser.add_argument("--lr_scale", type=float, default=1.0,
                        help="Multiply all learning rates by this factor (use 0.1 for Stage E)")

    # --- Other ---
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--max_init_points", type=int, default=200000,
                        help="Subsample scaffold PLY to this many points (0=no limit)")
    parser.add_argument("--checkpoint_interval", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Validate: must have either --scaffold_ply or --init_ply
    if args.scaffold_ply is None and args.init_ply is None:
        parser.error("Either --scaffold_ply or --init_ply must be provided.")

    return args


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
