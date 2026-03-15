"""Export engine-ready .ply with pruning and coordinate transforms.

Stage F of the Pano2Splat pipeline. Prunes low-quality Gaussians,
applies coordinate transforms for the target engine, and writes a
compressed .ply file suitable for real-time rendering.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml

from src.utils.ply_io import load_ply, save_ply


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def prune_by_opacity(gaussians: dict, threshold: float) -> dict:
    """Remove Gaussians with sigmoid(opacity) below threshold."""
    opacities_activated = sigmoid(gaussians["opacities"].ravel())
    mask = opacities_activated >= threshold
    return _apply_mask(gaussians, mask), int((~mask).sum())


def prune_by_volume(gaussians: dict, threshold: float = 1e-10) -> dict:
    """Remove Gaussians whose scale volume (product of exp(scales)) is tiny."""
    scales_activated = np.exp(gaussians["scales"])  # (N, 3)
    volume = np.prod(scales_activated, axis=1)      # (N,)
    mask = volume >= threshold
    return _apply_mask(gaussians, mask), int((~mask).sum())


def prune_by_max_scale(gaussians: dict, max_scale: float = 1.0) -> dict:
    """Remove Gaussians where any activated scale dimension exceeds max_scale."""
    scales_activated = np.exp(gaussians["scales"])  # (N, 3)
    mask = np.all(scales_activated <= max_scale, axis=1)
    return _apply_mask(gaussians, mask), int((~mask).sum())


def cap_by_count(gaussians: dict, max_count: int) -> dict:
    """Keep the top-N Gaussians by opacity if count exceeds max_count."""
    n = gaussians["positions"].shape[0]
    if n <= max_count:
        return gaussians, 0

    opacities_activated = sigmoid(gaussians["opacities"].ravel())
    # Indices of the top max_count by opacity (descending)
    top_indices = np.argpartition(opacities_activated, -max_count)[-max_count:]
    top_indices = np.sort(top_indices)  # preserve original order

    pruned = {}
    for key, arr in gaussians.items():
        pruned[key] = arr[top_indices]
    return pruned, n - max_count


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of quaternions in wxyz convention.

    Args:
        q1: (..., 4) first quaternion(s).
        q2: (..., 4) second quaternion(s).

    Returns:
        (..., 4) product quaternion(s).
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def transform_y_up_to_unreal(gaussians: dict) -> dict:
    """Transform from Y-up right-handed to Unreal (Z-up left-handed).

    Unreal Engine convention:
        X = forward, Y = right, Z = up, left-handed.

    From Y-up (One2Scene) to Z-up left-handed:
        (x, y, z) -> (x, z, -y)

    This is a 90-degree rotation around the X axis (det = +1).
    Positions are remapped directly.  Quaternion rotations are composed
    with the transform quaternion via Hamilton product (q_new = q_T * q_old).
    Scales are unchanged — they correspond to the Gaussian's principal axes,
    which are carried along by the rotation composition.
    """
    out = dict(gaussians)

    # Positions: (x, y, z) -> (x, z, -y)
    pos = gaussians["positions"].copy()
    new_pos = np.empty_like(pos)
    new_pos[:, 0] = pos[:, 0]
    new_pos[:, 1] = pos[:, 2]
    new_pos[:, 2] = -pos[:, 1]
    out["positions"] = new_pos

    # Rotations: compose with q_T for 90° rotation around X axis.
    # q_T = (cos(π/4), sin(π/4), 0, 0) = (√2/2, √2/2, 0, 0)
    c = np.float32(np.sqrt(2.0) / 2.0)
    n = gaussians["rotations"].shape[0]
    q_T = np.broadcast_to(
        np.array([c, c, 0, 0], dtype=np.float32), (n, 4)
    ).copy()
    out["rotations"] = quaternion_multiply(q_T, gaussians["rotations"])

    # Scales stay unchanged — they are per-principal-axis of the Gaussian.
    # The rotation composition already handles the axis reorientation.

    return out


def apply_world_scale(gaussians: dict, scale: float) -> dict:
    """Apply uniform scale factor to positions and scales."""
    if scale == 1.0:
        return gaussians

    out = dict(gaussians)
    out["positions"] = gaussians["positions"] * scale
    # Scales are in log-space: log(s * scale) = log(s) + log(scale)
    out["scales"] = gaussians["scales"] + np.log(scale)
    return out


def fps_estimate(count: int) -> str:
    """Estimate FPS bracket based on Gaussian count."""
    if count < 500_000:
        return "60+ FPS (comfortable)"
    elif count <= 1_000_000:
        return "30-60 FPS (target)"
    elif count <= 1_500_000:
        return "20-30 FPS (upper bound)"
    else:
        return "Below target FPS"


def _apply_mask(gaussians: dict, mask: np.ndarray) -> dict:
    """Apply a boolean mask to all arrays in the gaussians dict."""
    return {key: arr[mask] for key, arr in gaussians.items()}


def _cli_flags(argv: list[str]) -> set[str]:
    flags = set()
    for arg in argv:
        if arg.startswith("--"):
            flags.add(arg.split("=", 1)[0])
    return flags


def _maybe_apply_config(args: argparse.Namespace, cli_flags: set[str], name: str, value) -> None:
    if value is None:
        return
    if f"--{name}" in cli_flags:
        return
    setattr(args, name.replace("-", "_"), value)


def apply_config_defaults(args: argparse.Namespace, cli_flags: set[str]) -> argparse.Namespace:
    if not args.config:
        return args

    with open(args.config, "r") as f:
        config = yaml.safe_load(f) or {}

    export_cfg = config.get("export", {})
    overlay = {
        "prune_opacity": export_cfg.get("prune_opacity_threshold"),
        "max_gaussians": export_cfg.get("max_gaussians"),
        "target_engine": export_cfg.get("target_engine"),
    }
    for name, value in overlay.items():
        _maybe_apply_config(args, cli_flags, name, value)

    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Export engine-ready .ply with pruning and coordinate transforms."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to trained .ply (Stage D or E output)",
    )
    parser.add_argument(
        "--output", type=str,
        default="pipeline_output/stage_f/final_export.ply",
        help="Output .ply path",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Optional pipeline.yaml path for export defaults",
    )
    parser.add_argument(
        "--prune_opacity", type=float, default=0.01,
        help="Opacity threshold for pruning (default 0.01)",
    )
    parser.add_argument(
        "--max_scale", type=float, default=1.0,
        help="Prune Gaussians with any exp(scale) > this (default 1.0)",
    )
    parser.add_argument(
        "--max_gaussians", type=int, default=1_500_000,
        help="Maximum Gaussian count (default 1500000)",
    )
    parser.add_argument(
        "--target_engine", type=str, default="unreal",
        choices=["unreal", "none"],
        help="Target engine for coordinate transform (default: unreal, none=skip)",
    )
    parser.add_argument(
        "--world_scale", type=float, default=1.0,
        help="Uniform scale factor for real-world units (default 1.0)",
    )
    args = parser.parse_args(argv)
    return apply_config_defaults(args, _cli_flags(argv))


def main() -> None:
    args = parse_args()

    # Load input
    input_path = Path(args.input)
    input_size = os.path.getsize(input_path)
    print(f"Loading {input_path}")
    gaussians = load_ply(args.input)
    n_input = gaussians["positions"].shape[0]
    print(f"  Input: {n_input:,} Gaussians, {input_size / 1e6:.1f} MB")

    # Prune low opacity
    gaussians, n_opacity = prune_by_opacity(gaussians, args.prune_opacity)
    if n_opacity > 0:
        print(f"  Pruned {n_opacity:,} low-opacity Gaussians "
              f"(sigmoid < {args.prune_opacity})")

    # Prune extreme scales
    gaussians, n_scale = prune_by_max_scale(gaussians, args.max_scale)
    if n_scale > 0:
        print(f"  Pruned {n_scale:,} extreme-scale Gaussians "
              f"(exp(scale) > {args.max_scale})")

    # Prune small volume
    gaussians, n_volume = prune_by_volume(gaussians, threshold=1e-10)
    if n_volume > 0:
        print(f"  Pruned {n_volume:,} near-zero volume Gaussians")

    # Cap count
    gaussians, n_capped = cap_by_count(gaussians, args.max_gaussians)
    if n_capped > 0:
        print(f"  Capped to {args.max_gaussians:,} Gaussians "
              f"(removed {n_capped:,} lowest-opacity)")

    n_output = gaussians["positions"].shape[0]
    print(f"  After pruning: {n_output:,} Gaussians "
          f"({n_output / n_input * 100:.1f}% retained)")

    # Coordinate transform
    if args.target_engine == "unreal":
        print("  Applying Y-up -> Unreal (Z-up left-handed) transform")
        gaussians = transform_y_up_to_unreal(gaussians)

    # World scale
    if args.world_scale != 1.0:
        print(f"  Applying world scale: {args.world_scale}")
        gaussians = apply_world_scale(gaussians, args.world_scale)

    # Save
    save_ply(args.output, gaussians)
    output_size = os.path.getsize(args.output)

    print(f"\nExport complete:")
    print(f"  Output: {args.output}")
    print(f"  Gaussians: {n_output:,}")
    print(f"  File size: {output_size / 1e6:.1f} MB")
    print(f"  Estimated performance: {fps_estimate(n_output)}")


if __name__ == "__main__":
    main()
