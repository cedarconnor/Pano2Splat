"""Camera trajectory generator for Pano2Splat Stage B.

Generates 100 camera views across 5 rings + pole supplements for
full spherical coverage of the scene. See design doc section 6.3.
"""

import json
import logging
from pathlib import Path
from typing import Union

import numpy as np

from src.utils.camera import look_at, make_intrinsic

logger = logging.getLogger(__name__)


def generate_ring(
    n_views: int,
    elevation_deg: float,
    radius: float,
    look_inward: bool = True,
) -> list[np.ndarray]:
    """Generate camera extrinsics for a ring of views.

    Camera positions are distributed uniformly in azimuth on a ring at the
    given elevation and radius from the world origin. All cameras share the
    same up reference [0, 1, 0] (Y-up convention).

    Args:
        n_views: Number of views evenly spaced around the ring.
        elevation_deg: Elevation angle in degrees (positive = above horizon).
        radius: Distance from the world origin to each camera.
        look_inward: If True cameras point toward origin; otherwise outward.

    Returns:
        List of n_views 4x4 numpy arrays (world-to-camera extrinsics).
    """
    poses: list[np.ndarray] = []
    elevation = np.radians(elevation_deg)

    for i in range(n_views):
        azimuth = 2.0 * np.pi * i / n_views

        # Camera position on the ring
        x = radius * np.cos(elevation) * np.cos(azimuth)
        y = radius * np.sin(elevation)
        z = radius * np.cos(elevation) * np.sin(azimuth)
        eye = np.array([x, y, z])

        # Look direction
        if look_inward:
            target = np.array([0.0, 0.0, 0.0])
        else:
            # Look outward: target is twice the position (radially away)
            target = np.array([x * 2.0, y, z * 2.0])

        extrinsic = look_at(eye, target)
        poses.append(extrinsic)

    return poses


def generate_pole_views(n_views: int = 8) -> list[np.ndarray]:
    """Generate supplementary views looking straight up and straight down.

    Produces 4 upward-looking and 4 downward-looking cameras positioned
    at cardinal directions (0, 90, 180, 270 deg azimuth) at a small
    radius from the center.

    Args:
        n_views: Total number of pole views (must be even). Default 8.

    Returns:
        List of 4x4 extrinsic matrices.
    """
    if n_views % 2 != 0:
        raise ValueError(f"n_views must be even, got {n_views}")

    poses: list[np.ndarray] = []
    half = n_views // 2
    radius = 0.2  # small offset from center
    cardinal_azimuths = [2.0 * np.pi * i / half for i in range(half)]

    for azimuth in cardinal_azimuths:
        # Upward-looking view
        x = radius * np.cos(azimuth)
        z = radius * np.sin(azimuth)
        eye = np.array([x, 0.0, z])
        target_up = np.array([x, 5.0, z])  # look straight up
        # Use a forward-facing vector as the up hint to avoid
        # degeneracy when forward is parallel to world up
        up_hint = np.array([-np.cos(azimuth), 0.0, -np.sin(azimuth)])
        poses.append(look_at(eye, target_up, up=up_hint))

    for azimuth in cardinal_azimuths:
        # Downward-looking view
        x = radius * np.cos(azimuth)
        z = radius * np.sin(azimuth)
        eye = np.array([x, 0.0, z])
        target_down = np.array([x, -5.0, z])  # look straight down
        up_hint = np.array([-np.cos(azimuth), 0.0, -np.sin(azimuth)])
        poses.append(look_at(eye, target_down, up=up_hint))

    return poses


def generate_trajectory(config: dict) -> dict:
    """Combine all rings and pole views into a full camera trajectory.

    Args:
        config: Dict with keys matching the trajectory section of
                configs/pipeline.yaml:
                - rings: list of dicts with n_views, elevation_deg,
                         radius, look_inward
                - pole_views: int (total pole supplement views)
                - fov_deg: float
                - image_width: int
                - image_height: int

    Returns:
        Dict with key "poses", a list of 100 dicts each containing:
        - "id": int
        - "extrinsic": 4x4 nested list (world-to-camera)
        - "intrinsic": 3x3 nested list
        - "width": int
        - "height": int
    """
    rings = config["rings"]
    pole_count = config.get("pole_views", 8)
    fov_deg = config["fov_deg"]
    width = config["image_width"]
    height = config["image_height"]

    intrinsic = make_intrinsic(fov_deg, width, height)

    all_extrinsics: list[np.ndarray] = []

    # Generate ring views
    for ring_cfg in rings:
        ring_poses = generate_ring(
            n_views=ring_cfg["n_views"],
            elevation_deg=ring_cfg["elevation_deg"],
            radius=ring_cfg["radius"],
            look_inward=ring_cfg.get("look_inward", True),
        )
        all_extrinsics.extend(ring_poses)
        logger.info(
            "Ring '%s': %d views at elev=%.1f deg, r=%.2f, inward=%s",
            ring_cfg.get("name", "unnamed"),
            ring_cfg["n_views"],
            ring_cfg["elevation_deg"],
            ring_cfg["radius"],
            ring_cfg.get("look_inward", True),
        )

    # Generate pole supplement views
    if pole_count > 0:
        pole_poses = generate_pole_views(pole_count)
        all_extrinsics.extend(pole_poses)
        logger.info("Pole views: %d", pole_count)

    logger.info("Total trajectory views: %d", len(all_extrinsics))

    # Assemble output
    poses = []
    for idx, ext in enumerate(all_extrinsics):
        poses.append({
            "id": idx,
            "extrinsic": ext.tolist(),
            "intrinsic": intrinsic.tolist(),
            "width": width,
            "height": height,
        })

    return {"poses": poses}


def save_trajectory(trajectory: dict, output_path: Union[str, Path]) -> None:
    """Save trajectory dict to a JSON file.

    Args:
        trajectory: Dict as returned by generate_trajectory().
        output_path: Destination file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(trajectory, f, indent=2)
    logger.info("Trajectory saved to %s", output_path)


def check_ring4_coverage(
    scaffold_renders_dir: Union[str, Path],
    threshold: float = 0.4,
) -> list[dict]:
    """Check Ring 4 outward-looking renders for empty pixel ratio.

    Loads rendered images from the scaffold and computes the fraction of
    near-black (empty) pixels. Warns if any view exceeds the threshold.

    Args:
        scaffold_renders_dir: Directory containing render_*.png files.
        threshold: Maximum acceptable fraction of empty pixels (default 0.4).

    Returns:
        List of dicts with "file", "empty_ratio", and "pass" for each
        Ring 4 view checked.
    """
    from PIL import Image

    renders_dir = Path(scaffold_renders_dir)
    # Ring 4 starts after rings 1-3: views 24+16+16 = 56 through 75 (20 views)
    ring4_start = 56
    ring4_end = 76
    black_threshold = 10  # pixel values below this are considered empty

    results: list[dict] = []
    for idx in range(ring4_start, ring4_end):
        render_path = renders_dir / f"render_{idx:03d}.png"
        if not render_path.exists():
            logger.warning("Ring 4 render not found: %s", render_path)
            continue

        img = np.array(Image.open(render_path).convert("RGB"))
        # A pixel is "empty" if all channels are below the threshold
        empty_mask = np.all(img < black_threshold, axis=-1)
        empty_ratio = float(empty_mask.sum()) / empty_mask.size

        passed = empty_ratio <= threshold
        if not passed:
            logger.warning(
                "Ring 4 view %d: %.1f%% empty pixels (threshold: %.0f%%) — "
                "consider reducing Ring 4 radius or switching to inward-looking",
                idx, empty_ratio * 100, threshold * 100,
            )

        results.append({
            "file": str(render_path),
            "empty_ratio": round(empty_ratio, 4),
            "pass": passed,
        })

    n_failed = sum(1 for r in results if not r["pass"])
    if n_failed > 0:
        logger.warning(
            "Ring 4 coverage check: %d/%d views exceed %.0f%% empty threshold",
            n_failed, len(results), threshold * 100,
        )
    else:
        logger.info("Ring 4 coverage check: all views passed")

    return results


# ---------------------------------------------------------------------------
# Main: generate trajectory from pipeline.yaml and save to stage_b output
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config_path = Path("configs/pipeline.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {config_path}")

    with open(config_path, "r") as f:
        pipeline_cfg = yaml.safe_load(f)

    traj_cfg = pipeline_cfg["trajectory"]
    trajectory = generate_trajectory(traj_cfg)

    output_dir = Path(pipeline_cfg["output"]["root"]) / "stage_b"
    output_path = output_dir / "trajectory.json"
    save_trajectory(trajectory, output_path)

    print(f"Generated {len(trajectory['poses'])} camera poses")
    print(f"Saved to {output_path}")
