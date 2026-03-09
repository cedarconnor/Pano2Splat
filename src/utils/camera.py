"""Camera matrix utility functions for Pano2Splat pipeline.

Provides helper functions for constructing intrinsic/extrinsic matrices
and serializing camera parameters to/from JSON.
"""

import json
from pathlib import Path
from typing import Union

import numpy as np


def make_intrinsic(fov_deg: float, width: int, height: int) -> np.ndarray:
    """Construct a 3x3 camera intrinsic matrix from horizontal FOV.

    Args:
        fov_deg: Horizontal field of view in degrees.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        3x3 numpy array with focal lengths and principal point.
    """
    fov_rad = np.radians(fov_deg)
    fx = width / (2.0 * np.tan(fov_rad / 2.0))
    fy = fx  # square pixels
    cx = width / 2.0
    cy = height / 2.0
    return np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def look_at(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray = np.array([0.0, 1.0, 0.0]),
) -> np.ndarray:
    """Construct a 4x4 world-to-camera extrinsic matrix.

    Uses OpenGL-style convention: camera looks along -Z in camera space,
    Y is up, X is right.

    Args:
        eye: Camera position in world coordinates (3,).
        target: Point the camera looks at in world coordinates (3,).
        up: World up direction (3,). Defaults to [0, 1, 0].

    Returns:
        4x4 world-to-camera extrinsic matrix (numpy float64).
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    cam_up = np.cross(right, forward)

    # Rotation: rows are right, up, -forward (OpenGL convention)
    R = np.stack([right, cam_up, -forward], axis=0)
    t = -R @ eye

    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t
    return extrinsic


def load_cameras_json(path: Union[str, Path]) -> list[dict]:
    """Load camera parameters from a JSON file.

    Expects a JSON object with a "poses" key containing a list of dicts,
    each with "extrinsic" (4x4 nested list), "intrinsic" (3x3 nested list),
    "width", "height", and optionally "id".

    Args:
        path: Path to the JSON file.

    Returns:
        List of dicts with numpy arrays for "extrinsic" and "intrinsic",
        plus integer "width", "height", and "id".
    """
    path = Path(path)
    with open(path, "r") as f:
        data = json.load(f)

    poses = data.get("poses", data)  # support bare list or wrapped object
    cameras = []
    for entry in poses:
        cam = {
            "id": entry.get("id", len(cameras)),
            "extrinsic": np.array(entry["extrinsic"], dtype=np.float64),
            "intrinsic": np.array(entry["intrinsic"], dtype=np.float64),
            "width": int(entry["width"]),
            "height": int(entry["height"]),
        }
        cameras.append(cam)
    return cameras


def save_cameras_json(cameras: list[dict], path: Union[str, Path]) -> None:
    """Save camera parameters to a JSON file.

    Args:
        cameras: List of dicts, each with "id", "extrinsic" (4x4 numpy or list),
                 "intrinsic" (3x3 numpy or list), "width", "height".
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    poses = []
    for cam in cameras:
        ext = cam["extrinsic"]
        intr = cam["intrinsic"]
        poses.append({
            "id": cam["id"],
            "extrinsic": ext.tolist() if isinstance(ext, np.ndarray) else ext,
            "intrinsic": intr.tolist() if isinstance(intr, np.ndarray) else intr,
            "width": int(cam["width"]),
            "height": int(cam["height"]),
        })

    with open(path, "w") as f:
        json.dump({"poses": poses}, f, indent=2)
