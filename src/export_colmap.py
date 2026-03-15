"""Export pipeline cameras + SEVA frames to COLMAP text format.

Creates a standard COLMAP dataset that can be loaded by LichtFeld Studio,
COLMAP GUI, or any 3DGS trainer expecting COLMAP input.

Directory structure produced:
    <output>/
    ├── images/           # Copied SEVA frames
    │   ├── frame_000.png
    │   └── ...
    └── sparse/0/
        ├── cameras.txt   # Single shared PINHOLE camera
        ├── images.txt    # Per-image quaternion + translation
        └── points3D.txt  # Seed points from scaffold (optional)
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

from src.utils.camera import load_cameras_json


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to COLMAP quaternion (w, x, y, z).

    Uses Shepperd's method for numerical stability.
    """
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]

    trace = m00 + m11 + m22

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    return np.array([w, x, y, z], dtype=np.float64)


def write_cameras_txt(path: Path, width: int, height: int,
                      fx: float, fy: float, cx: float, cy: float) -> None:
    """Write cameras.txt with a single shared PINHOLE camera."""
    with open(path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {width} {height} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")


def write_images_txt(path: Path, cameras: list[dict],
                     image_names: list[str]) -> None:
    """Write images.txt with per-image extrinsics.

    COLMAP images.txt format: two lines per image.
    Line 1: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID IMAGE_NAME
    Line 2: 2D points (empty for us)
    """
    with open(path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        f.write(f"# Number of images: {len(cameras)}\n")

        for i, (cam, img_name) in enumerate(zip(cameras, image_names)):
            ext = cam["extrinsic"]  # 4x4 world-to-camera
            R = ext[:3, :3]
            t = ext[:3, 3]

            qvec = rotmat_to_qvec(R)
            image_id = i + 1
            camera_id = 1  # all share one camera

            f.write(f"{image_id} "
                    f"{qvec[0]:.10f} {qvec[1]:.10f} {qvec[2]:.10f} {qvec[3]:.10f} "
                    f"{t[0]:.10f} {t[1]:.10f} {t[2]:.10f} "
                    f"{camera_id} {img_name}\n")
            f.write("\n")  # empty 2D points line


def write_points3d_txt(path: Path, points: np.ndarray | None = None) -> None:
    """Write points3D.txt with optional seed points.

    If points is None, writes a minimal single-point file.
    """
    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")

        if points is None or len(points) == 0:
            f.write("# Number of points: 1\n")
            f.write("1 0.0 0.0 0.0 128 128 128 0.0\n")
        else:
            f.write(f"# Number of points: {len(points)}\n")
            for i, p in enumerate(points):
                f.write(f"{i + 1} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 128 128 128 0.0\n")


def load_seed_points(ply_path: str, max_points: int = 10000) -> np.ndarray:
    """Load and subsample positions from a PLY file for seeding."""
    from src.utils.ply_io import load_ply
    gaussians = load_ply(ply_path)
    positions = gaussians["positions"]
    n = positions.shape[0]
    if n > max_points:
        indices = np.random.default_rng(42).choice(n, max_points, replace=False)
        positions = positions[indices]
    print(f"  Loaded {n:,} scaffold points, subsampled to {positions.shape[0]:,}")
    return positions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export pipeline cameras + SEVA frames to COLMAP text format."
    )
    parser.add_argument(
        "--cameras", type=str, required=True,
        help="Path to training_cameras.json",
    )
    parser.add_argument(
        "--images", type=str, required=True,
        help="Directory containing SEVA denoised frames (frame_000.png, ...)",
    )
    parser.add_argument(
        "--output", type=str,
        default="pipeline_output/colmap_export",
        help="Output directory for COLMAP dataset",
    )
    parser.add_argument(
        "--points", type=str, default=None,
        help="Optional PLY file for seed points (scaffold positions)",
    )
    parser.add_argument(
        "--max_seed_points", type=int, default=10000,
        help="Maximum number of seed points to include (default 10000)",
    )
    parser.add_argument(
        "--copy_images", action="store_true", default=False,
        help="Copy images instead of symlinking (default: symlink)",
    )
    args = parser.parse_args()

    # Load cameras
    cameras = load_cameras_json(args.cameras)
    n_cams = len(cameras)
    print(f"Loaded {n_cams} cameras from {args.cameras}")

    # Get intrinsics from first camera (all share the same)
    intr = cameras[0]["intrinsic"]
    width = cameras[0]["width"]
    height = cameras[0]["height"]

    # Convert normalized intrinsics to pixel space
    fx_norm, fy_norm = intr[0, 0], intr[1, 1]
    cx_norm, cy_norm = intr[0, 2], intr[1, 2]

    # Check if already pixel-space or normalized
    if fx_norm < 10.0:
        # Normalized — convert to pixel space
        fx = fx_norm * width
        fy = fy_norm * height
        cx = cx_norm * width
        cy = cy_norm * height
    else:
        # Already pixel space
        fx, fy, cx, cy = fx_norm, fy_norm, cx_norm, cy_norm

    print(f"  Intrinsics (pixel): fx={fx:.3f} fy={fy:.3f} cx={cx:.3f} cy={cy:.3f}")
    print(f"  Image size: {width}x{height}")

    # Find source images
    images_dir = Path(args.images)
    image_files = sorted(images_dir.glob("frame_*.png"))
    if len(image_files) != n_cams:
        print(f"WARNING: Found {len(image_files)} images but {n_cams} cameras. "
              f"Using min({len(image_files)}, {n_cams}).")
        n_used = min(len(image_files), n_cams)
        image_files = image_files[:n_used]
        cameras = cameras[:n_used]
    else:
        n_used = n_cams

    # Create output structure
    out_dir = Path(args.output)
    out_images = out_dir / "images"
    out_sparse = out_dir / "sparse" / "0"
    out_images.mkdir(parents=True, exist_ok=True)
    out_sparse.mkdir(parents=True, exist_ok=True)

    # Copy/symlink images
    image_names = []
    for img_path in image_files:
        name = img_path.name
        image_names.append(name)
        dst = out_images / name
        if dst.exists():
            dst.unlink()
        if args.copy_images:
            shutil.copy2(img_path, dst)
        else:
            # Use copy on Windows (symlinks need admin privileges)
            if sys.platform == "win32":
                shutil.copy2(img_path, dst)
            else:
                dst.symlink_to(img_path.resolve())

    action = "Copied" if (args.copy_images or sys.platform == "win32") else "Symlinked"
    print(f"  {action} {len(image_names)} images to {out_images}")

    # Write cameras.txt
    cameras_path = out_sparse / "cameras.txt"
    write_cameras_txt(cameras_path, width, height, fx, fy, cx, cy)
    print(f"  Wrote {cameras_path}")

    # Write images.txt
    images_path = out_sparse / "images.txt"
    write_images_txt(images_path, cameras, image_names)
    print(f"  Wrote {images_path}")

    # Write points3D.txt
    points3d_path = out_sparse / "points3D.txt"
    seed_points = None
    if args.points:
        seed_points = load_seed_points(args.points, args.max_seed_points)
    write_points3d_txt(points3d_path, seed_points)
    print(f"  Wrote {points3d_path}")

    print(f"\nCOLMAP export complete: {out_dir}")
    print(f"  {n_used} images, 1 shared PINHOLE camera")
    if seed_points is not None:
        print(f"  {len(seed_points):,} seed points from scaffold")


if __name__ == "__main__":
    main()
