"""Stage E: Super-resolution upscaling of Gaussian splat renders.

Uses SeedVR2 (one-step diffusion transformer) to upscale 512x512 renders
to 2048x2048 with temporal consistency across views. Frames are stitched
into a video ordered along the camera trajectory so that SeedVR2's temporal
attention enforces cross-view consistency.

Usage:
    python -m src.upscale \
        --input pipeline_output/stage_e/renders_native/ \
        --output pipeline_output/stage_e/renders_upscaled/ \
        --target_resolution 2048
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


SEEDVR2_DIR = Path(__file__).parent.parent / "third_party" / "SeedVR2"


def get_sorted_images(image_dir: str) -> list[Path]:
    """Get sorted list of image files from directory."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    image_dir = Path(image_dir)
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in exts)
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return images


def images_to_video(image_paths: list[Path], video_path: str, fps: float = 24.0) -> None:
    """Stitch images into a lossless video for SeedVR2 processing."""
    first = cv2.imread(str(image_paths[0]))
    h, w = first.shape[:2]

    # Use ffmpeg for lossless encoding to avoid compression artifacts
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "0",  # lossless
        "-pix_fmt", "yuv444p",  # no chroma subsampling
        video_path,
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for p in image_paths:
        frame = cv2.imread(str(p))
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg encoding failed (exit code {proc.returncode})")

    size_mb = os.path.getsize(video_path) / (1024 * 1024)
    print(f"  Created temp video: {len(image_paths)} frames, {w}x{h}, {size_mb:.1f} MB")


def video_to_images(video_path: str, output_dir: str, expected_count: int) -> list[Path]:
    """Extract frames from SeedVR2 output video to individual PNGs."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    paths = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = Path(output_dir) / f"frame_{idx:03d}.png"
        cv2.imwrite(str(out_path), frame)
        paths.append(out_path)
        idx += 1
    cap.release()

    if idx != expected_count:
        print(f"  WARNING: Expected {expected_count} frames, got {idx}")

    return paths


def collect_seedvr2_png_output(output_dir: str, target_dir: str, expected_count: int) -> list[Path]:
    """Collect PNG frames from SeedVR2 output directory and rename to our convention."""
    output_dir = Path(output_dir)
    target_dir_path = Path(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    # When output_dir == target_dir (individual mode), old frame_*.png from
    # previous runs may still exist. Only collect SeedVR2's output files
    # (which preserve input names like render_*.png), not our renamed files.
    if output_dir.resolve() == target_dir_path.resolve():
        # Remove old frame_*.png from previous runs
        for old in sorted(target_dir_path.glob("frame_*.png")):
            old.unlink()

    # SeedVR2 PNG output: find all PNGs, sort by name
    pngs = sorted(output_dir.glob("*.png"))
    if not pngs:
        # Check subdirectories
        pngs = sorted(output_dir.rglob("*.png"))

    paths = []
    for idx, src in enumerate(pngs):
        dst = Path(target_dir) / f"frame_{idx:03d}.png"
        if src.resolve() != dst.resolve():
            shutil.move(str(src), str(dst))
        paths.append(dst)

    if len(paths) != expected_count:
        print(f"  WARNING: Expected {expected_count} frames, got {len(paths)}")

    return paths


def run_seedvr2(
    input_path: str,
    output_path: str,
    target_resolution: int = 2048,
    model: str = "seedvr2_ema_3b_fp16.safetensors",
    batch_size: int = 5,
    color_correction: str = "lab",
    model_dir: str | None = None,
) -> None:
    """Run SeedVR2 CLI on a video or image."""
    cli_path = SEEDVR2_DIR / "inference_cli.py"
    if not cli_path.exists():
        raise FileNotFoundError(
            f"SeedVR2 CLI not found: {cli_path}\n"
            "Run: git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git "
            "third_party/SeedVR2"
        )

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = [
        sys.executable, str(cli_path),
        input_path,
        "--resolution", str(target_resolution),
        "--dit_model", model,
        "--batch_size", str(batch_size),
        "--color_correction", color_correction,
        "--output_format", "png",
        "--output", output_path,
        "--seed", "42",
    ]
    if model_dir:
        cmd.extend(["--model_dir", model_dir])

    print(f"  Running SeedVR2: {model}, {target_resolution}px, batch_size={batch_size}")
    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"SeedVR2 failed (exit code {result.returncode})")


def upscale(args: argparse.Namespace) -> None:
    """Main upscaling workflow."""
    print("=" * 50)
    print("  Stage E: SeedVR2 Upscale")
    print("=" * 50)

    # 1. Gather input images
    image_paths = get_sorted_images(args.input)
    n_frames = len(image_paths)
    first = cv2.imread(str(image_paths[0]))
    h, w = first.shape[:2]
    print(f"\n  Input: {n_frames} images, {w}x{h}")
    print(f"  Target: {args.target_resolution}x{args.target_resolution}")
    print(f"  Model: {args.model}")
    print(f"  Mode: {'video (temporal consistency)' if args.video_mode else 'individual images'}")

    os.makedirs(args.output, exist_ok=True)

    if args.video_mode:
        # 2a. Stitch frames into temp video for temporal consistency
        print(f"\n[1/3] Creating temp video from {n_frames} frames...")
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_video = os.path.join(tmpdir, "input.mp4")
            images_to_video(image_paths, temp_video, fps=24.0)

            # 3. Run SeedVR2 on the video
            print(f"\n[2/3] Running SeedVR2 upscale...")
            output_subdir = os.path.join(tmpdir, "upscaled")
            run_seedvr2(
                input_path=temp_video,
                output_path=output_subdir,
                target_resolution=args.target_resolution,
                model=args.model,
                batch_size=args.batch_size,
                color_correction=args.color_correction,
                model_dir=args.model_dir,
            )

            # 4. Check for output video or PNGs
            print(f"\n[3/3] Collecting upscaled frames...")
            # SeedVR2 may output a video or PNG sequence depending on format
            output_videos = list(Path(output_subdir).rglob("*.mp4"))
            if output_videos:
                video_to_images(str(output_videos[0]), args.output, n_frames)
            else:
                collect_seedvr2_png_output(output_subdir, args.output, n_frames)
    else:
        # 2b. Process directory of individual images (no temporal consistency)
        print(f"\n[1/1] Running SeedVR2 on image directory...")
        run_seedvr2(
            input_path=args.input,
            output_path=args.output,
            target_resolution=args.target_resolution,
            model=args.model,
            batch_size=1,
            color_correction=args.color_correction,
            model_dir=args.model_dir,
        )
        # Rename output to our naming convention
        collect_seedvr2_png_output(args.output, args.output, n_frames)

    # 5. Report results
    output_pngs = sorted(Path(args.output).glob("frame_*.png"))
    if output_pngs:
        sample = cv2.imread(str(output_pngs[0]))
        oh, ow = sample.shape[:2]
        total_mb = sum(p.stat().st_size for p in output_pngs) / (1024 * 1024)
        print(f"\n  Results:")
        print(f"    Frames: {len(output_pngs)}")
        print(f"    Resolution: {ow}x{oh}")
        print(f"    Total size: {total_mb:.1f} MB")
    else:
        print("\n  WARNING: No output frames found!")

    print("=" * 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage E: Upscale renders using SeedVR2"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Directory of input images to upscale")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for upscaled images")
    parser.add_argument("--target_resolution", type=int, default=2048,
                        help="Target resolution (default: 2048)")
    parser.add_argument("--model", type=str,
                        default="seedvr2_ema_3b_fp16.safetensors",
                        help="SeedVR2 model name (default: 3B FP16)")
    parser.add_argument("--batch_size", type=int, default=5,
                        help="Batch size for video mode (use 4n+1: 5,9,13; default: 5)")
    parser.add_argument("--color_correction", type=str, default="lab",
                        choices=["none", "lab", "wavelet", "adain"],
                        help="Color correction mode (default: lab)")
    parser.add_argument("--video_mode", action="store_true", default=True,
                        help="Stitch frames as video for temporal consistency (default)")
    parser.add_argument("--no_video_mode", action="store_false", dest="video_mode",
                        help="Process each image independently (no temporal consistency)")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Custom model directory (auto-downloads if not set)")

    return parser.parse_args()


def main():
    args = parse_args()
    upscale(args)


if __name__ == "__main__":
    main()
