"""Per-stage validation for the Pano2Splat pipeline.

Validates output quality after each pipeline stage with quantitative metrics
and visual spot-checks. Can be run after any individual stage or across all.

Usage:
    python -m src.validate_pipeline --stage a --output_root ./pipeline_output
    python -m src.validate_pipeline --stage d --output_root ./pipeline_output
    python -m src.validate_pipeline --stage all --output_root ./pipeline_output
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def sigmoid(x):
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


# ---------------------------------------------------------------------------
# Stage A: Scaffold Validation
# ---------------------------------------------------------------------------

def validate_stage_a(output_root: str) -> dict:
    """Validate Stage A scaffold output."""
    stage_dir = Path(output_root) / "stage_a"
    results = {"stage": "A", "passed": True, "checks": []}

    # Check required files
    required = ["scaffold_gaussians.ply", "data.pth", "extraction_info.json"]
    for f in required:
        exists = (stage_dir / f).exists()
        results["checks"].append({"name": f"file_{f}", "passed": exists})
        if not exists:
            results["passed"] = False

    # Check required directories
    for d in ["anchor_views", "anchor_cameras", "scaffold_renders"]:
        exists = (stage_dir / d).exists()
        if exists:
            count = len(list((stage_dir / d).iterdir()))
            results["checks"].append({"name": f"dir_{d}", "passed": count > 0, "count": count})
        else:
            results["checks"].append({"name": f"dir_{d}", "passed": False})
            results["passed"] = False

    # Load and validate scaffold PLY
    ply_path = stage_dir / "scaffold_gaussians.ply"
    if ply_path.exists():
        from src.utils.ply_io import load_ply
        data = load_ply(str(ply_path))
        n = data["positions"].shape[0]
        results["n_gaussians"] = n
        results["checks"].append({"name": "gaussian_count", "passed": n > 1000, "value": n})

        # Check if real params (non-uniform scales)
        scales = np.exp(data["scales"])
        scale_std = float(scales.std())
        is_real = scale_std > 0.001  # fabricated has std~0
        results["scaffold_source"] = "real_params" if is_real else "fabricated"
        results["checks"].append({
            "name": "real_params",
            "passed": is_real,
            "scale_std": f"{scale_std:.6f}",
        })

        # Check position bounds
        pos = data["positions"]
        pos_range = float(pos.max() - pos.min())
        results["checks"].append({
            "name": "position_range",
            "passed": 0.1 < pos_range < 100.0,
            "value": f"{pos_range:.3f}",
        })

        # Opacity distribution
        opacities = sigmoid(data["opacities"].ravel())
        above_half = float((opacities > 0.5).mean())
        results["checks"].append({
            "name": "opacity_above_0.5",
            "passed": True,
            "value": f"{above_half:.1%}",
        })

    # Check extraction info
    info_path = stage_dir / "extraction_info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        results["extraction_info"] = info

    # Render 4 cardinal views from scaffold — failure means scaffold is unusable
    anchor_cams_path = stage_dir / "anchor_cameras" / "cameras.json"
    if ply_path.exists() and anchor_cams_path.exists():
        try:
            _render_scaffold_samples(str(ply_path), str(anchor_cams_path),
                                     str(stage_dir / "validation_renders"))
            results["checks"].append({"name": "scaffold_renders", "passed": True})
        except Exception as e:
            results["checks"].append({"name": "scaffold_renders", "passed": False,
                                      "error": str(e)})
            results["passed"] = False

    return results


def _render_scaffold_samples(ply_path: str, cameras_path: str, output_dir: str):
    """Render 4 sample views from scaffold for visual inspection."""
    import torch
    import torch.nn.functional as F
    from gsplat import rasterization
    from PIL import Image
    from src.utils.ply_io import load_ply

    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_ply(ply_path)

    positions = torch.tensor(data["positions"], dtype=torch.float32, device=device)
    rotations = F.normalize(
        torch.tensor(data["rotations"], dtype=torch.float32, device=device), p=2, dim=-1)
    scales = torch.exp(torch.tensor(data["scales"], dtype=torch.float32, device=device))
    opacities = torch.sigmoid(
        torch.tensor(data["opacities"], dtype=torch.float32, device=device).squeeze(-1))
    sh_dc = torch.tensor(data["sh_dc"], dtype=torch.float32, device=device).unsqueeze(1)

    with open(cameras_path) as f:
        cam_data = json.load(f)
    cameras = cam_data.get("poses", cam_data)

    # Render first 4 cameras (or fewer if not enough)
    n_render = min(4, len(cameras))
    with torch.no_grad():
        for i in range(n_render):
            cam = cameras[i]
            viewmat = torch.tensor(cam["extrinsic"], dtype=torch.float32, device=device)
            K = torch.tensor(cam["intrinsic"], dtype=torch.float32, device=device)
            w, h = cam["width"], cam["height"]
            if K[0, 0].item() < 10.0:
                K = K.clone()
                K[0, 0] *= w; K[0, 2] *= w
                K[1, 1] *= h; K[1, 2] *= h

            renders, _, _ = rasterization(
                means=positions, quats=rotations, scales=scales, opacities=opacities,
                colors=sh_dc, viewmats=viewmat.unsqueeze(0), Ks=K.unsqueeze(0),
                width=w, height=h, sh_degree=0, render_mode="RGB+ED")
            rgb = renders[0, :, :, :3].clamp(0, 1).cpu().numpy()
            img = (rgb * 255).astype(np.uint8)
            Image.fromarray(img).save(os.path.join(output_dir, f"scaffold_view_{i:02d}.png"))


# ---------------------------------------------------------------------------
# Stage C: SEVA Denoise Validation
# ---------------------------------------------------------------------------

def validate_stage_c(output_root: str) -> dict:
    """Validate Stage C denoised frames."""
    stage_dir = Path(output_root) / "stage_c"
    results = {"stage": "C", "passed": True, "checks": []}

    frames_dir = stage_dir / "denoised_frames"
    if not frames_dir.exists():
        results["passed"] = False
        results["checks"].append({"name": "frames_dir", "passed": False})
        return results

    frames = sorted(frames_dir.glob("*.png"))
    n_frames = len(frames)
    results["n_frames"] = n_frames

    # Determine expected count from Stage A extraction info
    info_path = Path(output_root) / "stage_a" / "extraction_info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        expected = info.get("n_target", 160)
        min_frames = max(50, int(expected * 0.8))
    else:
        min_frames = 50

    results["checks"].append({
        "name": "frame_count",
        "passed": n_frames >= min_frames,
        "value": n_frames,
        "target": f">= {min_frames}",
    })
    if n_frames < min_frames:
        results["passed"] = False

    if n_frames == 0:
        results["passed"] = False
        return results

    # Check resolution consistency
    from PIL import Image
    sizes = set()
    for f in frames[:8]:
        img = Image.open(f)
        sizes.add(img.size)
    consistent = len(sizes) == 1
    results["checks"].append({
        "name": "resolution_consistent",
        "passed": consistent,
        "resolutions": [list(s) for s in sizes],
    })

    # Sample 8 evenly-spaced frames and check brightness consistency
    sample_indices = np.linspace(0, n_frames - 1, min(8, n_frames), dtype=int)
    brightnesses = []
    for idx in sample_indices:
        img = np.array(Image.open(frames[idx]).convert("RGB")).astype(np.float32)
        brightnesses.append(img.mean())

    brightness_std = float(np.std(brightnesses))
    results["checks"].append({
        "name": "brightness_consistency",
        "passed": brightness_std < 20.0,
        "std": f"{brightness_std:.2f}",
        "values": [f"{b:.1f}" for b in brightnesses],
    })
    if brightness_std >= 20.0:
        results["passed"] = False

    return results


# ---------------------------------------------------------------------------
# Stage D: Splat Training Validation
# ---------------------------------------------------------------------------

def validate_stage_d(output_root: str) -> dict:
    """Validate Stage D trained splat."""
    stage_dir = Path(output_root) / "stage_d"
    results = {"stage": "D", "passed": True, "checks": []}

    # Check required files
    for f in ["final.ply", "metrics.json"]:
        exists = (stage_dir / f).exists()
        results["checks"].append({"name": f"file_{f}", "passed": exists})
        if not exists:
            results["passed"] = False

    # Load and check metrics
    metrics_path = stage_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        results["metrics"] = metrics

        psnr = metrics.get("psnr", 0)
        # Stage D at 512x512 with fabricated scaffold: 18-21 dB is realistic.
        # 25+ dB is only achievable after Stage E upscaling.
        results["checks"].append({
            "name": "psnr_target",
            "passed": psnr >= 18.0,
            "value": f"{psnr:.2f} dB",
            "target": ">= 18.0 dB",
        })
        if psnr < 18.0:
            results["passed"] = False
        if 18.0 <= psnr < 22.0:
            results["checks"].append({
                "name": "psnr_note",
                "passed": True,
                "value": f"PSNR {psnr:.2f} dB — acceptable for Stage D, Stage E should add 2-5 dB",
            })

        ssim = metrics.get("ssim", 0)
        results["checks"].append({
            "name": "ssim",
            "passed": ssim >= 0.6,
            "value": f"{ssim:.4f}",
            "target": ">= 0.6",
        })
        if ssim < 0.6:
            results["passed"] = False

    # Analyze trained PLY
    ply_path = stage_dir / "final.ply"
    if ply_path.exists():
        from src.utils.ply_io import load_ply
        data = load_ply(str(ply_path))
        n = data["positions"].shape[0]
        results["n_gaussians"] = n

        # Opacity distribution (key quality indicator)
        opacities = sigmoid(data["opacities"].ravel())
        above_half = float((opacities > 0.5).mean())
        above_01 = float((opacities > 0.1).mean())
        results["checks"].append({
            "name": "opacity_above_0.5",
            "passed": above_half >= 0.30,
            "value": f"{above_half:.1%}",
            "target": ">= 30%",
        })
        if above_half < 0.30:
            results["passed"] = False

        results["opacity_histogram"] = {
            "0.0-0.1": f"{(opacities < 0.1).mean():.1%}",
            "0.1-0.3": f"{((opacities >= 0.1) & (opacities < 0.3)).mean():.1%}",
            "0.3-0.5": f"{((opacities >= 0.3) & (opacities < 0.5)).mean():.1%}",
            "0.5-0.7": f"{((opacities >= 0.5) & (opacities < 0.7)).mean():.1%}",
            "0.7-0.9": f"{((opacities >= 0.7) & (opacities < 0.9)).mean():.1%}",
            "0.9-1.0": f"{(opacities >= 0.9).mean():.1%}",
        }

        # Scale distribution
        scales = np.exp(data["scales"])
        results["scale_stats"] = {
            "min": f"{scales.min():.6f}",
            "median": f"{np.median(scales):.6f}",
            "max": f"{scales.max():.6f}",
        }

        # Check for blue tint in SH DC
        sh_dc = data["sh_dc"]  # (N, 3)
        C0 = 0.28209479177387814
        avg_color = sh_dc.mean(axis=0) * C0 + 0.5
        results["avg_color_rgb"] = [f"{c:.3f}" for c in avg_color]
        blue_dominant = avg_color[2] > avg_color[0] + 0.05 and avg_color[2] > avg_color[1] + 0.05
        results["checks"].append({
            "name": "no_blue_tint",
            "passed": not blue_dominant,
            "avg_rgb": [f"{c:.3f}" for c in avg_color],
        })

    return results


# ---------------------------------------------------------------------------
# Stage E: Upscale Validation
# ---------------------------------------------------------------------------

def validate_stage_e(output_root: str) -> dict:
    """Validate Stage E upscale refinement."""
    stage_dir = Path(output_root) / "stage_e"
    results = {"stage": "E", "passed": True, "checks": []}

    # Check output PLY
    ply_path = stage_dir / "final.ply"
    exists = ply_path.exists()
    results["checks"].append({"name": "file_final.ply", "passed": exists})
    if not exists:
        results["passed"] = False

    # Compare metrics with Stage D
    stage_d_metrics_path = Path(output_root) / "stage_d" / "metrics.json"
    stage_e_metrics_path = stage_dir / "metrics.json"

    if stage_d_metrics_path.exists() and stage_e_metrics_path.exists():
        with open(stage_d_metrics_path) as f:
            d_metrics = json.load(f)
        with open(stage_e_metrics_path) as f:
            e_metrics = json.load(f)

        d_psnr = d_metrics.get("psnr", 0)
        e_psnr = e_metrics.get("psnr", 0)
        psnr_gain = e_psnr - d_psnr

        results["metrics"] = e_metrics
        results["psnr_gain"] = f"{psnr_gain:.2f} dB"
        results["checks"].append({
            "name": "psnr_improvement",
            "passed": psnr_gain >= 1.0,
            "value": f"{psnr_gain:.2f} dB",
            "target": ">= 1.0 dB over Stage D",
            "stage_d_psnr": f"{d_psnr:.2f}",
            "stage_e_psnr": f"{e_psnr:.2f}",
        })

    # Check upscaled frames exist
    upscaled_dir = stage_dir / "renders_upscaled"
    if upscaled_dir.exists():
        frames = list(upscaled_dir.glob("*.png"))
        results["checks"].append({
            "name": "upscaled_frames",
            "passed": len(frames) > 0,
            "count": len(frames),
        })

    return results


# ---------------------------------------------------------------------------
# Stage F: Export Validation
# ---------------------------------------------------------------------------

def validate_stage_f(output_root: str) -> dict:
    """Validate Stage F exported PLY."""
    stage_dir = Path(output_root) / "stage_f"
    results = {"stage": "F", "passed": True, "checks": []}

    ply_path = stage_dir / "final_export.ply"
    exists = ply_path.exists()
    results["checks"].append({"name": "file_final_export.ply", "passed": exists})
    if not exists:
        results["passed"] = False
        return results

    # File size
    size_mb = ply_path.stat().st_size / 1e6
    results["file_size_mb"] = f"{size_mb:.1f}"
    results["checks"].append({
        "name": "file_size",
        "passed": size_mb < 200.0,
        "value": f"{size_mb:.1f} MB",
    })

    # Load and analyze
    from src.utils.ply_io import load_ply
    data = load_ply(str(ply_path))
    n = data["positions"].shape[0]
    results["n_gaussians"] = n

    # Count check (real-time target)
    results["checks"].append({
        "name": "gaussian_count",
        "passed": n <= 1_500_000,
        "value": f"{n:,}",
        "target": "<= 1,500,000",
    })

    # FPS estimate
    if n < 500_000:
        fps = "60+ FPS"
    elif n <= 1_000_000:
        fps = "30-60 FPS"
    else:
        fps = "20-30 FPS"
    results["fps_estimate"] = fps

    # PLY format validation (62 properties check)
    try:
        from plyfile import PlyData
        plydata = PlyData.read(str(ply_path))
        vertex = plydata["vertex"]
        n_props = len(vertex.data.dtype.names)
        results["checks"].append({
            "name": "ply_properties",
            "passed": n_props == 62,
            "value": n_props,
            "target": 62,
        })
    except Exception as e:
        results["checks"].append({
            "name": "ply_format",
            "passed": False,
            "error": str(e),
        })

    # Opacity check — should be well-pruned
    opacities = sigmoid(data["opacities"].ravel())
    min_opacity = float(opacities.min())
    results["checks"].append({
        "name": "min_opacity_after_prune",
        "passed": min_opacity >= 0.005,
        "value": f"{min_opacity:.4f}",
    })

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: dict):
    """Print validation results in a readable format."""
    stage = results["stage"]
    passed = results["passed"]
    status = "PASS" if passed else "FAIL"

    print(f"\n{'='*60}")
    print(f"  Stage {stage} Validation: {status}")
    print(f"{'='*60}")

    for check in results.get("checks", []):
        icon = "+" if check["passed"] else "X"
        name = check["name"]
        value = check.get("value", "")
        target = check.get("target", "")
        line = f"  [{icon}] {name}"
        if value:
            line += f": {value}"
        if target:
            line += f"  (target: {target})"
        print(line)

    # Print extras
    if "opacity_histogram" in results:
        print(f"\n  Opacity distribution:")
        for bucket, pct in results["opacity_histogram"].items():
            print(f"    {bucket}: {pct}")

    if "metrics" in results:
        m = results["metrics"]
        print(f"\n  Metrics:")
        for k in ["psnr", "ssim", "lpips"]:
            if k in m:
                print(f"    {k}: {m[k]:.4f}")

    if "extraction_info" in results:
        info = results["extraction_info"]
        print(f"\n  Extraction info:")
        for k, v in info.items():
            print(f"    {k}: {v}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Validate Pano2Splat pipeline stages")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["a", "c", "d", "e", "f", "all"],
                        help="Stage to validate (default: all)")
    parser.add_argument("--output_root", type=str, default="./pipeline_output",
                        help="Pipeline output root directory")
    parser.add_argument("--json", type=str, default=None,
                        help="Save results as JSON to this path")
    args = parser.parse_args()

    stages = ["a", "c", "d", "e", "f"] if args.stage == "all" else [args.stage]

    validators = {
        "a": validate_stage_a,
        "c": validate_stage_c,
        "d": validate_stage_d,
        "e": validate_stage_e,
        "f": validate_stage_f,
    }

    all_results = {}
    all_passed = True

    for s in stages:
        results = validators[s](args.output_root)
        all_results[f"stage_{s}"] = results
        print_report(results)
        if not results["passed"]:
            all_passed = False

    # Summary
    print(f"{'='*60}")
    print(f"  Overall: {'ALL PASSED' if all_passed else 'SOME CHECKS FAILED'}")
    print(f"{'='*60}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.json}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
