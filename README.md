# Pano2Splat

Convert a single 360° equirectangular panorama into a production-ready 3D Gaussian Splat (`.ply`) for Unreal Engine.

Built on [One2Scene](https://github.com/imlixinyang/One2Scene) (ICLR 2026). Runs end-to-end on a single NVIDIA A6000 (48 GB VRAM).

## Pipeline

```
Input: 360° panorama (equirectangular JPEG/PNG)
                │
    ┌───────────▼───────────┐
    │  A  Scaffold (One2Scene)  │  Feed-forward → coarse Gaussians + depth maps
    └───────────┬───────────┘
                │
    ┌───────────▼───────────┐
    │  B  Camera Trajectory     │  One2Scene built-in 160 views by default
    └───────────┬───────────┘
                │
    ┌───────────▼───────────┐
    │  C  SEVA Denoise          │  Sequential diffusion → photorealistic frames
    └───────────┬───────────┘
                │
    ┌───────────▼───────────┐
    │  D  Splat Optimization    │  gsplat 30K iterations, depth-regularized
    └───────────┬───────────┘
                │
    ┌───────────▼───────────┐
    │  E  Upscale Refinement    │  SeedVR2 4× → fine-tune with SR frames
    └───────────┬───────────┘
                │
    ┌───────────▼───────────┐
    │  F  Export & Compress     │  Prune + Unreal coordinate transform
    └───────────┬───────────┘
                │
Output: engine-ready .ply (Z-up left-handed)
```

## Quick Start

Windows (PowerShell):

```powershell
# 1. Clone with submodules
git clone --recursive https://github.com/your-org/Pano2Splat.git
cd Pano2Splat

# 2. Set up environment
.\scripts\setup_one2scene.bat

# 3. Download model weights (~20 GB)
.\scripts\download_weights.bat

# 4. Place your panorama
Copy-Item .\my_panorama.jpg .\input\panorama.jpg

# 5. Run the full pipeline
.\scripts\run_pipeline.bat all
```

Linux or Git Bash:

```bash
# 1. Clone with submodules
git clone --recursive https://github.com/your-org/Pano2Splat.git
cd Pano2Splat

# 2. Set up environment
bash scripts/setup_one2scene.sh

# 3. Download model weights (~20 GB)
bash scripts/download_weights.sh

# 4. Place your panorama
cp my_panorama.jpg input/panorama.jpg

# 5. Run the full pipeline
bash scripts/run_pipeline.sh all
```

The final splat is written to `pipeline_output/stage_f/final_export.ply`.

## Running Individual Stages

Windows:

```powershell
.\scripts\run_pipeline.bat a
.\scripts\run_pipeline.bat b
.\scripts\run_pipeline.bat c
.\scripts\run_pipeline.bat d
.\scripts\run_pipeline.bat e
.\scripts\run_pipeline.bat f
```

Linux or Git Bash:

```bash
bash scripts/run_pipeline.sh a
bash scripts/run_pipeline.sh b
bash scripts/run_pipeline.sh c
bash scripts/run_pipeline.sh d
bash scripts/run_pipeline.sh e
bash scripts/run_pipeline.sh f
```

## Tests

Windows:

```powershell
.\scripts\run_tests.bat -q
```

Linux or Git Bash:

```bash
bash scripts/run_tests.sh -q
```

## Hardware Requirements

| Requirement | Minimum |
|-------------|---------|
| GPU | NVIDIA A6000 (48 GB VRAM) |
| CUDA | 12.4+ (system), torch bundles 12.6 runtime |
| OS | Windows 11 or Linux |
| Disk | ~40 GB (models + intermediate outputs) |

**VRAM by stage:**
- Stage A: ~15 GB
- Stage C (SEVA): 25–35 GB (tightest fit, batch_size=1)
- Stage D (gsplat): 15–25 GB

If Stage C OOMs: reduce `n_target_views`, enable fp16, enable flash attention — in that order.

## Dependencies

Core: `torch>=2.7.0+cu126`, `gsplat>=1.5.0`, `numpy`, `Pillow`

Attention acceleration: `xformers`, `flash-attn`, `triton-windows`, `sageattention`

Metrics: `lpips`, `pytorch-msssim`

See `requirements.txt` for the full list. Install with:

```bash
uv pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu126
```

### Windows Notes

- Use the batch wrappers in `scripts/` from PowerShell. They resolve Git Bash explicitly and avoid the broken WSL `bash.exe` launcher.
- Stage A, D, and E automatically use `tests/run_with_vs.bat`, so `gsplat` gets the required Visual Studio toolchain environment.
- gsplat JIT compilation requires Visual Studio (vcvars64.bat). Use `tests/run_with_vs.bat` as a Python wrapper.
- gsplat's `-Wno-attributes` flag must be patched out for MSVC (see `tasks/lessons.md` #6).
- torch header `CUDACachingAllocator.h` has a Windows SDK conflict (`#define small char`). Patch `bool small` → `bool is_small`.

## Project Structure

```
Pano2Splat/
├── src/
│   ├── extract_scaffold.py    # Stage A→B: data.pth → 3DGS .ply + cameras
│   ├── trajectory.py          # Stage B: optional custom 100-view fallback
│   ├── train_splat.py         # Stage D: gsplat optimization (core)
│   ├── render_splat.py        # Stage E: render trained splat
│   ├── upscale.py             # Stage E: SeedVR2 upscaling wrapper
│   ├── export_splat.py        # Stage F: prune + Unreal transform
│   └── utils/
│       ├── ply_io.py          # 3DGS PLY read/write
│       ├── camera.py          # Camera matrix utilities
│       └── metrics.py         # PSNR/SSIM/LPIPS
├── scripts/
│   ├── run_pipeline.sh        # Full pipeline orchestrator
│   ├── run_pipeline.bat       # Windows wrapper for the orchestrator
│   ├── run_tests.sh           # Cross-platform test entrypoint
│   ├── run_tests.bat          # Windows test wrapper
│   ├── run_stage_a.sh         # Scaffold generation wrapper
│   ├── run_stage_c.sh         # SEVA denoise wrapper
│   ├── setup_one2scene.sh     # Environment setup
│   ├── setup_one2scene.bat    # Windows wrapper for setup
│   └── download_weights.sh    # Model weight download
├── configs/
│   ├── pipeline.yaml          # Master config (all stages)
│   └── seva_single_gpu.yaml   # Single-GPU SEVA overrides
├── third_party/
│   └── One2Scene/             # Git submodule (pinned commit)
├── input/                     # Place panorama here
├── pipeline_output/           # All stage outputs
├── models/                    # Model checkpoints
├── tasks/
│   ├── todo.md                # Task tracker
│   └── lessons.md             # Lessons learned (23 entries)
└── docs/
    └── pano2splat-design-doc-v2.md
```

## Configuration

The orchestrator now passes `configs/pipeline.yaml` into Stages D, E, and F. Key parameters:

```yaml
# Stage D
training:
  iterations: 30000
  densify_grad_threshold: 0.0001   # safer default for fabricated scaffold init
  max_init_points: 200000          # Subsample dense scaffold
  opacity_reset_interval: 100000   # Effectively disabled
  lambda_l1: 0.8
  lambda_ssim: 0.2
  lambda_depth: 0.1               # Anneals to 0 by iter 15K

# Stage E
upscale:
  finetune_iterations: 10000
  lr_scale: 0.1
  densify_grad_threshold: 0.0002
  lambda_depth: 0.05

# Stage F
export:
  target_engine: unreal            # Z-up left-handed
  prune_opacity_threshold: 0.01
  max_gaussians: 1500000
```

## Output

| Stage | Output | Typical Size |
|-------|--------|-------------|
| A | `scaffold_gaussians.ply` + depth maps | ~88 MB |
| B | `trajectory.json` (160 cameras by default) | ~50 KB |
| C | 160 denoised frames (512×512 PNG) | ~50 MB |
| D | `final.ply` + `metrics.json` | ~47 MB |
| F | `final_export.ply` (pruned) | ~33 MB |

**Verified metrics (scaffold-only supervision):**
- Stage D: 200K Gaussians, PSNR 19.68 dB @ 30K iter, 5.2 min
- Stage F: 137K Gaussians, 32.5 MB, 60+ FPS target

With SEVA-denoised supervision, PSNR target is ≥ 25 dB.

## Technical Details

### PLY Format
Gaussian parameters are stored in **parameter space**: log-scales, opacity logits, unnormalized quaternions (wxyz). At render time, apply:
- `scales = exp(log_scales)`
- `opacities = sigmoid(logit_opacities)`
- `quats = normalize(raw_quats)`

### Coordinate Transform (Unreal)
The One2Scene → Unreal transform `(x,y,z) → (x,z,-y)` is a proper rotation (det=+1). Quaternion rotation uses Hamilton product `q_new = q_T * q_old` where `q_T = (√2/2, √2/2, 0, 0)`. Component shuffling is mathematically incorrect for anisotropic Gaussians.

### Depth Regularization
Scaffold depth maps are rendered from the coarse point cloud at anchor cameras via gsplat. Depth loss uses global median-ratio scale alignment and anneals to zero by iteration 15K — early geometry anchoring with late photometric freedom.

## Known Limitations

- **Scaffold uses fabricated Gaussian parameters** (isotropic 0.01 scales, identity rotations) from the fused point cloud. Real scaffold Gaussians require One2Scene checkpoint loading.
- **PSNR ~20 dB with scaffold-only supervision.** Denoised frames from Stage C are needed to reach the 25+ dB target.
- **Default trajectory is still One2Scene's built-in 160-view path.** `src/trajectory.py` is available as a custom fallback, but it is not the default flow.
- **Pole regions** (top/bottom 15° of panorama) will have lower quality. This is inherent to equirectangular projection.

## License

This project integrates One2Scene under its original license. See `third_party/One2Scene/LICENSE` for details.
