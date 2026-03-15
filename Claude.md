# CLAUDE.md — Pano2Splat

## Project Overview
Single 360° panorama → production Gaussian splat (.ply) for Unreal/Unity.  
Built on One2Scene (ICLR 2026). Runs on a single A6000 (48GB).  
See `docs/pano2splat-design-doc-v2.md` for full architecture.

## Pipeline Stages
```
A: Scaffold (One2Scene feed-forward GS)  →  coarse .ply + depth maps
B: Camera trajectory (100 views, 5 rings)  →  extrinsic matrices
C: SEVA denoise (sequential, batch=1)  →  100 photorealistic frames
D: Splat optimization (gsplat, 30K iter)  →  trained .ply
E: Upscale refinement (Real-ESRGAN 4×)  →  refined .ply
F: Export & compress  →  engine-ready .ply/.splat
```

## Hardware Constraints
- **Single A6000, 48GB VRAM** — every stage must fit in this envelope
- Stage C is the tightest (~25-35GB at batch=1). If OOM: reduce `n_target_views`, enable fp16, enable flash attention. Try in that order.
- Stage D peaks at ~15-25GB. Can support up to ~3-5M Gaussians.
- Never assume multi-GPU. All torchrun calls use `--nproc_per_node=1`.

## Key Files
```
src/extract_scaffold.py            # Stage A→B: extract data.pth to 3DGS format
src/trajectory.py                  # Stage B: camera trajectory generation
src/train_splat.py                 # Stage D: splat optimization (core)
src/render_splat.py                # Stage E: render from trained splat
src/export_splat.py                # Stage F: prune + export for Unreal
src/utils/ply_io.py                # 3DGS PLY read/write
src/utils/camera.py                # Camera matrix utilities
src/utils/metrics.py               # PSNR/SSIM/LPIPS computation
configs/pipeline.yaml              # Master config (all stages)
configs/seva_single_gpu.yaml       # Single-GPU SEVA overrides
scripts/run_pipeline.sh            # Full pipeline orchestrator
docs/pano2splat-design-doc-v2.md   # Architecture doc
```

---

## Workflow Orchestration

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately — don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start

### 4. Verification Before Done
- Never mark a task complete without proving it works
- For pipeline stages: run the stage, inspect output, confirm expected artifacts exist
- For VRAM changes: monitor with `nvidia-smi` during first run to confirm fit
- For splat quality: render 3-5 validation views and check visually
- Ask yourself: "Would this survive a real panorama with complex geometry?"

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- Skip this for config tweaks, path changes, simple fixes
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, CUDA OOMs — then resolve them
- Zero context switching required from the user

---

## Task Management
1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

---

## Project-Specific Rules

### Coordinate Systems
- One2Scene scaffold outputs are in arbitrary scale — note the scale factor when exporting
- Unreal uses Z-up left-handed. Unity uses Y-up left-handed. Verify orientation on export.
- Camera extrinsics are world-to-camera 4×4 matrices. Don't mix up conventions.

### Panorama Handling
- Input is always equirectangular. Never assume cubemap unless explicitly converting.
- Pole regions (top/bottom 15%) will have lower quality. This is expected, not a bug.
- One2Scene internally decomposes to tangent-plane patches — don't second-guess this.

### SEVA Denoise Gotchas
- Default config assumes 8 GPUs. Always verify `nproc_per_node=1` before running.
- If frames show style drift or hallucinated objects between views, the batch size or conditioning is wrong — check the config, don't try to fix it in Stage D.
- SEVA outputs are the ground truth for Stage D. If they look bad, fix Stage C. Don't train a splat on bad supervision.

### Splat Optimization
- Initialize from scaffold Gaussians. Never from random or SfM points.
- `densify_grad_threshold` must be ~0.00005 for this pipeline (40× lower than default). Panoramic scenes with 100 views need aggressive densification.
- Depth regularization anneals to 0 by iteration 15K. This is intentional — early geometry anchoring, late photometric freedom.
- Target 500K–1.5M Gaussians for real-time engine playback. Prune before export.

### Upscale Refinement
- Real-ESRGAN introduces multi-view inconsistencies. This is expected.
- Compensate with: low learning rate (0.1×), maintained depth loss, higher densify threshold (0.0002).
- If floaters appear after Stage E, the SR artifacts are leaking in. Increase `lambda_depth` or reduce iterations.

### Quality Validation Checklist
After each stage, confirm before proceeding:
- **Stage A**: Scaffold renders show recognizable geometry. Depth maps are plausible.
- **Stage C**: Frames are sharp, photorealistic, consistent across views. No style drift.
- **Stage D**: PSNR 25+ dB on training views. No floaters from non-training angles.
- **Stage E**: Texture detail improved. No new floaters. Geometry unchanged.
- **Stage F**: Loads in target engine. Correct orientation. Acceptable FPS.

---

## Core Principles
- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.
- **VRAM Is The Bottleneck**: Always think about memory. Profile before and after changes.
- **Views > Tricks**: More well-distributed camera views will always beat fancier optimization. When quality is low, add views before adding complexity.
