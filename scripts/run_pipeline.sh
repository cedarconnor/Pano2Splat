#!/bin/bash
set -e

# ============================================
# Pano2Splat: Full Pipeline Orchestrator
# Single A6000 (48GB VRAM)
# ============================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# --- Configuration ---
PANO_INPUT="${PANO_INPUT:-./input/panorama.jpg}"
WORK="${OUTPUT_ROOT:-./pipeline_output}"
ONE2SCENE="${PROJECT_ROOT}/third_party/One2Scene"
CONFIG="${PROJECT_ROOT}/configs/pipeline.yaml"

# Stage control: run all by default, or specify --stage X
STAGE="${1:-all}"

# --- Helpers ---
SECONDS=0

log() {
    echo ""
    echo "================================================================"
    echo "  $1"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"
}

elapsed() {
    local t=$SECONDS
    printf "  Completed in %02d:%02d:%02d\n" $((t/3600)) $((t%3600/60)) $((t%60))
}

check_file() {
    if [ ! -f "$1" ]; then
        echo "ERROR: Expected file not found: $1"
        echo "Stage failed validation. Aborting."
        exit 1
    fi
}

check_dir_not_empty() {
    if [ ! -d "$1" ] || [ -z "$(ls -A "$1" 2>/dev/null)" ]; then
        echo "ERROR: Expected directory missing or empty: $1"
        echo "Stage failed validation. Aborting."
        exit 1
    fi
}

# --- Setup ---
mkdir -p "$WORK"/{stage_a,stage_b,stage_c,stage_d,stage_e,stage_f}

if [ ! -f "$PANO_INPUT" ]; then
    echo "ERROR: Input panorama not found at: $PANO_INPUT"
    echo "Place your equirectangular panorama at $PANO_INPUT or set PANO_INPUT env var."
    exit 1
fi

TOTAL_START=$SECONDS

# ---- Stage A: Scaffold Generation ----
run_stage_a() {
    log "Stage A: Scaffold Generation (One2Scene)"
    STAGE_START=$SECONDS

    bash scripts/run_stage_a.sh "$PANO_INPUT"

    # Validation
    check_dir_not_empty "$WORK/stage_a"
    echo "  Stage A outputs:"
    ls -la "$WORK/stage_a/" 2>/dev/null || true

    SECONDS=$((SECONDS - STAGE_START))
    elapsed
    SECONDS=$STAGE_START
}

# ---- Stage B: Camera Trajectory ----
run_stage_b() {
    log "Stage B: Camera Trajectory"
    STAGE_START=$SECONDS

    # One2Scene builds its own trajectory internally (stored in data.pth).
    # Stage A extraction saves these as target_cameras.json.
    # We can either use those OR generate a custom 100-view trajectory.
    if [ -f "$WORK/stage_a/target_cameras.json" ]; then
        echo "  Using One2Scene's built-in trajectory from Stage A"
        mkdir -p "$WORK/stage_b"
        cp "$WORK/stage_a/target_cameras.json" "$WORK/stage_b/trajectory.json"
    else
        echo "  Generating custom 100-view trajectory"
        python -m src.trajectory
    fi

    # Validation
    check_file "$WORK/stage_b/trajectory.json"
    POSE_COUNT=$(python -c "
import json
with open('$WORK/stage_b/trajectory.json') as f:
    data = json.load(f)
poses = data.get('poses', data)
print(len(poses))
")
    echo "  Trajectory: $POSE_COUNT views"

    SECONDS=$((SECONDS - STAGE_START))
    elapsed
    SECONDS=$STAGE_START
}

# ---- Stage C: SEVA Denoise ----
run_stage_c() {
    log "Stage C: SEVA Denoise (sequential, single-GPU)"
    STAGE_START=$SECONDS

    bash scripts/run_stage_c.sh

    # Validation
    check_dir_not_empty "$WORK/stage_c/denoised_frames"
    FRAME_COUNT=$(ls "$WORK/stage_c/denoised_frames/"*.png 2>/dev/null | wc -l)
    echo "  Denoised frames: $FRAME_COUNT"
    if [ "$FRAME_COUNT" -lt 10 ]; then
        echo "WARNING: Only $FRAME_COUNT denoised frames found. Expected ~100."
    fi

    # Record resolution
    FIRST_FRAME=$(ls "$WORK/stage_c/denoised_frames/"*.png 2>/dev/null | head -1)
    if [ -n "$FIRST_FRAME" ]; then
        python -c "
from PIL import Image
img = Image.open('$FIRST_FRAME')
print(f'  SEVA output resolution: {img.size[0]}x{img.size[1]}')
"
    fi

    SECONDS=$((SECONDS - STAGE_START))
    elapsed
    SECONDS=$STAGE_START
}

# ---- Stage D: Splat Optimization ----
run_stage_d() {
    log "Stage D: Splat Optimization (30K iterations)"
    STAGE_START=$SECONDS

    # Use Stage C cameras if available, otherwise Stage B trajectory
    CAMERAS="$WORK/stage_b/trajectory.json"
    if [ -f "$WORK/stage_c/cameras.json" ]; then
        CAMERAS="$WORK/stage_c/cameras.json"
    fi

    python -m src.train_splat \
        --scaffold_ply "$WORK/stage_a/scaffold_gaussians.ply" \
        --images "$WORK/stage_c/denoised_frames/" \
        --cameras "$CAMERAS" \
        --scaffold_depths "$WORK/stage_a/anchor_depths/" \
        --anchor_cameras "$WORK/stage_a/anchor_cameras/cameras.json" \
        --output "$WORK/stage_d/" \
        --iterations 30000 \
        --densify_grad_threshold 0.00005 \
        --lambda_depth 0.1 \
        --depth_anneal_end 15000

    # Validation
    check_file "$WORK/stage_d/final.ply"
    check_file "$WORK/stage_d/metrics.json"

    echo "  Stage D metrics:"
    python -c "
import json
with open('$WORK/stage_d/metrics.json') as f:
    m = json.load(f)
print(f'    PSNR:       {m.get(\"psnr\", \"N/A\"):.2f} dB')
print(f'    SSIM:       {m.get(\"ssim\", \"N/A\"):.4f}')
print(f'    LPIPS:      {m.get(\"lpips\", \"N/A\"):.4f}')
print(f'    Gaussians:  {m.get(\"n_gaussians\", \"N/A\")}')
psnr = m.get('psnr', 0)
if psnr < 25:
    print(f'    WARNING: PSNR {psnr:.1f} dB is below 25 dB target')
"

    SECONDS=$((SECONDS - STAGE_START))
    elapsed
    SECONDS=$STAGE_START
}

# ---- Stage E: Upscale Refinement ----
run_stage_e() {
    log "Stage E: Upscale Refinement (Real-ESRGAN 4x)"
    STAGE_START=$SECONDS

    # Step 1: Render from trained splat
    echo "  [E.1] Rendering from trained splat..."
    python -m src.render_splat \
        --model_path "$WORK/stage_d/final.ply" \
        --cameras "$WORK/stage_d/training_cameras.json" \
        --output_dir "$WORK/stage_e/renders_native/"

    # Step 2: Upscale with Real-ESRGAN
    echo "  [E.2] Upscaling with Real-ESRGAN 4x..."
    REALESRGAN_DIR="$ONE2SCENE/third_party/Real-ESRGAN"
    if [ ! -d "$REALESRGAN_DIR" ]; then
        echo "WARNING: Real-ESRGAN not found at $REALESRGAN_DIR"
        echo "Skipping upscaling. Using native renders for fine-tuning."
        cp -r "$WORK/stage_e/renders_native/" "$WORK/stage_e/renders_upscaled/"
    else
        mkdir -p "$WORK/stage_e/renders_upscaled"
        python "$REALESRGAN_DIR/inference_realesrgan.py" \
            -i "$WORK/stage_e/renders_native/" \
            -o "$WORK/stage_e/renders_upscaled/" \
            -n RealESRGAN_x4plus \
            -s 4
    fi

    # Step 3: Fine-tune splat against upscaled images
    echo "  [E.3] Fine-tuning splat against upscaled frames..."
    python -m src.train_splat \
        --init_ply "$WORK/stage_d/final.ply" \
        --images "$WORK/stage_e/renders_upscaled/" \
        --cameras "$WORK/stage_d/training_cameras.json" \
        --scaffold_depths "$WORK/stage_a/anchor_depths/" \
        --anchor_cameras "$WORK/stage_a/anchor_cameras/cameras.json" \
        --output "$WORK/stage_e/" \
        --iterations 10000 \
        --lr_scale 0.1 \
        --densify_grad_threshold 0.0002 \
        --lambda_depth 0.05

    # Validation
    check_file "$WORK/stage_e/final.ply"
    echo "  Stage E complete. Compare refined vs Stage D output visually."

    SECONDS=$((SECONDS - STAGE_START))
    elapsed
    SECONDS=$STAGE_START
}

# ---- Stage F: Export ----
run_stage_f() {
    log "Stage F: Export & Compression"
    STAGE_START=$SECONDS

    # Use Stage E output if available, otherwise Stage D
    if [ -f "$WORK/stage_e/final.ply" ]; then
        INPUT_PLY="$WORK/stage_e/final.ply"
        echo "  Using Stage E refined splat"
    else
        INPUT_PLY="$WORK/stage_d/final.ply"
        echo "  Using Stage D splat (Stage E not available)"
    fi

    python -m src.export_splat \
        --input "$INPUT_PLY" \
        --output "$WORK/stage_f/final_export.ply" \
        --prune_opacity 0.01 \
        --max_gaussians 1500000 \
        --target_engine unreal

    # Validation
    check_file "$WORK/stage_f/final_export.ply"

    SECONDS=$((SECONDS - STAGE_START))
    elapsed
    SECONDS=$STAGE_START
}

# ---- Dispatch ----
case "$STAGE" in
    all)
        run_stage_a
        run_stage_b
        run_stage_c
        run_stage_d
        run_stage_e
        run_stage_f
        ;;
    a|A) run_stage_a ;;
    b|B) run_stage_b ;;
    c|C) run_stage_c ;;
    d|D) run_stage_d ;;
    e|E) run_stage_e ;;
    f|F) run_stage_f ;;
    *)
        echo "Usage: $0 [stage]"
        echo "  stage: a, b, c, d, e, f, or all (default)"
        exit 1
        ;;
esac

# ---- Summary ----
TOTAL_ELAPSED=$((SECONDS - TOTAL_START))
log "Pipeline Complete"
printf "  Total time: %02d:%02d:%02d\n" $((TOTAL_ELAPSED/3600)) $((TOTAL_ELAPSED%3600/60)) $((TOTAL_ELAPSED%60))

if [ "$STAGE" = "all" ]; then
    echo ""
    echo "  Output: $WORK/stage_f/final_export.ply"
    echo "  Load in SuperSplat or Unreal Engine to verify."
    echo ""
fi
