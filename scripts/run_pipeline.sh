#!/bin/bash
set -e

# ============================================
# Pano2Splat: Full Pipeline Orchestrator
# Single A6000 (48GB VRAM)
# ============================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
source "$SCRIPT_DIR/common.sh"

# --- Configuration ---
PANO_INPUT="${PANO_INPUT:-./input/panorama.jpg}"
WORK="${OUTPUT_ROOT:-./pipeline_output}"
# Export so sub-scripts (run_stage_a.sh, run_stage_c.sh) use the same root
export OUTPUT_ROOT="$WORK"
ONE2SCENE="${PROJECT_ROOT}/third_party/One2Scene"
CONFIG="${PROJECT_ROOT}/configs/pipeline.yaml"

# Stage control: run all by default, or specify a stage as positional arg or --stage X
STAGE="all"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage) STAGE="$2"; shift 2 ;;
        --stage=*) STAGE="${1#*=}"; shift ;;
        -*) echo "Unknown option: $1"; exit 1 ;;
        *) STAGE="$1"; shift ;;
    esac
done

# --- Helpers ---
log() {
    echo ""
    echo "================================================================"
    echo "  $1"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"
}

elapsed() {
    local start_time="$1"
    local end_time
    end_time=$(date +%s)
    local t=$((end_time - start_time))
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

TOTAL_START=$(date +%s)

# ---- Scaffold Generation ----
run_stage_a() {
    log "Scaffold Generation (One2Scene)"
    STAGE_START=$(date +%s)

    bash scripts/run_stage_a.sh "$PANO_INPUT"

    # Validation
    check_dir_not_empty "$WORK/stage_a"
    echo "  Stage A outputs:"
    ls -la "$WORK/stage_a/" 2>/dev/null || true

    elapsed "$STAGE_START"
}

# ---- Camera Trajectory ----
run_stage_b() {
    log "Camera Trajectory"
    STAGE_START=$(date +%s)

    # One2Scene builds its own trajectory internally (stored in data.pth).
    # Stage A extraction saves these as target_cameras.json.
    # We can either use those OR generate a custom 100-view trajectory.
    if [ -f "$WORK/stage_a/target_cameras.json" ]; then
        echo "  Using One2Scene's built-in trajectory from Stage A"
        mkdir -p "$WORK/stage_b"
        cp "$WORK/stage_a/target_cameras.json" "$WORK/stage_b/trajectory.json"
    else
        echo "  Generating custom 100-view trajectory"
        run_project_python "$PROJECT_ROOT" -m src.trajectory
    fi

    # Validation
    check_file "$WORK/stage_b/trajectory.json"
    POSE_COUNT=$(run_project_python "$PROJECT_ROOT" -c "
import json
with open('$WORK/stage_b/trajectory.json') as f:
    data = json.load(f)
poses = data.get('poses', data)
print(len(poses))
")
    echo "  Trajectory: $POSE_COUNT views"

    elapsed "$STAGE_START"
}

# ---- SEVA Denoise ----
run_stage_c() {
    log "SEVA Denoise (sequential, single-GPU)"
    STAGE_START=$(date +%s)

    bash scripts/run_stage_c.sh

    # Validation
    check_dir_not_empty "$WORK/stage_c/denoised_frames"
    FRAME_COUNT=$(ls "$WORK/stage_c/denoised_frames/"*.png 2>/dev/null | wc -l)
    echo "  Denoised frames: $FRAME_COUNT"
    if [ "$FRAME_COUNT" -lt 10 ]; then
        echo "WARNING: Only $FRAME_COUNT denoised frames found. Expected 160 target frames."
    fi

    # Record resolution
    FIRST_FRAME=$(ls "$WORK/stage_c/denoised_frames/"*.png 2>/dev/null | head -1)
    if [ -n "$FIRST_FRAME" ]; then
        run_project_python "$PROJECT_ROOT" -c "
from PIL import Image
img = Image.open('$FIRST_FRAME')
print(f'  SEVA output resolution: {img.size[0]}x{img.size[1]}')
"
    fi

    elapsed "$STAGE_START"
}

# ---- Splat Optimization ----
run_stage_d() {
    log "Splat Optimization (gsplat, multi-view batch training)"
    STAGE_START=$(date +%s)

    # Use Stage C cameras if available, otherwise Stage B trajectory
    CAMERAS="$WORK/stage_b/trajectory.json"
    if [ -f "$WORK/stage_c/cameras.json" ]; then
        CAMERAS="$WORK/stage_c/cameras.json"
    fi

    run_project_python_vs "$PROJECT_ROOT" -m src.train_splat \
        --scaffold_ply "$WORK/stage_a/scaffold_gaussians.ply" \
        --images "$WORK/stage_c/denoised_frames/" \
        --cameras "$CAMERAS" \
        --scaffold_depths "$WORK/stage_a/anchor_depths/" \
        --anchor_cameras "$WORK/stage_a/anchor_cameras/cameras.json" \
        --output "$WORK/stage_d/" \
        --config "$CONFIG"

    # Validation
    check_file "$WORK/stage_d/final.ply"
    check_file "$WORK/stage_d/metrics.json"

    echo "  Splat optimization metrics:"
    run_project_python "$PROJECT_ROOT" -c "
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

    elapsed "$STAGE_START"
}

# ---- Upscale Refinement ----
run_stage_e() {
    log "Upscale Refinement (SeedVR2 4x)"
    STAGE_START=$(date +%s)

    # Step 1: Render from trained splat
    echo "  [E.1] Rendering from trained splat..."
    run_project_python_vs "$PROJECT_ROOT" -m src.render_splat \
        --model_path "$WORK/stage_d/final.ply" \
        --cameras "$WORK/stage_d/training_cameras.json" \
        --output_dir "$WORK/stage_e/renders_native/"

    # Step 2: Upscale with SeedVR2 (temporal consistency via one-step diffusion)
    echo "  [E.2] Upscaling with SeedVR2 4x..."
    PYTHONIOENCODING=utf-8 run_project_python "$PROJECT_ROOT" -m src.upscale \
        --input "$WORK/stage_e/renders_native/" \
        --output "$WORK/stage_e/renders_upscaled/" \
        --target_resolution 2048 \
        --no_video_mode

    # Step 3: Fine-tune splat against upscaled images
    echo "  [E.3] Fine-tuning splat against upscaled frames..."
    run_project_python_vs "$PROJECT_ROOT" -m src.train_splat \
        --init_ply "$WORK/stage_d/final.ply" \
        --images "$WORK/stage_e/renders_upscaled/" \
        --cameras "$WORK/stage_d/training_cameras.json" \
        --scaffold_depths "$WORK/stage_a/anchor_depths/" \
        --anchor_cameras "$WORK/stage_a/anchor_cameras/cameras.json" \
        --output "$WORK/stage_e/" \
        --config "$CONFIG"

    # Validation
    check_file "$WORK/stage_e/final.ply"
    echo "  Upscale refinement complete. Compare refined vs base splat output visually."

    elapsed "$STAGE_START"
}

# ---- Export ----
run_stage_f() {
    log "Export & Compression"
    STAGE_START=$(date +%s)

    # Use Stage E output if available, otherwise Stage D
    if [ -f "$WORK/stage_e/final.ply" ]; then
        INPUT_PLY="$WORK/stage_e/final.ply"
        echo "  Using upscale-refined splat"
    else
        INPUT_PLY="$WORK/stage_d/final.ply"
        echo "  Using base splat (upscale refinement not available)"
    fi

    run_project_python "$PROJECT_ROOT" -m src.export_splat \
        --input "$INPUT_PLY" \
        --output "$WORK/stage_f/final_export.ply" \
        --config "$CONFIG"

    # Validation
    check_file "$WORK/stage_f/final_export.ply"

    elapsed "$STAGE_START"
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
    a|A|scaffold)   run_stage_a ;;
    b|B|trajectory) run_stage_b ;;
    c|C|denoise)    run_stage_c ;;
    d|D|train)      run_stage_d ;;
    e|E|upscale)    run_stage_e ;;
    f|F|export)     run_stage_f ;;
    *)
        echo "Usage: $0 [stage]"
        echo "  stage: scaffold, trajectory, denoise, train, upscale, export, or all (default)"
        echo "  (also accepts: a, b, c, d, e, f)"
        exit 1
        ;;
esac

# ---- Summary ----
TOTAL_ELAPSED=$(( $(date +%s) - TOTAL_START ))
log "Pipeline Complete"
printf "  Total time: %02d:%02d:%02d\n" $((TOTAL_ELAPSED/3600)) $((TOTAL_ELAPSED%3600/60)) $((TOTAL_ELAPSED%60))

if [ "$STAGE" = "all" ]; then
    echo ""
    echo "  Output: $WORK/stage_f/final_export.ply"
    echo "  Load in SuperSplat or Unreal Engine to verify."
    echo ""
fi
