#!/bin/bash
set -e

# =============================================================================
# Pano2Splat: Stage C - SEVA Denoise
# Runs One2Scene SEVA denoiser to produce photorealistic frames from scaffold
# renders. Single-GPU configuration (nproc_per_node=1).
#
# Requires Stage A output (data.pth) in pipeline_output/stage_a/.
#
# Usage: ./scripts/run_stage_c.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ONE2SCENE_DIR="$PROJECT_ROOT/third_party/One2Scene"
DENOISE_DIR="$ONE2SCENE_DIR/src_denoise"

# Paths
STAGE_A_OUT="$PROJECT_ROOT/pipeline_output/stage_a"
STAGE_C_OUT="$PROJECT_ROOT/pipeline_output/stage_c"
DENOISE_CKPT="$PROJECT_ROOT/models/one2scene_denoise.ckpt"
SEVA_CONFIG="$DENOISE_DIR/configs/seva_denoise_mostclosecamera.yaml"

# The data.pth from Stage A contains images + cameras for SEVA input
DATA_PTH="$STAGE_A_OUT/data.pth"

echo "============================================"
echo "  Pano2Splat: Stage C - SEVA Denoise"
echo "============================================"
echo ""
echo "  Data input:   $DATA_PTH"
echo "  Checkpoint:   $DENOISE_CKPT"
echo "  SEVA config:  $SEVA_CONFIG"
echo "  Output:       $STAGE_C_OUT"
echo ""

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
if [ ! -f "$DATA_PTH" ]; then
    echo "[ERROR] data.pth not found: $DATA_PTH"
    echo "        Run Stage A first: bash scripts/run_stage_a.sh"
    exit 1
fi

if [ ! -d "$DENOISE_DIR" ]; then
    echo "[ERROR] Denoise directory not found: $DENOISE_DIR"
    echo "        One2Scene may not be properly initialized."
    exit 1
fi

if [ ! -f "$DENOISE_CKPT" ]; then
    echo "[ERROR] Denoise checkpoint not found: $DENOISE_CKPT"
    echo "        Run: bash scripts/download_weights.sh"
    exit 1
fi

if [ ! -f "$SEVA_CONFIG" ]; then
    echo "[ERROR] SEVA config not found: $SEVA_CONFIG"
    exit 1
fi

# ---------------------------------------------------------------------------
# Create temporary config with path overrides
# ---------------------------------------------------------------------------
echo "[1/4] Creating config overrides..."

# SEVA's config has hardcoded paths. We create a temporary override file
# that points to our actual data.pth and output directory.
mkdir -p "$STAGE_C_OUT"

SEVA_DENOISE_OUT="$STAGE_C_OUT/seva_output"
mkdir -p "$SEVA_DENOISE_OUT"

# Create an override config that patches the hardcoded paths
OVERRIDE_CONFIG="$STAGE_C_OUT/seva_override.yaml"
cat > "$OVERRIDE_CONFIG" << YAML
# Auto-generated SEVA path overrides for Pano2Splat
# Points datafolder to Stage A output and save_dir to Stage C output.
test_data:
  params:
    datafolder: "${DATA_PTH}"

model:
  params:
    save_dir: "${SEVA_DENOISE_OUT}"
YAML

echo "  Override config: $OVERRIDE_CONFIG"
echo "  SEVA output dir: $SEVA_DENOISE_OUT"

# ---------------------------------------------------------------------------
# Run SEVA denoise
# ---------------------------------------------------------------------------
echo ""
echo "[2/4] Running SEVA denoise (single GPU, sequential)..."
echo "  Expected VRAM: 25-35GB. Expected time: 25-50 min."
echo "  Monitor with: watch nvidia-smi"
echo ""

START_TIME=$(date +%s)

cd "$DENOISE_DIR"

CUDA_VISIBLE_DEVICES=0 torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --master_addr=localhost \
    --master_port=21471 \
    main.py \
    --base "$SEVA_CONFIG" "$OVERRIDE_CONFIG" \
    --no_date \
    --train=False \
    --debug \
    --resume="$DENOISE_CKPT"

cd "$PROJECT_ROOT"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "  SEVA denoise complete in $((ELAPSED / 60))m $((ELAPSED % 60))s"

# ---------------------------------------------------------------------------
# Discovery: find denoised frames
# ---------------------------------------------------------------------------
echo ""
echo "[3/4] Discovering denoised frames..."

DISCOVERY_LOG="$STAGE_C_OUT/discovery_log.txt"
echo "# Stage C Discovery Log - $(date -u +"%Y-%m-%dT%H:%M:%SZ")" > "$DISCOVERY_LOG"
echo "# Elapsed: $((ELAPSED / 60))m $((ELAPSED % 60))s" >> "$DISCOVERY_LOG"
echo "" >> "$DISCOVERY_LOG"

# Search for output frames
echo "  Output files:" | tee -a "$DISCOVERY_LOG"
find "$SEVA_DENOISE_OUT" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.mp4" \) \
    -exec ls -lh {} \; 2>/dev/null | tee -a "$DISCOVERY_LOG"

# Also check if outputs went to the hardcoded default path
FALLBACK_DIR="$ONE2SCENE_DIR/demo_outputs/render_denoise"
if [ -d "$FALLBACK_DIR" ]; then
    echo "" | tee -a "$DISCOVERY_LOG"
    echo "  Fallback output dir:" | tee -a "$DISCOVERY_LOG"
    find "$FALLBACK_DIR" -type f -name "*.png" -exec ls -lh {} \; 2>/dev/null | tee -a "$DISCOVERY_LOG"
fi

# ---------------------------------------------------------------------------
# Copy denoised frames to canonical location
# ---------------------------------------------------------------------------
echo ""
echo "[4/4] Organizing denoised frames..."

DENOISED_DIR="$STAGE_C_OUT/denoised_frames"
mkdir -p "$DENOISED_DIR"

# Find PNG files that are denoised frames (not _input variants)
FRAME_SRC=""
for SEARCH_DIR in "$SEVA_DENOISE_OUT" "$FALLBACK_DIR"; do
    if [ -d "$SEARCH_DIR" ]; then
        # Look for denoised PNGs (exclude _input files which are scaffold renders)
        FOUND=$(find "$SEARCH_DIR" -maxdepth 3 -name "*.png" ! -name "*_input*" 2>/dev/null | head -1)
        if [ -n "$FOUND" ]; then
            FRAME_SRC="$(dirname "$FOUND")"
            break
        fi
    fi
done

if [ -z "$FRAME_SRC" ]; then
    echo "[WARN] No denoised frames found. Check discovery_log.txt."
    echo "  Searched: $SEVA_DENOISE_OUT, $FALLBACK_DIR"
else
    echo "  Source: $FRAME_SRC"

    # Copy and rename frames to frame_000.png, frame_001.png, etc.
    # WARNING: SEVA may output conditioning frames (first 6) alongside
    # denoised target frames.  The video export in model_wrapper.py
    # skips the first 6 with image[6:].  If SEVA outputs numbered PNGs,
    # we may need to skip the first 6 here too so that frame indices
    # align with target_cameras.json.  Verify after first successful
    # Stage C run.
    IDX=0
    for F in $(find "$FRAME_SRC" -maxdepth 1 -name "*.png" ! -name "*_input*" | sort); do
        cp "$F" "$DENOISED_DIR/frame_$(printf '%03d' $IDX).png"
        IDX=$((IDX + 1))
    done

    echo "  Copied $IDX denoised frames to $DENOISED_DIR"

    # Copy cameras from Stage A (SEVA doesn't modify them)
    if [ -f "$STAGE_A_OUT/target_cameras.json" ]; then
        cp "$STAGE_A_OUT/target_cameras.json" "$STAGE_C_OUT/cameras.json"
        echo "  Copied target cameras to $STAGE_C_OUT/cameras.json"
    fi

    # Record resolution
    FIRST_FRAME="$DENOISED_DIR/frame_000.png"
    if [ -f "$FIRST_FRAME" ]; then
        python -c "
from PIL import Image
img = Image.open('$FIRST_FRAME')
w, h = img.size
print(f'{w}x{h}')
with open('$STAGE_C_OUT/resolution.txt', 'w') as f:
    f.write(f'{w}x{h}\n')
print(f'Resolution recorded: {w}x{h}')
" 2>/dev/null || echo "  [WARN] Could not detect resolution (PIL not available)"
    fi
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "  Stage C Results"
echo "============================================"
echo ""
echo "  Time: $((ELAPSED / 60))m $((ELAPSED % 60))s"

FRAME_COUNT=$(find "$DENOISED_DIR" -name "frame_*.png" 2>/dev/null | wc -l)
echo "  Denoised frames: $FRAME_COUNT"

if [ -f "$STAGE_C_OUT/resolution.txt" ]; then
    RES=$(cat "$STAGE_C_OUT/resolution.txt")
    echo "  Resolution: $RES"
fi

if [ "$FRAME_COUNT" -lt 10 ]; then
    echo ""
    echo "  [WARN] Low frame count ($FRAME_COUNT). Expected ~100+."
    echo "  Check discovery_log.txt and SEVA output logs."
fi

echo ""
echo "  Discovery log: $STAGE_C_OUT/discovery_log.txt"
echo "  Next step: python -m src.train_splat (Stage D)"
echo "============================================"
