#!/bin/bash
set -e

# =============================================================================
# Pano2Splat: Stage A - Scaffold Generation
# Runs One2Scene scaffold inference on a single panorama, then extracts
# Gaussians, cameras, and scaffold renders into canonical format.
#
# Usage: ./scripts/run_stage_a.sh [path/to/panorama.jpg]
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ONE2SCENE_DIR="$PROJECT_ROOT/third_party/One2Scene"

# Input panorama (default: ./input/panorama.jpg)
INPUT_PANO="${1:-$PROJECT_ROOT/input/panorama.jpg}"

# Output directory
STAGE_A_OUT="$PROJECT_ROOT/pipeline_output/stage_a"

# Model checkpoint
SCAFFOLD_CKPT="$PROJECT_ROOT/models/one2scene_scaffold.ckpt"

echo "============================================"
echo "  Pano2Splat: Stage A - Scaffold Generation"
echo "============================================"
echo ""
echo "  Input panorama:  $INPUT_PANO"
echo "  One2Scene dir:   $ONE2SCENE_DIR"
echo "  Checkpoint:      $SCAFFOLD_CKPT"
echo "  Output:          $STAGE_A_OUT"
echo ""

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
if [ ! -f "$INPUT_PANO" ]; then
    echo "[ERROR] Input panorama not found: $INPUT_PANO"
    exit 1
fi

if [ ! -d "$ONE2SCENE_DIR" ]; then
    echo "[ERROR] One2Scene directory not found: $ONE2SCENE_DIR"
    echo "        Run: git submodule update --init --recursive"
    exit 1
fi

if [ ! -f "$SCAFFOLD_CKPT" ]; then
    echo "[ERROR] Scaffold checkpoint not found: $SCAFFOLD_CKPT"
    echo "        Run: bash scripts/download_weights.sh"
    exit 1
fi

# ---------------------------------------------------------------------------
# Prepare input
# ---------------------------------------------------------------------------
echo "[1/4] Preparing input panorama..."

# DatasetDemo reads panorama.jpg from the roots directory (default: demo_outputs/)
DEMO_DIR="$ONE2SCENE_DIR/demo_outputs"
mkdir -p "$DEMO_DIR"
cp "$INPUT_PANO" "$DEMO_DIR/panorama.jpg"
echo "  Panorama copied to: $DEMO_DIR/panorama.jpg"

# Create output directory
mkdir -p "$STAGE_A_OUT"

# ---------------------------------------------------------------------------
# Run One2Scene scaffold generation
# ---------------------------------------------------------------------------
echo ""
echo "[2/4] Running One2Scene scaffold generation..."
echo "  This typically takes 2-5 minutes on an A6000."
echo ""

START_TIME=$(date +%s)

cd "$ONE2SCENE_DIR"

# One2Scene entry point: src/main.py with Hydra config (+experiment=demo).
# The demo dataset reads panorama.jpg from roots dir, decomposes to cubemap,
# runs scaffold encoder, and saves data.pth + fused.ply.
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$ONE2SCENE_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Convert Git Bash paths to Windows paths for Python/OpenCV compatibility
WIN_STAGE_A_OUT=$(cygpath -w "$STAGE_A_OUT" 2>/dev/null || echo "$STAGE_A_OUT")
WIN_SCAFFOLD_CKPT=$(cygpath -w "$SCAFFOLD_CKPT" 2>/dev/null || echo "$SCAFFOLD_CKPT")

python src/main.py \
    +experiment=demo \
    mode=test \
    test.output_path="$WIN_STAGE_A_OUT/raw" \
    "dataset.re10k.roots=[./demo_outputs]" \
    checkpointing.load="$WIN_SCAFFOLD_CKPT"

cd "$PROJECT_ROOT"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "  Scaffold generation complete in $((ELAPSED / 60))m $((ELAPSED % 60))s"

# ---------------------------------------------------------------------------
# Discovery: find actual output files
# ---------------------------------------------------------------------------
echo ""
echo "[3/4] Discovering output files..."

DISCOVERY_LOG="$STAGE_A_OUT/discovery_log.txt"
echo "# Stage A Discovery Log - $(date -u +"%Y-%m-%dT%H:%M:%SZ")" > "$DISCOVERY_LOG"
echo "# Input: $INPUT_PANO" >> "$DISCOVERY_LOG"
echo "# Elapsed: $((ELAPSED / 60))m $((ELAPSED % 60))s" >> "$DISCOVERY_LOG"
echo "" >> "$DISCOVERY_LOG"

# Search One2Scene output directories for artifacts
for SEARCH_DIR in \
    "$STAGE_A_OUT/raw" \
    "$ONE2SCENE_DIR/outputs" \
    "$ONE2SCENE_DIR/demo_outputs" \
    "$ONE2SCENE_DIR/demo_case"; do
    if [ -d "$SEARCH_DIR" ]; then
        echo "  [DIR] $SEARCH_DIR" | tee -a "$DISCOVERY_LOG"
        find "$SEARCH_DIR" -type f \( \
            -name "*.ply" -o -name "*.pth" -o -name "*.pt" -o \
            -name "*.npy" -o -name "*.png" -o -name "*.json" -o \
            -name "*.mp4" \
        \) -exec ls -lh {} \; 2>/dev/null | tee -a "$DISCOVERY_LOG"
        echo "" | tee -a "$DISCOVERY_LOG"
    fi
done

# Locate the critical data.pth file
DATA_PTH=$(find "$STAGE_A_OUT/raw" "$ONE2SCENE_DIR/outputs" "$ONE2SCENE_DIR/demo_outputs" \
    -maxdepth 5 -name "data.pth" 2>/dev/null | head -1 || true)

FUSED_PLY=""
if [ -n "$DATA_PTH" ]; then
    echo "  Found data.pth: $DATA_PTH"
    # fused.ply should be in the same directory
    FUSED_PLY="$(dirname "$DATA_PTH")/fused.ply"
    if [ ! -f "$FUSED_PLY" ]; then
        FUSED_PLY=$(find "$(dirname "$DATA_PTH")/.." -maxdepth 3 -name "fused.ply" 2>/dev/null | head -1 || true)
    fi
    [ -n "$FUSED_PLY" ] && echo "  Found fused.ply: $FUSED_PLY"
else
    echo "  [WARN] data.pth not found in expected locations."
    echo "  Check discovery_log.txt for output file locations."
fi

# Copy data.pth to stage_a for downstream use
if [ -n "$DATA_PTH" ] && [ -f "$DATA_PTH" ]; then
    cp "$DATA_PTH" "$STAGE_A_OUT/data.pth"
    echo "  Copied data.pth to $STAGE_A_OUT/"
fi

# ---------------------------------------------------------------------------
# Extract scaffold data into canonical format
# ---------------------------------------------------------------------------
echo ""
echo "[4/4] Extracting scaffold data..."

EXTRACT_ARGS="--output_dir $STAGE_A_OUT"

if [ -f "$STAGE_A_OUT/data.pth" ]; then
    EXTRACT_ARGS="$EXTRACT_ARGS --data_pth $STAGE_A_OUT/data.pth"
elif [ -n "$DATA_PTH" ] && [ -f "$DATA_PTH" ]; then
    EXTRACT_ARGS="$EXTRACT_ARGS --data_pth $DATA_PTH"
else
    echo "[ERROR] Cannot find data.pth from Stage A output."
    echo "  Discovery log: $DISCOVERY_LOG"
    exit 1
fi

if [ -n "$FUSED_PLY" ] && [ -f "$FUSED_PLY" ]; then
    EXTRACT_ARGS="$EXTRACT_ARGS --fused_ply $FUSED_PLY"
fi

cd "$PROJECT_ROOT"
python -m src.extract_scaffold $EXTRACT_ARGS

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "  Stage A Results"
echo "============================================"
echo ""

TOTAL_ELAPSED=$(($(date +%s) - START_TIME))
echo "  Total time: $((TOTAL_ELAPSED / 60))m $((TOTAL_ELAPSED % 60))s"
echo ""

for F in "scaffold_gaussians.ply" "data.pth" "extraction_info.json"; do
    if [ -f "$STAGE_A_OUT/$F" ]; then
        SIZE=$(du -h "$STAGE_A_OUT/$F" | cut -f1)
        echo "  [OK]   $F ($SIZE)"
    else
        echo "  [WARN] $F NOT FOUND"
    fi
done

for D in "anchor_views" "anchor_cameras" "scaffold_renders"; do
    if [ -d "$STAGE_A_OUT/$D" ]; then
        COUNT=$(find "$STAGE_A_OUT/$D" -type f 2>/dev/null | wc -l)
        echo "  [OK]   $D/ ($COUNT files)"
    else
        echo "  [WARN] $D/ NOT FOUND"
    fi
done

echo ""
echo "  Discovery log: $STAGE_A_OUT/discovery_log.txt"
echo "  Next step: bash scripts/run_stage_c.sh"
echo "============================================"
