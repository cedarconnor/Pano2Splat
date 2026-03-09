#!/bin/bash
set -e

# =============================================================================
# Pano2Splat: Download Model Weights
# Downloads One2Scene scaffold and denoise checkpoints from HuggingFace,
# plus the SDXL-VAE required by the SEVA denoiser.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_ROOT/models"
ONE2SCENE_DIR="$PROJECT_ROOT/third_party/One2Scene"

echo "============================================"
echo "  Pano2Splat: Download Model Weights"
echo "  Project root: $PROJECT_ROOT"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
if ! command -v huggingface-cli &> /dev/null; then
    echo "[ERROR] huggingface-cli not found."
    echo "        Install with: pip install huggingface_hub[cli]"
    echo "        Then login:   huggingface-cli login"
    exit 1
fi

# Check that huggingface-cli is authenticated
if ! huggingface-cli whoami &> /dev/null; then
    echo "[WARN] huggingface-cli does not appear to be logged in."
    echo "       Some downloads may fail if the repo requires authentication."
    echo "       Run: huggingface-cli login --token YOUR_TOKEN"
    echo ""
fi

# ---------------------------------------------------------------------------
# Create models directory
# ---------------------------------------------------------------------------
echo "[1/3] Setting up models directory..."
echo "--------------------------------------------------------------"
mkdir -p "$MODELS_DIR"
echo "[OK]   Models directory: $MODELS_DIR"

# ---------------------------------------------------------------------------
# Download One2Scene scaffold checkpoint
# ---------------------------------------------------------------------------
echo ""
echo "[2/3] Downloading One2Scene model weights..."
echo "--------------------------------------------------------------"

SCAFFOLD_CKPT="$MODELS_DIR/one2scene_scaffold.ckpt"
DENOISE_CKPT="$MODELS_DIR/one2scene_denoise.ckpt"

if [ -f "$SCAFFOLD_CKPT" ]; then
    SCAFFOLD_SIZE=$(du -h "$SCAFFOLD_CKPT" | cut -f1)
    echo "[SKIP] one2scene_scaffold.ckpt already exists ($SCAFFOLD_SIZE)"
else
    echo "       Downloading one2scene_scaffold.ckpt from mutou0308/One2Scene..."
    echo "       (This may take several minutes depending on connection speed)"
    huggingface-cli download mutou0308/One2Scene one2scene_scaffold.ckpt \
        --repo-type dataset \
        --local-dir "$MODELS_DIR"
    echo "[OK]   one2scene_scaffold.ckpt downloaded."
fi

if [ -f "$DENOISE_CKPT" ]; then
    DENOISE_SIZE=$(du -h "$DENOISE_CKPT" | cut -f1)
    echo "[SKIP] one2scene_denoise.ckpt already exists ($DENOISE_SIZE)"
else
    echo "       Downloading one2scene_denoise.ckpt from mutou0308/One2Scene..."
    echo "       (This is a large file ~19GB, please be patient)"
    huggingface-cli download mutou0308/One2Scene one2scene_denoise.ckpt \
        --repo-type dataset \
        --local-dir "$MODELS_DIR"
    echo "[OK]   one2scene_denoise.ckpt downloaded."
fi

# ---------------------------------------------------------------------------
# Download SDXL-VAE for SEVA denoiser
# ---------------------------------------------------------------------------
echo ""
echo "[3/3] Downloading SDXL-VAE..."
echo "--------------------------------------------------------------"

SDXL_VAE_DIR="$ONE2SCENE_DIR/src_denoise/sdxl-vae"

if [ ! -d "$ONE2SCENE_DIR/src_denoise" ]; then
    echo "[ERROR] One2Scene src_denoise directory not found at:"
    echo "        $ONE2SCENE_DIR/src_denoise"
    echo "        Make sure One2Scene is properly set up first."
    echo "        Run: scripts/setup_one2scene.sh"
    exit 1
fi

if [ -d "$SDXL_VAE_DIR" ] && [ "$(ls -A "$SDXL_VAE_DIR" 2>/dev/null)" ]; then
    echo "[SKIP] sdxl-vae directory already exists and is non-empty."
    echo "       Location: $SDXL_VAE_DIR"
else
    echo "       Cloning stabilityai/sdxl-vae from HuggingFace..."
    echo "       (This downloads the VAE model files)"
    cd "$ONE2SCENE_DIR/src_denoise"
    git clone https://huggingface.co/stabilityai/sdxl-vae sdxl-vae
    echo "[OK]   sdxl-vae downloaded."
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "  Download Summary"
echo "============================================"
echo ""

echo "  Model weights directory: $MODELS_DIR"
echo ""

if [ -f "$SCAFFOLD_CKPT" ]; then
    SCAFFOLD_SIZE=$(du -h "$SCAFFOLD_CKPT" | cut -f1)
    echo "  [OK] one2scene_scaffold.ckpt  ($SCAFFOLD_SIZE)"
else
    echo "  [!!] one2scene_scaffold.ckpt  MISSING"
fi

if [ -f "$DENOISE_CKPT" ]; then
    DENOISE_SIZE=$(du -h "$DENOISE_CKPT" | cut -f1)
    echo "  [OK] one2scene_denoise.ckpt   ($DENOISE_SIZE)"
else
    echo "  [!!] one2scene_denoise.ckpt   MISSING"
fi

if [ -d "$SDXL_VAE_DIR" ] && [ "$(ls -A "$SDXL_VAE_DIR" 2>/dev/null)" ]; then
    echo "  [OK] sdxl-vae                 ($SDXL_VAE_DIR)"
else
    echo "  [!!] sdxl-vae                 MISSING"
fi

echo ""
echo "  Next steps:"
echo "    1. Place your panorama in input/panorama.jpg"
echo "    2. Run scripts/run_stage_a.sh to generate scaffold"
echo "============================================"
