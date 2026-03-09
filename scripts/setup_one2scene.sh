#!/bin/bash
set -e

# =============================================================================
# Pano2Splat: One2Scene Environment Setup
# Automates installation per design doc section 4.2
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ONE2SCENE_DIR="$PROJECT_ROOT/third_party/One2Scene"

echo "============================================"
echo "  Pano2Splat: One2Scene Setup"
echo "  Project root: $PROJECT_ROOT"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
if [ ! -d "$ONE2SCENE_DIR" ]; then
    echo "[ERROR] One2Scene directory not found at: $ONE2SCENE_DIR"
    echo "        Expected as a git submodule in third_party/One2Scene."
    echo "        Run: git submodule update --init --recursive"
    exit 1
fi

if ! command -v conda &> /dev/null; then
    echo "[ERROR] conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 1: Install HunyuanWorld and create conda environment
# ---------------------------------------------------------------------------
echo ""
echo "[1/6] Installing HunyuanWorld and creating conda environment..."
echo "--------------------------------------------------------------"

HUNYUAN_DIR="$ONE2SCENE_DIR/third_party/HunyuanWorld-1.0"

if [ ! -d "$HUNYUAN_DIR" ]; then
    echo "[ERROR] HunyuanWorld-1.0 not found at: $HUNYUAN_DIR"
    echo "        Expected in One2Scene/third_party/HunyuanWorld-1.0"
    exit 1
fi

HUNYUAN_YAML="$HUNYUAN_DIR/docker/HunyuanWorld.yaml"
if [ ! -f "$HUNYUAN_YAML" ]; then
    echo "[ERROR] HunyuanWorld.yaml not found at: $HUNYUAN_YAML"
    exit 1
fi

# Check if conda env already exists
if conda env list | grep -q "one2scene"; then
    echo "[SKIP] Conda environment 'one2scene' already exists."
    echo "       To recreate, run: conda env remove -n one2scene"
else
    echo "       Creating conda env from $HUNYUAN_YAML ..."
    conda env create -f "$HUNYUAN_YAML"
    echo "[OK]   Conda environment 'one2scene' created."
fi

# Activate the environment for subsequent installs
echo "       Activating 'one2scene' environment..."
eval "$(conda shell.bash hook)"
conda activate one2scene
echo "[OK]   Environment 'one2scene' activated."

# ---------------------------------------------------------------------------
# Step 2: Install Real-ESRGAN
# ---------------------------------------------------------------------------
echo ""
echo "[2/6] Installing Real-ESRGAN..."
echo "--------------------------------------------------------------"

REALESRGAN_DIR="$ONE2SCENE_DIR/third_party/Real-ESRGAN"

if [ -d "$REALESRGAN_DIR" ]; then
    echo "[SKIP] Real-ESRGAN directory already exists at: $REALESRGAN_DIR"
else
    echo "       Cloning Real-ESRGAN into $ONE2SCENE_DIR/third_party/ ..."
    cd "$ONE2SCENE_DIR/third_party"
    git clone https://github.com/xinntao/Real-ESRGAN.git
fi

cd "$REALESRGAN_DIR"
echo "       Installing Real-ESRGAN dependencies..."
pip install basicsr-fixed facexlib gfpgan
pip install -r requirements.txt
echo "       Running setup.py develop..."
python setup.py develop
echo "[OK]   Real-ESRGAN installed."

# ---------------------------------------------------------------------------
# Step 3: Install ZIM
# ---------------------------------------------------------------------------
echo ""
echo "[3/6] Installing ZIM..."
echo "--------------------------------------------------------------"

ZIM_DIR="$ONE2SCENE_DIR/third_party/ZIM"

if [ -d "$ZIM_DIR" ]; then
    echo "[SKIP] ZIM directory already exists at: $ZIM_DIR"
else
    echo "       Cloning ZIM into $ONE2SCENE_DIR/third_party/ ..."
    cd "$ONE2SCENE_DIR/third_party"
    git clone https://github.com/naver-ai/ZIM.git
fi

cd "$ZIM_DIR"
echo "       Installing ZIM package..."
pip install -e .

# Download ZIM model weights
ZIM_WEIGHTS_DIR="$ZIM_DIR/zim_vit_l_2092"
mkdir -p "$ZIM_WEIGHTS_DIR"

if [ -f "$ZIM_WEIGHTS_DIR/encoder.onnx" ]; then
    echo "[SKIP] ZIM encoder.onnx already exists."
else
    echo "       Downloading ZIM encoder.onnx..."
    wget -q --show-progress -O "$ZIM_WEIGHTS_DIR/encoder.onnx" \
        "https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/encoder.onnx"
    echo "[OK]   encoder.onnx downloaded."
fi

if [ -f "$ZIM_WEIGHTS_DIR/decoder.onnx" ]; then
    echo "[SKIP] ZIM decoder.onnx already exists."
else
    echo "       Downloading ZIM decoder.onnx..."
    wget -q --show-progress -O "$ZIM_WEIGHTS_DIR/decoder.onnx" \
        "https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/decoder.onnx"
    echo "[OK]   decoder.onnx downloaded."
fi

echo "[OK]   ZIM installed."

# ---------------------------------------------------------------------------
# Step 4: Install One2Scene Python dependencies
# ---------------------------------------------------------------------------
echo ""
echo "[4/6] Installing One2Scene requirements..."
echo "--------------------------------------------------------------"

cd "$ONE2SCENE_DIR"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "[OK]   One2Scene requirements installed."
else
    echo "[WARN] requirements.txt not found at $ONE2SCENE_DIR/requirements.txt"
    echo "       Skipping. You may need to install dependencies manually."
fi

# ---------------------------------------------------------------------------
# Step 5: Install gsplat (for Stage D splat optimization)
# ---------------------------------------------------------------------------
echo ""
echo "[5/6] Installing gsplat..."
echo "--------------------------------------------------------------"

if python -c "import gsplat" 2>/dev/null; then
    echo "[SKIP] gsplat already installed."
else
    pip install gsplat
    echo "[OK]   gsplat installed."
fi

# ---------------------------------------------------------------------------
# Step 6: Verify installation
# ---------------------------------------------------------------------------
echo ""
echo "[6/6] Verifying installation..."
echo "--------------------------------------------------------------"

ERRORS=0

# Check conda env
if ! conda env list | grep -q "one2scene"; then
    echo "[FAIL] Conda env 'one2scene' not found."
    ERRORS=$((ERRORS + 1))
else
    echo "[OK]   Conda env 'one2scene' exists."
fi

# Check Real-ESRGAN
if [ -d "$REALESRGAN_DIR" ] && [ -f "$REALESRGAN_DIR/setup.py" ]; then
    echo "[OK]   Real-ESRGAN directory present."
else
    echo "[FAIL] Real-ESRGAN not properly installed."
    ERRORS=$((ERRORS + 1))
fi

# Check ZIM
if [ -d "$ZIM_DIR" ] && [ -f "$ZIM_WEIGHTS_DIR/encoder.onnx" ] && [ -f "$ZIM_WEIGHTS_DIR/decoder.onnx" ]; then
    echo "[OK]   ZIM installed with model weights."
else
    echo "[FAIL] ZIM not properly installed or weights missing."
    ERRORS=$((ERRORS + 1))
fi

# Check gsplat
if python -c "import gsplat" 2>/dev/null; then
    echo "[OK]   gsplat importable."
else
    echo "[FAIL] gsplat not importable."
    ERRORS=$((ERRORS + 1))
fi

echo ""
echo "============================================"
if [ $ERRORS -eq 0 ]; then
    echo "  Setup complete. All checks passed."
else
    echo "  Setup finished with $ERRORS error(s)."
    echo "  Review the output above and fix issues."
fi
echo ""
echo "  Next steps:"
echo "    1. Run scripts/download_weights.sh to get model weights"
echo "    2. Place your panorama in input/panorama.jpg"
echo "    3. Run scripts/run_stage_a.sh to generate scaffold"
echo "============================================"
