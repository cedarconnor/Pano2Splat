#!/bin/bash
set -e

# Pano2Splat — One-shot environment setup
# Run once after cloning the repository.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Pano2Splat Setup ==="

# 1. Initialize and update git submodules
echo "[1/4] Initializing git submodules..."
git submodule update --init --recursive

# 2. Install One2Scene dependencies
echo "[2/4] Setting up One2Scene environment..."
bash scripts/setup_one2scene.sh

# 3. Install Pano2Splat Python dependencies
echo "[3/4] Installing Pano2Splat dependencies..."
pip install -r requirements.txt

# 4. Download model weights
echo "[4/4] Downloading model weights..."
bash scripts/download_weights.sh

# Create output directories
mkdir -p pipeline_output/{stage_a,stage_b,stage_c,stage_d,stage_e,stage_f}
mkdir -p input

echo ""
echo "=== Setup complete ==="
echo "Place your panorama at: input/panorama.jpg"
echo "Run the pipeline with: bash scripts/run_pipeline.sh"
