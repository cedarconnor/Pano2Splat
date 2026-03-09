"""Launcher for One2Scene that patches attention backends on Windows.

On Windows without xformers, PyTorch's scaled_dot_product_attention may fail
with "No available kernel" because flash/mem-efficient kernels require xformers.
This script forces the math SDP backend before importing One2Scene.
"""
import sys
import os
import torch

# Force math-only SDP backend (works without xformers)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

# Run One2Scene's main.py
one2scene_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
one2scene_dir = os.path.join(one2scene_dir, "third_party", "One2Scene")

# Add One2Scene to sys.path so 'from src...' imports work
if one2scene_dir not in sys.path:
    sys.path.insert(0, one2scene_dir)

# Change to One2Scene directory for Hydra config resolution
os.chdir(one2scene_dir)

# Import and run
exec(open(os.path.join(one2scene_dir, "src", "main.py")).read())
