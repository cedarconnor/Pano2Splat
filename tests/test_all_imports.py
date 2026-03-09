"""Verify all key libraries import correctly."""
import torch
print(f"torch {torch.__version__} (CUDA {torch.version.cuda})")

import xformers
print(f"xformers {xformers.__version__}")

import triton
print(f"triton {triton.__version__}")

import flash_attn
print(f"flash_attn {flash_attn.__version__}")

import sageattention
try:
    print(f"sageattention {sageattention.__version__}")
except AttributeError:
    print("sageattention OK (no __version__)")

from gsplat.cuda._backend import _C
print(f"gsplat CUDA: OK ({len([x for x in dir(_C) if not x.startswith('_')])} ops)")

print("\nAll imports OK")
