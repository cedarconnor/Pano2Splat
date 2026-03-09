import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.ply_io import load_ply
d = load_ply(sys.argv[1])
for k, v in d.items():
    print(f"  {k}: {v.shape} ({v.dtype})")
