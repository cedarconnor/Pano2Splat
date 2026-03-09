"""Read/write standard 3DGS .ply format.

Compatible with SuperSplat, Unity, and Unreal 3DGS plugins.
Uses the plyfile library for robust binary PLY handling.
"""

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def load_ply(path: str) -> dict:
    """Load a 3D Gaussian Splatting PLY file.

    Returns dict with keys:
        'positions':  (N, 3)  float32  — xyz
        'scales':     (N, 3)  float32  — log-space
        'rotations':  (N, 4)  float32  — quaternion wxyz
        'opacities':  (N, 1)  float32  — logit-space
        'sh_dc':      (N, 3)  float32  — DC spherical harmonics
        'sh_rest':    (N, K)  float32  — higher-order SH coeffs
                      K = 45 for degree 3, 24 for degree 2, 9 for degree 1, 0 for degree 0
    All values are numpy arrays.
    """
    plydata = PlyData.read(str(path))
    vertex = plydata["vertex"]
    n = vertex.count

    # Positions
    positions = np.stack(
        [vertex["x"], vertex["y"], vertex["z"]], axis=-1
    ).astype(np.float32)

    # Scales
    scales = np.stack(
        [vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=-1
    ).astype(np.float32)

    # Rotations (wxyz)
    rotations = np.stack(
        [vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]],
        axis=-1,
    ).astype(np.float32)

    # Opacities
    opacities = vertex["opacity"].astype(np.float32).reshape(n, 1)

    # SH DC term
    sh_dc = np.stack(
        [vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=-1
    ).astype(np.float32)

    # SH higher-order coefficients — collect however many exist
    sh_rest_cols = []
    idx = 0
    while True:
        name = f"f_rest_{idx}"
        if name in vertex.data.dtype.names:
            sh_rest_cols.append(vertex[name].astype(np.float32))
            idx += 1
        else:
            break

    if sh_rest_cols:
        sh_rest = np.stack(sh_rest_cols, axis=-1)
    else:
        sh_rest = np.zeros((n, 0), dtype=np.float32)

    return {
        "positions": positions,
        "scales": scales,
        "rotations": rotations,
        "opacities": opacities,
        "sh_dc": sh_dc,
        "sh_rest": sh_rest,
    }


def save_ply(path: str, gaussians: dict) -> None:
    """Save a 3D Gaussian Splatting PLY file (binary little-endian).

    *gaussians* dict has the same keys as :func:`load_ply` output.
    ``sh_rest`` may have 0, 9, 24, or 45 columns (SH degree 0-3).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    positions = gaussians["positions"]  # (N, 3)
    scales = gaussians["scales"]        # (N, 3)
    rotations = gaussians["rotations"]  # (N, 4)
    opacities = gaussians["opacities"]  # (N, 1)
    sh_dc = gaussians["sh_dc"]          # (N, 3)
    sh_rest = gaussians["sh_rest"]      # (N, K)

    n = positions.shape[0]
    n_sh_rest = sh_rest.shape[1] if sh_rest.ndim == 2 else 0

    # Build dtype
    props = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
        ("opacity", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
    ]
    for i in range(n_sh_rest):
        props.append((f"f_rest_{i}", "f4"))

    vertex_data = np.empty(n, dtype=props)

    vertex_data["x"] = positions[:, 0]
    vertex_data["y"] = positions[:, 1]
    vertex_data["z"] = positions[:, 2]
    vertex_data["scale_0"] = scales[:, 0]
    vertex_data["scale_1"] = scales[:, 1]
    vertex_data["scale_2"] = scales[:, 2]
    vertex_data["rot_0"] = rotations[:, 0]
    vertex_data["rot_1"] = rotations[:, 1]
    vertex_data["rot_2"] = rotations[:, 2]
    vertex_data["rot_3"] = rotations[:, 3]
    vertex_data["opacity"] = opacities.ravel()
    vertex_data["f_dc_0"] = sh_dc[:, 0]
    vertex_data["f_dc_1"] = sh_dc[:, 1]
    vertex_data["f_dc_2"] = sh_dc[:, 2]
    for i in range(n_sh_rest):
        vertex_data[f"f_rest_{i}"] = sh_rest[:, i]

    el = PlyElement.describe(vertex_data, "vertex")
    PlyData([el], byte_order="<").write(str(path))
