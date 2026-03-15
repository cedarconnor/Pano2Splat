"""Read/write standard 3DGS .ply format.

Compatible with SuperSplat, Unity, and Unreal 3DGS plugins.
Uses the plyfile library for robust binary PLY handling.

SH coefficient storage:
    Standard 3DGS PLY format: channel-first [R0..R14, G0..G14, B0..B14]
    Internal gsplat format:   interleaved   [R0,G0,B0, R1,G1,B1, ...]

    load_ply converts channel-first (PLY) → interleaved (internal).
    save_ply converts interleaved (internal) → channel-first (PLY).
"""

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def _sh_channel_first_to_interleaved(sh_rest: np.ndarray) -> np.ndarray:
    """Convert SH rest from channel-first PLY format to interleaved internal format.

    PLY:      (N, 45) as [R0..R14, G0..G14, B0..B14]
    Internal: (N, 45) as [R0,G0,B0, R1,G1,B1, ..., R14,G14,B14]
    """
    n_rest = sh_rest.shape[1]
    if n_rest == 0:
        return sh_rest
    n_bands = n_rest // 3
    # (N, 45) → (N, 3, bands) channel-first → (N, bands, 3) interleaved → (N, 45)
    return sh_rest.reshape(-1, 3, n_bands).transpose(0, 2, 1).reshape(-1, n_rest)


def _sh_interleaved_to_channel_first(sh_rest: np.ndarray) -> np.ndarray:
    """Convert SH rest from interleaved internal format to channel-first PLY format.

    Internal: (N, 45) as [R0,G0,B0, R1,G1,B1, ..., R14,G14,B14]
    PLY:      (N, 45) as [R0..R14, G0..G14, B0..B14]
    """
    n_rest = sh_rest.shape[1]
    if n_rest == 0:
        return sh_rest
    n_bands = n_rest // 3
    # (N, 45) → (N, bands, 3) interleaved → (N, 3, bands) channel-first → (N, 45)
    return sh_rest.reshape(-1, n_bands, 3).transpose(0, 2, 1).reshape(-1, n_rest)


def load_ply(path: str) -> dict:
    """Load a 3D Gaussian Splatting PLY file.

    Returns dict with keys:
        'positions':  (N, 3)  float32  — xyz
        'scales':     (N, 3)  float32  — log-space
        'rotations':  (N, 4)  float32  — quaternion wxyz
        'opacities':  (N, 1)  float32  — logit-space
        'sh_dc':      (N, 3)  float32  — DC spherical harmonics
        'sh_rest':    (N, K)  float32  — higher-order SH coeffs (interleaved)
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
        # Convert from standard channel-first PLY format to interleaved internal format
        sh_rest = _sh_channel_first_to_interleaved(sh_rest)
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

    Property order matches the standard 3DGS format:
        x,y,z, nx,ny,nz, f_dc_*, f_rest_*, opacity, scale_*, rot_*
    SH rest is converted from interleaved to channel-first for standard compatibility.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    positions = gaussians["positions"]  # (N, 3)
    scales = gaussians["scales"]        # (N, 3)
    rotations = gaussians["rotations"]  # (N, 4)
    opacities = gaussians["opacities"]  # (N, 1)
    sh_dc = gaussians["sh_dc"]          # (N, 3)
    sh_rest = gaussians["sh_rest"]      # (N, K) interleaved

    n = positions.shape[0]
    n_sh_rest = sh_rest.shape[1] if sh_rest.ndim == 2 else 0

    # Convert SH rest from interleaved to channel-first for standard PLY format
    if n_sh_rest > 0:
        sh_rest_ply = _sh_interleaved_to_channel_first(sh_rest)
    else:
        sh_rest_ply = sh_rest

    # Build dtype in standard 3DGS property order
    props = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
    ]
    for i in range(n_sh_rest):
        props.append((f"f_rest_{i}", "f4"))
    props.append(("opacity", "f4"))
    props.extend([
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ])

    vertex_data = np.empty(n, dtype=props)

    vertex_data["x"] = positions[:, 0]
    vertex_data["y"] = positions[:, 1]
    vertex_data["z"] = positions[:, 2]
    vertex_data["nx"] = 0.0
    vertex_data["ny"] = 0.0
    vertex_data["nz"] = 0.0
    vertex_data["f_dc_0"] = sh_dc[:, 0]
    vertex_data["f_dc_1"] = sh_dc[:, 1]
    vertex_data["f_dc_2"] = sh_dc[:, 2]
    for i in range(n_sh_rest):
        vertex_data[f"f_rest_{i}"] = sh_rest_ply[:, i]
    vertex_data["opacity"] = opacities.ravel()
    vertex_data["scale_0"] = scales[:, 0]
    vertex_data["scale_1"] = scales[:, 1]
    vertex_data["scale_2"] = scales[:, 2]
    vertex_data["rot_0"] = rotations[:, 0]
    vertex_data["rot_1"] = rotations[:, 1]
    vertex_data["rot_2"] = rotations[:, 2]
    vertex_data["rot_3"] = rotations[:, 3]

    el = PlyElement.describe(vertex_data, "vertex")
    PlyData([el], byte_order="<").write(str(path))
