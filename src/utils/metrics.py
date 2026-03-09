"""Quality metrics for 3D Gaussian Splatting evaluation.

Computes PSNR, SSIM, and LPIPS on GPU tensors.
"""

import torch
from pytorch_msssim import ssim as _ssim_fn

# Module-level cache for the LPIPS network (lazy init).
_lpips_model = {}


def _ensure_4d(t: torch.Tensor) -> torch.Tensor:
    """Promote (C, H, W) tensors to (1, C, H, W)."""
    if t.ndim == 3:
        return t.unsqueeze(0)
    return t


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR between two images.

    Args:
        pred:   (B, C, H, W) or (C, H, W) tensor in [0, 1].
        target: same shape as *pred*.

    Returns:
        PSNR in dB (float, higher is better).
    """
    pred = _ensure_4d(pred)
    target = _ensure_4d(target)
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0.0:
        return float("inf")
    return -10.0 * torch.log10(torch.tensor(mse, dtype=torch.float64)).item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """SSIM (structural similarity).

    Args:
        pred:   (B, C, H, W) or (C, H, W) tensor in [0, 1].
        target: same shape as *pred*.

    Returns:
        SSIM value (float in [0, 1], higher is better).
    """
    pred = _ensure_4d(pred)
    target = _ensure_4d(target)
    return _ssim_fn(pred, target, data_range=1.0, size_average=True).item()


def compute_lpips(
    pred: torch.Tensor, target: torch.Tensor, net: str = "vgg"
) -> float:
    """LPIPS perceptual distance.

    The underlying network is lazily initialised on first call and cached
    for subsequent calls with the same *net* type.

    Args:
        pred:   (B, C, H, W) or (C, H, W) tensor in [0, 1].
        target: same shape as *pred*.
        net:    backbone — ``'vgg'`` (default) or ``'alex'``.

    Returns:
        LPIPS distance (float, lower is better).
    """
    import lpips as _lpips_lib  # deferred to avoid import cost when unused

    pred = _ensure_4d(pred)
    target = _ensure_4d(target)

    if net not in _lpips_model:
        model = _lpips_lib.LPIPS(net=net, verbose=False)
        model = model.to(pred.device)
        model.eval()
        _lpips_model[net] = model

    model = _lpips_model[net]
    # Move model to the same device as inputs if needed.
    if next(model.parameters()).device != pred.device:
        model = model.to(pred.device)
        _lpips_model[net] = model

    with torch.no_grad():
        # lpips expects inputs in [-1, 1]; normalize=True handles [0, 1] -> [-1, 1].
        dist = model(pred, target, normalize=True)
    return dist.mean().item()


def compute_all_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Compute PSNR, SSIM, and LPIPS in one call.

    Args:
        pred:   (B, C, H, W) or (C, H, W) tensor in [0, 1].
        target: same shape as *pred*.

    Returns:
        ``{'psnr': float, 'ssim': float, 'lpips': float}``
    """
    return {
        "psnr": compute_psnr(pred, target),
        "ssim": compute_ssim(pred, target),
        "lpips": compute_lpips(pred, target),
    }
