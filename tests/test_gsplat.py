"""Quick test of gsplat rasterization with synthetic data."""

import math

import torch
from gsplat import rasterization


def test_gsplat_rasterization() -> None:
    n = 500
    means = torch.randn(n, 3).cuda() * 0.5
    quats = torch.randn(n, 4).cuda()
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = torch.exp(torch.randn(n, 3).cuda() * 0.5 - 2)
    opacities = torch.sigmoid(torch.randn(n).cuda())
    colors = torch.rand(n, 1, 3).cuda()

    viewmat = torch.eye(4).cuda().unsqueeze(0)
    viewmat[0, 2, 3] = -3.0

    fov, width, height = 70, 256, 256
    fx = width / (2 * math.tan(math.radians(fov / 2)))
    k = torch.tensor(
        [[fx, 0, width / 2], [0, fx, height / 2], [0, 0, 1]],
        dtype=torch.float32,
    ).cuda().unsqueeze(0)

    renders, alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat,
        Ks=k,
        width=width,
        height=height,
        sh_degree=0,
        render_mode="RGB+ED",
    )

    print(f"Render: {renders.shape}")
    print(f"RGB range: [{renders[..., :3].min():.3f}, {renders[..., :3].max():.3f}]")
    print(f"Depth range: [{renders[..., 3].min():.3f}, {renders[..., 3].max():.3f}]")

    assert renders.shape == (1, height, width, 4)
    assert torch.isfinite(renders).all()
    assert alphas.shape == (1, height, width, 1)
