"""Quick test of gsplat rasterization with synthetic data."""
import math
import torch
from gsplat import rasterization

N = 500
means = torch.randn(N, 3).cuda() * 0.5
quats = torch.randn(N, 4).cuda()
quats = quats / quats.norm(dim=-1, keepdim=True)
scales = torch.exp(torch.randn(N, 3).cuda() * 0.5 - 2)
opacities = torch.sigmoid(torch.randn(N).cuda())
colors = torch.rand(N, 1, 3).cuda()

viewmat = torch.eye(4).cuda().unsqueeze(0)
viewmat[0, 2, 3] = -3.0

fov, W, H = 70, 256, 256
fx = W / (2 * math.tan(math.radians(fov / 2)))
K = torch.tensor(
    [[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]],
    dtype=torch.float32,
).cuda().unsqueeze(0)

renders, alphas, info = rasterization(
    means=means,
    quats=quats,
    scales=scales,
    opacities=opacities,
    colors=colors,
    viewmats=viewmat,
    Ks=K,
    width=W,
    height=H,
    sh_degree=0,
    render_mode="RGB+ED",
)
print(f"Render: {renders.shape}")
print(f"RGB range: [{renders[..., :3].min():.3f}, {renders[..., :3].max():.3f}]")
print(f"Depth range: [{renders[..., 3].min():.3f}, {renders[..., 3].max():.3f}]")
print("gsplat rasterization OK")
