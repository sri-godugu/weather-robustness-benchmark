import torch
import torch.nn.functional as F
from .registry import register

def _gaussian_kernel(k: int, sigma: float, device):
    ax = torch.arange(k, device=device) - (k - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel

@register("blur")
def blur(x: torch.Tensor, severity: int) -> torch.Tensor:
    s = max(1, min(5, int(severity)))
    k = [3, 5, 7, 9, 11][s-1]
    sigma = [0.8, 1.0, 1.3, 1.6, 2.0][s-1]
    B, C, H, W = x.shape
    kernel = _gaussian_kernel(k, sigma, x.device).view(1, 1, k, k)
    kernel = kernel.repeat(C, 1, 1, 1)  # depthwise
    y = F.conv2d(x, kernel, padding=k//2, groups=C)
    return torch.clamp(y, 0, 1)
