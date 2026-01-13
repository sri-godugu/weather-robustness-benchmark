import torch
from .registry import register

@register("snow")
def snow(x: torch.Tensor, severity: int) -> torch.Tensor:
    s = max(1, min(5, int(severity)))
    p = [0.01, 0.02, 0.035, 0.055, 0.08][s-1]     # snow density
    strength = [0.25, 0.30, 0.35, 0.45, 0.55][s-1]

    B, C, H, W = x.shape
    flakes = (torch.rand((B, 1, H, W), device=x.device) < p).float()
    flakes = torch.nn.functional.avg_pool2d(flakes, kernel_size=3, stride=1, padding=1)
    y = x * (1 - strength * flakes) + strength * flakes  # brighten where flakes exist
    return torch.clamp(y, 0, 1)
