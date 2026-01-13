import torch
from .registry import register

@register("fog")
def fog(x: torch.Tensor, severity: int) -> torch.Tensor:
    # severity: 1..5
    s = max(1, min(5, int(severity)))
    alpha = [0.08, 0.14, 0.22, 0.32, 0.45][s-1]  # veiling strength
    gamma = [0.95, 0.90, 0.85, 0.78, 0.70][s-1]  # low-contrast
    veil = torch.ones_like(x) * 0.85  # slightly off-white haze
    y = (1 - alpha) * x + alpha * veil
    y = torch.clamp(y, 0, 1)
    y = torch.clamp(y ** gamma, 0, 1)
    return y
