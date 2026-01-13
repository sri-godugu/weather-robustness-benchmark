import torch
from .registry import register

@register("lowlight")
def lowlight(x: torch.Tensor, severity: int) -> torch.Tensor:
    s = max(1, min(5, int(severity)))
    gamma = [1.4, 1.7, 2.0, 2.4, 2.8][s-1]     # darker
    noise = [0.01, 0.02, 0.03, 0.05, 0.08][s-1]  # gaussian
    y = torch.clamp(x ** gamma, 0, 1)
    y = y + noise * torch.randn_like(y)
    return torch.clamp(y, 0, 1)
