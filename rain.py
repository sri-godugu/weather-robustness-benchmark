import torch
from .registry import register

@register("rain")
def rain(x: torch.Tensor, severity: int) -> torch.Tensor:
    # simple rain streak overlay in tensor space
    s = max(1, min(5, int(severity)))
    density = [30, 60, 90, 130, 180][s-1]   # number of streaks per image
    strength = [0.08, 0.10, 0.12, 0.15, 0.18][s-1]
    length = [8, 10, 12, 14, 16][s-1]

    B, C, H, W = x.shape
    y = x.clone()
    # draw streaks on a single-channel mask, then add to RGB
    mask = torch.zeros((B, 1, H, W), device=x.device, dtype=x.dtype)

    for b in range(B):
        # random starting points
        xs = torch.randint(0, W, (density,), device=x.device)
        ys = torch.randint(0, H, (density,), device=x.device)
        for i in range(density):
            x0, y0 = xs[i].item(), ys[i].item()
            # diagonal down-right
            for t in range(length):
                xx = x0 + t
                yy = y0 + t
                if 0 <= xx < W and 0 <= yy < H:
                    mask[b, 0, yy, xx] = 1.0

    # blur mask slightly by cheap neighborhood averaging (no opencv)
    mask = torch.nn.functional.avg_pool2d(mask, kernel_size=3, stride=1, padding=1)
    y = y + strength * mask.repeat(1, 3, 1, 1)
    return torch.clamp(y, 0, 1)
