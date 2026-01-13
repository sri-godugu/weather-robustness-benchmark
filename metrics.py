import torch
import numpy as np

@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    return (pred == y).float().mean().item()

@torch.no_grad()
def mean_confidence_and_entropy(logits: torch.Tensor) -> tuple[float, float]:
    probs = torch.softmax(logits, dim=1)
    conf = probs.max(dim=1).values
    entropy = -(probs * torch.log(torch.clamp(probs, 1e-12, 1.0))).sum(dim=1)
    return conf.mean().item(), entropy.mean().item()

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
