from typing import Callable, Dict
import torch

CORRUPTIONS: Dict[str, Callable[[torch.Tensor, int], torch.Tensor]] = {}

def register(name: str):
    def deco(fn):
        CORRUPTIONS[name] = fn
        return fn
    return deco

def get_corruption(name: str):
    if name not in CORRUPTIONS:
        raise KeyError(f"Unknown corruption: {name}. Available: {list(CORRUPTIONS.keys())}")
    return CORRUPTIONS[name]
