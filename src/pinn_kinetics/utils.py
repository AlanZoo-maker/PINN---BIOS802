# utils.py
import torch


def exact_solution(k: float, a0: float, t: torch.Tensor) -> torch.Tensor:
    """Defines the concentration for the product in [A] + [B] --> [AB], where [A] = [B] in the aqueous reaction."""
    denom = 1 + k * a0 * t
    u = a0 - (a0 / denom)
    return u
