# utils.py
import torch


def exact_solution(k: float, A0: float, t: torch.Tensor) -> torch.Tensor:
    "Defines the analytical solution for the rate of reaction in a first-order reaction."
    denom = 1 + k * A0 * t
    u = A0 - (A0 / denom)
    return u
