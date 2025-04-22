# model.py
import torch.nn as nn
from torch import Tensor
from typing import cast


class FCN(nn.Module):
    def __init__(
        self, N_INPUT: int, N_OUTPUT: int, N_HIDDEN: int, N_LAYERS: int
    ) -> None:
        super().__init__()
        activation = nn.Tanh
        layers = [nn.Linear(N_INPUT, N_HIDDEN), activation()]
        for _ in range(N_LAYERS - 1):
            layers += [nn.Linear(N_HIDDEN, N_HIDDEN), activation()]
        layers += [nn.Linear(N_HIDDEN, N_OUTPUT)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.net(x))
