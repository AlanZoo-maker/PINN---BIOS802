# model.py
import torch.nn as nn
from torch import Tensor
from typing import cast

# This model, along with parts of the plotting code with the train loop can be found under Benn Moseley's GitHub page
# for tutorials in Scientific Machine Learning.


class FCN(nn.Module):
    def __init__(
        self, n_input: int, n_output: int, n_hidden: int, n_layers: int
    ) -> None:
        super().__init__()
        activation = nn.Tanh
        layers = [nn.Linear(n_input, n_hidden), activation()]  # 1st layer
        for _ in range(n_layers - 1):  # add n-1 layers
            layers += [nn.Linear(n_hidden, n_hidden), activation()]
        layers += [nn.Linear(n_hidden, n_output)]  # add output layer
        self.net = nn.Sequential(*layers)  # organize the layers

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.net(x))
