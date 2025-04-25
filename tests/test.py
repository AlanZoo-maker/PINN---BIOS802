import matplotlib

matplotlib.use("Agg")

import torch
import random
import numpy as np
from torch import nn

from pinn_kinetics.model import FCN
from pinn_kinetics.train import train


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_model_learns_all_losses():
    # Setup
    a0, k = 3.0, 15
    t_boundary = torch.tensor([[0.0]], requires_grad=True)
    t_physics = torch.linspace(0, 1, 30).view(-1, 1).requires_grad_(True)
    t_test = torch.linspace(0, 1, 100).view(-1, 1)

    pinn = FCN(1, 1, 32, 3)

    # Train
    df_losses = train(
        pinn, t_boundary, t_physics, t_test, a0, k, num_epochs=2001, plot=False
    )

    # Assert all losses decreased over training
    assert (
        df_losses["Loss_BC"].iloc[-1] < df_losses["Loss_BC"].iloc[0]
    ), "Boundary loss did not decrease"
    assert (
        df_losses["Loss_DBC"].iloc[-1] < df_losses["Loss_DBC"].iloc[0]
    ), "Derivative boundary loss did not decrease"
    assert (
        df_losses["Loss_Physics"].iloc[-1] < df_losses["Loss_Physics"].iloc[0]
    ), "Physics loss did not decrease"
    assert (
        df_losses["Loss_Data"].iloc[-1] < df_losses["Loss_Data"].iloc[0]
    ), "Data loss did not decrease"


def test_forward_deterministic():
    set_seed(42)
    model = FCN(n_input=3, n_output=1, n_hidden=4, n_layers=2)

    # Set all weights and biases to constants
    for layer in model.net:
        if isinstance(layer, nn.Linear):
            nn.init.constant_(layer.weight, 0.5)
            nn.init.constant_(layer.bias, 0.1)

    input_tensor = torch.tensor(
        [[1.0, 2.0, 4.0]]
    )  # number of elements here should match n_input
    output = model(input_tensor)

    # Check output shape
    expected_shape = (1, 1)  # scalar produced here
    assert (
        output.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {output.shape}"
    assert torch.isfinite(output).all(), "Output contains NaN or Inf"


def test_zero_input():
    set_seed(42)  # Ensures deterministic weight

    model = FCN(2, 1, 4, 2)
    with torch.no_grad():
        output = model(torch.zeros(1, 2))  # batch size 1, 2 inputs/columns

    assert torch.isfinite(output).all(), "Output is not finite"
    assert output.shape == (1, 1)
    print("Output:", output.item())  # Will now always print the same value


def test_model_structure():
    model = FCN(n_input=3, n_output=1, n_hidden=5, n_layers=3)
    layers = list(model.net)

    # Expected 4 Linear layers: 1 input + (3-1) hidden + 1 output
    expected_num_linear = 4
    actual_num_linear = sum(1 for layer in layers if isinstance(layer, nn.Linear))

    # 1st layer: input layer (nn.linear(n_input = 3, n_hidden = 5))
    # 2nd layer: 1st hidden layer with 5 nodes (nn.linear(n_input = 3, n_hidden = 5))
    # 3rd layer: 2nd hidden layer (nn.linear(n_hidden = 5, n_hidden = 5))
    # 4th layer: output layer (nn.linear( n_hidden = 5, n_output = 1))

    assert (
        actual_num_linear == expected_num_linear
    ), f"Expected {expected_num_linear} Linear layers, got {actual_num_linear}"


def test_deterministic_output():
    set_seed(42)
    model1 = FCN(2, 1, 4, 2)
    out1 = model1(torch.ones(1, 2))

    set_seed(42)
    model2 = FCN(2, 1, 4, 2)
    out2 = model2(torch.ones(1, 2))

    # Check if the outputs from two identical models (with the same seed) are equal
    assert torch.allclose(out1, out2)
