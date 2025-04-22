# train.py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from typing import Callable
from torch import Tensor
from torch.nn import Module  # Import Module for typing nn.Module
from utils import exact_solution
from pandas import DataFrame  # Import DataFrame for type annotation
from pinn_kinetics.synthetic_data import generate_noisy_lab_data


# Get the synthetic lab data
df: DataFrame = generate_noisy_lab_data()

# Convert to PyTorch tensors
t_data: Tensor = (
    torch.tensor(df["t"].values, dtype=torch.float32).view(-1, 1).requires_grad_(True)
)
u_data: Tensor = torch.tensor(df["u"].values, dtype=torch.float32).view(-1, 1)


def train(
    pinn: Module,  # Change to nn.Module
    t_boundary: Tensor,
    t_physics: Tensor,
    t_test: Tensor,
    A0: float,
    k: float,
    num_epochs: int = 1001,
) -> None:
    # Optimizer
    optimiser = optim.Adam(pinn.parameters(), lr=1e-3)

    # Exact solution for plotting

    # After converting to numpy for plotting
    u_exact: Tensor = exact_solution(k, A0, t_test).detach()
    u_exact2: Tensor = exact_solution(8, A0, t_test).detach()

    # Training loop
    for i in range(num_epochs):
        optimiser.zero_grad()

        # Boundary loss: u(0) = 0
        u_boundary = pinn(t_boundary)
        loss_bc = (u_boundary - 0.0) ** 2

        # Derivative at t = 0: du/dt = k * A0^2
        dudt_boundary = torch.autograd.grad(
            u_boundary, t_boundary, torch.ones_like(u_boundary), create_graph=True
        )[0]
        loss_dbc = (dudt_boundary - k * A0**2) ** 2  # Change this

        # Physics loss: du/dt = k (A0 - u)^2
        u_physics = pinn(t_physics)
        dudt_physics = torch.autograd.grad(
            u_physics, t_physics, torch.ones_like(u_physics), create_graph=True
        )[0]

        physics_residual = dudt_physics - k * (A0 - u_physics) ** 2  # Change this
        loss_phys = torch.mean(physics_residual**2)

        # Data loss:
        u_pred2 = pinn(t_test)
        loss_data = torch.mean((u_pred2 - u_exact) ** 2)

        # Total loss
        loss = 1e-4 * loss_bc + 1e-4 * loss_dbc + (1e-3 * loss_phys) + 1e-3 * loss_data

        loss.backward()
        optimiser.step()

        # Plotting
        if i % 100 == 0:
            plot_results(
                t_test,
                u_exact,
                u_exact2,
                pinn,
                t_physics,
                t_boundary,
                t_data,
                u_data,
                i,
            )


def plot_results(
    t_test: Tensor,
    u_exact: Tensor,
    u_exact2: Tensor,
    pinn: Callable[[Tensor], Tensor],
    t_physics: Tensor,
    t_boundary: Tensor,
    t_data: Tensor,
    u_data: Tensor,
    step: int,
) -> None:
    with torch.no_grad():
        u_pred = pinn(t_test)
    plt.figure(figsize=(6, 3))
    plt.plot(
        t_test.cpu().numpy(),
        u_exact.cpu().numpy(),
        label="Exact",
        color="black",
        linestyle="--",
    )
    plt.plot(
        t_test.cpu().numpy(),
        u_exact2.cpu().numpy(),
        label="Exact2",
        color="black",
        linestyle="--",
    )
    plt.plot(t_test.cpu().numpy(), u_pred.cpu().numpy(), label="PINN", color="green")

    plt.scatter(
        t_physics.detach().cpu().numpy(),
        np.zeros_like(t_physics.detach().cpu().numpy()),
        s=15,
        c="blue",
        label="Physics pts",
        alpha=0.3,
    )

    # Plot the noisy lab data as purple dots (fixing the index issue here)
    plt.scatter(
        t_data.detach().cpu().numpy(),
        u_data.detach().cpu().numpy(),
        s=15,
        c="purple",
        label="Lab Data",
        alpha=0.6,
    )

    plt.scatter(
        t_boundary.detach().cpu().numpy(),
        [0],
        s=50,
        c="red",
        label="Boundary",
        alpha=0.6,
    )

    plt.title(f"Training Step: {step}")
    plt.legend()
    plt.grid(True)
    plt.show()
