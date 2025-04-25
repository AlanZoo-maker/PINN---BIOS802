# train.py
from typing import Callable
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from pandas import DataFrame  # Import DataFrame for type annotation
from torch import Tensor
from torch.nn import Module  # Import Module for typing nn.Module
from pinn_kinetics.synthetic_data import generate_noisy_lab_data
from pinn_kinetics.utils import exact_solution


# Get the synthetic lab data
df: DataFrame = generate_noisy_lab_data(
    a0=3.0,
    k=12.0,
    t_max=0.5,
    n_points=30,
    noise_std=0.5,  # ← Try 0.01 (low noise) to 0.2 (high noise)
    seed=1,
    filename="lab_data.csv",
)

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
    a0: float,
    k: float,
    num_epochs: int = 2001,
    plot=True,
) -> pd.DataFrame:

    # loss logs
    loss_bc_log = []
    loss_dbc_log = []
    loss_phys_log = []
    loss_data_log = []

    # Optimizer
    optimiser = optim.Adam(pinn.parameters(), lr=1e-3)

    # Exact solutions; dotted lines
    u_exact: Tensor = exact_solution(k, a0, t_test).detach()

    # Training loop: loss terms are backpropagated
    for i in range(num_epochs):
        optimiser.zero_grad()

        # Boundary loss: u(0) = 0
        u_boundary = pinn(t_boundary)
        loss_bc = (u_boundary - 0.0) ** 2

        # Derivative at t = 0: du/dt = k * a0^2
        dudt_boundary = torch.autograd.grad(
            u_boundary, t_boundary, torch.ones_like(t_boundary), create_graph=True
        )[0]
        loss_dbc = (dudt_boundary - k * a0**2) ** 2

        # Physics loss: du/dt = k (a0- u)^2
        u_physics = pinn(t_physics)
        dudt_physics = torch.autograd.grad(
            u_physics, t_physics, torch.ones_like(u_physics), create_graph=True
        )[0]

        physics_residual = dudt_physics - k * (a0 - u_physics) ** 2
        loss_phys = torch.mean(physics_residual**2)

        # Data loss:
        u_pred2 = pinn(t_test)
        loss_data = torch.mean((u_pred2 - u_exact) ** 2)

        # Total loss
        loss = 1e-4 * loss_bc + 1e-4 * loss_dbc + (1e-3 * loss_phys) + 1e-3 * loss_data

        loss.backward()
        optimiser.step()

        if i % 500 == 0:
            print(
                f"Epoch {i}: BC Loss={loss_bc.item()}, DBC Loss={loss_dbc.item()}, Physics Loss={loss_phys.item()}, Data Loss={loss_data.item()}"
            )
            loss_bc_log.append(loss_bc.item())
            loss_dbc_log.append(loss_dbc.item())
            loss_phys_log.append(loss_phys.item())
            loss_data_log.append(loss_data.item())

            if plot:
                plot_results(
                    t_test,
                    u_exact,
                    pinn,
                    t_physics,
                    t_boundary,
                    i,
                )

    # Create and return the DataFrame
    df_losses = pd.DataFrame(
        {
            "Epoch": list(range(0, num_epochs, 500)),
            "Loss_BC": loss_bc_log,
            "Loss_DBC": loss_dbc_log,
            "Loss_Physics": loss_phys_log,
            "Loss_Data": loss_data_log,
        }
    )

    return df_losses


def plot_results(
    t_test: Tensor,
    u_exact: Tensor,
    pinn: Callable[[Tensor], Tensor],
    t_physics: Tensor,
    t_boundary: Tensor,
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
