import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def exact_solution(k, A0, t):
    "Defines the analytical solution for the product concentration in a first-order reaction."
    exp = torch.exp(-k * t)
    u = A0 * (1 - exp)
    return u


class FCN(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        layers = [nn.Linear(N_INPUT, N_HIDDEN), activation()]
        for _ in range(N_LAYERS - 1):
            layers += [nn.Linear(N_HIDDEN, N_HIDDEN), activation()]
        layers += [nn.Linear(N_HIDDEN, N_OUTPUT)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


torch.manual_seed(123)

# Neural network model
pinn = FCN(1, 1, 32, 3)

# Constants
A0, k = 2.0, 10

# Define training points
t_boundary = torch.tensor([[0.0]], requires_grad=True)
t_physics = torch.linspace(0, 1, 30).view(-1, 1).requires_grad_(True)
t_test = torch.linspace(0, 1, 100).view(-1, 1)

# Exact solution
u_exact = exact_solution(k, A0, t_test).detach()

u_exact2 = exact_solution(8, A0, t_test).detach()

# Optimizer
optimiser = torch.optim.Adam(pinn.parameters(), lr=1e-3)

# Training loop
for i in range(1001):
    optimiser.zero_grad()

    # Boundary loss: u(0) = 0
    u_boundary = pinn(t_boundary)
    loss_bc = (u_boundary - 0.0) ** 2

    # Derivative at t = 0: du/dt = k * A0
    dudt_boundary = torch.autograd.grad(
        u_boundary, t_boundary, torch.ones_like(u_boundary), create_graph=True
    )[0]
    loss_dbc = (dudt_boundary - k * A0) ** 2

    # Physics loss: du/dt = k (A0 - u)
    u_physics = pinn(t_physics)
    dudt_physics = torch.autograd.grad(
        u_physics, t_physics, torch.ones_like(u_physics), create_graph=True
    )[0]

    physics_residual = dudt_physics - k * (
        A0 - u_physics
    )  # CHATGPT Consulted this part of the code.
    loss_phys = torch.mean(physics_residual**2)

    # Total loss
    loss = loss_bc + 1e-4 * loss_dbc + 1e-1 * loss_phys
    loss.backward()
    optimiser.step()

    # Plotting
    if i % 100 == 0:
        with torch.no_grad():
            u_pred = pinn(t_test)
        plt.figure(figsize=(6, 3))
        plt.plot(
            t_test.numpy(),
            u_exact.numpy(),
            label="Exact",
            color="black",
            linestyle="--",
        )

        # add a line here for u_exact2.numpy()
        plt.plot(
            t_test.numpy(),
            u_exact2.numpy(),
            label="Exact2",
            color="black",
            linestyle="--",
        )

        plt.plot(t_test.numpy(), u_pred.numpy(), label="PINN", color="green")

        plt.scatter(
            t_physics.detach().numpy(),
            np.zeros_like(t_physics.detach().numpy()),
            s=15,
            c="blue",
            label="Physics pts",
            alpha=0.3,
        )
        plt.scatter(
            t_boundary.detach().numpy(), [0], s=50, c="red", label="Boundary", alpha=0.6
        )
        plt.title(f"Training Step: {i}")
        plt.legend()
        plt.grid(True)
        plt.show()
