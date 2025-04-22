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

# Constants for the first and second models
A0, k1, k2 = 2.0, 10, 5

# Neural network models for the two cases
pinn1 = FCN(1, 1, 32, 3)
pinn2 = FCN(1, 1, 32, 3)

# Define training points
t_boundary = torch.tensor([[0.0]], requires_grad=True)
t_physics = torch.linspace(0, 1, 30).view(-1, 1).requires_grad_(True)
t_test = torch.linspace(0, 1, 100).view(-1, 1)

# Exact solutions for both models
u_exact1 = exact_solution(k1, A0, t_test).detach()
u_exact2 = exact_solution(k2, A0, t_test).detach()

# Optimizer for both PINNs
optimiser1 = torch.optim.Adam(pinn1.parameters(), lr=1e-3)
optimiser2 = torch.optim.Adam(pinn2.parameters(), lr=1e-3)

# Training loop
for i in range(1001):
    # Training for the first PINN (u_exact1)
    optimiser1.zero_grad()

    u_boundary1 = pinn1(t_boundary)
    loss_bc1 = (u_boundary1 - 0.0) ** 2

    dudt_boundary1 = torch.autograd.grad(
        u_boundary1, t_boundary, torch.ones_like(u_boundary1), create_graph=True
    )[0]
    loss_dbc1 = (dudt_boundary1 - k1 * A0) ** 2

    u_physics1 = pinn1(t_physics)
    dudt_physics1 = torch.autograd.grad(
        u_physics1, t_physics, torch.ones_like(u_physics1), create_graph=True
    )[0]

    physics_residual1 = dudt_physics1 - k1 * (A0 - u_physics1)
    loss_phys1 = torch.mean(physics_residual1**2)

    loss1 = loss_bc1 + 1e-4 * loss_dbc1 + 1e-1 * loss_phys1
    loss1.backward()
    optimiser1.step()

    # Training for the second PINN (u_exact2)
    optimiser2.zero_grad()

    u_boundary2 = pinn2(t_boundary)
    loss_bc2 = (u_boundary2 - 0.0) ** 2

    dudt_boundary2 = torch.autograd.grad(
        u_boundary2, t_boundary, torch.ones_like(u_boundary2), create_graph=True
    )[0]
    loss_dbc2 = (dudt_boundary2 - k2 * A0) ** 2

    u_physics2 = pinn2(t_physics)
    dudt_physics2 = torch.autograd.grad(
        u_physics2, t_physics, torch.ones_like(u_physics2), create_graph=True
    )[0]

    physics_residual2 = dudt_physics2 - k2 * (A0 - u_physics2)
    loss_phys2 = torch.mean(physics_residual2**2)

    loss2 = loss_bc2 + 1e-4 * loss_dbc2 + 1e-1 * loss_phys2
    loss2.backward()
    optimiser2.step()

    # Plotting every 100 steps
    if i % 100 == 0:
        with torch.no_grad():
            u_pred1 = pinn1(t_test)
            u_pred2 = pinn2(t_test)

        # Plotting the results
        plt.figure(figsize=(6, 3))
        plt.plot(
            t_test.numpy(),
            u_exact1.numpy(),
            label="Exact (k=10)",
            color="black",
            linestyle="--",
        )
        plt.plot(
            t_test.numpy(),
            u_exact2.numpy(),
            label="Exact (k=8)",
            color="red",
            linestyle="--",
        )

        plt.plot(
            t_test.numpy(),
            u_pred1.detach().numpy(),
            label="PINN1 (k=10)",
            color="green",
        )
        plt.plot(
            t_test.numpy(), u_pred2.detach().numpy(), label="PINN2 (k=8)", color="blue"
        )

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
        plt.xlabel("Time(Units)")
        plt.ylabel("Product Concentration [P]")
        plt.legend()
        plt.grid(True)
        plt.show()

# First order reaction [A] --> [B] with different rate constants.
