import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# Reuse your original FCN definition
class FCN(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(nn.Linear(N_INPUT, N_HIDDEN), activation())
        self.fch = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(N_HIDDEN, N_HIDDEN), activation())
                for _ in range(N_LAYERS - 1)
            ]
        )
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


# Set random seed
torch.manual_seed(0)

# Reaction parameters
A0, B0 = 1.0, 1.0
k = 2.0  # rate constant

# Neural network: 1 input (t), 3 outputs (A, B, AB)
pinn = FCN(N_INPUT=1, N_OUTPUT=3, N_HIDDEN=32, N_LAYERS=3)

# Optimizer
optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)

# Time grid
t_physics = torch.linspace(0, 1, 100).view(-1, 1).requires_grad_()
t0 = torch.tensor([[0.0]], requires_grad=True)

# Training loop
for i in range(1001):
    optimizer.zero_grad()

    # Evaluate the network at training points
    out = pinn(t_physics)
    A, B, AB = out[:, 0:1], out[:, 1:2], out[:, 2:3]

    # Derivatives
    dA_dt = torch.autograd.grad(A, t_physics, torch.ones_like(A), create_graph=True)[0]
    dB_dt = torch.autograd.grad(B, t_physics, torch.ones_like(B), create_graph=True)[0]
    dAB_dt = torch.autograd.grad(AB, t_physics, torch.ones_like(AB), create_graph=True)[
        0
    ]

    # Reaction rate and physics residuals
    rate = -k * A * B
    res_A = dA_dt - rate
    res_B = dB_dt - rate
    res_AB = dAB_dt + rate  # dw/dt = -rate (rate is negative here)

    # Physics loss
    loss_phys = torch.mean(res_A**2 + res_B**2 + res_AB**2)

    # Initial condition loss
    out0 = pinn(t0)
    A0_pred, B0_pred, AB0_pred = out0[0, 0], out0[0, 1], out0[0, 2]
    loss_ic = (A0_pred - A0) ** 2 + (B0_pred - B0) ** 2 + (AB0_pred - 0.0) ** 2

    # Total loss
    loss = loss_ic + 1e-1 * loss_phys
    loss.backward()
    optimizer.step()

    # Print progress
    if i % 200 == 0:
        print(f"Step {i}: Loss = {loss.item():.6f}")

        # Plot
        t_test = torch.linspace(0, 1, 200).view(-1, 1)
        with torch.no_grad():
            pred = pinn(t_test)
        A_pred, B_pred, AB_pred = pred[:, 0], pred[:, 1], pred[:, 2]

        # Optional exact solution for equal A0 = B0
        A_exact = A0 / (1 + A0 * k * t_test)
        B_exact = A_exact
        AB_exact = A0 - A_exact

        plt.figure(figsize=(7, 4))
        plt.plot(t_test.numpy(), A_pred.numpy(), label="[A] (PINN)", color="tab:blue")
        plt.plot(t_test.numpy(), B_pred.numpy(), label="[B] (PINN)", color="tab:orange")
        plt.plot(
            t_test.numpy(), AB_pred.numpy(), label="[AB] (PINN)", color="tab:green"
        )
        plt.plot(
            t_test.numpy(),
            A_exact.numpy(),
            "--",
            color="tab:blue",
            alpha=0.5,
            label="[A] exact",
        )
        plt.plot(
            t_test.numpy(),
            AB_exact.numpy(),
            "--",
            color="tab:green",
            alpha=0.5,
            label="[AB] exact",
        )
        plt.title(f"Bimolecular Reaction at Step {i}")
        plt.xlabel("Time")
        plt.ylabel("Concentration")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
