# run_training.py
import torch
from model import FCN
from train import train


def main() -> None:
    # Constants
    a0, k = 3.0, 15

    # Define training points
    t_boundary = torch.tensor([[0.0]], requires_grad=True)
    t_physics = torch.linspace(0, 1, 30).view(-1, 1).requires_grad_(True)
    t_test = torch.linspace(0, 1, 100).view(-1, 1)

    # Create the model
    pinn = FCN(1, 1, 32, 3)

    # Train the model
    df_losses = train(pinn, t_boundary, t_physics, t_test, a0, k)

    print(df_losses)


if __name__ == "__main__":
    main()
