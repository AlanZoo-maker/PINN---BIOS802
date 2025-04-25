import numpy as np
import torch
import pandas as pd
from pandas import DataFrame
from pinn_kinetics.utils import exact_solution


def generate_noisy_lab_data(
    a0: float,  # Initial concentration
    k: float,  # Reaction rate constant
    t_max: float,  # Maximum time (s)
    n_points: int,  # Number of data points
    noise_std: float,  # Noise level (standard deviation)
    seed: int,  # Random seed for reproducibility
    filename: str = "lab_data.csv",
) -> DataFrame:
    # Set the random seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate time values as a PyTorch tensor
    t_vals = torch.linspace(0, t_max, n_points).view(-1, 1)  # 2D Tensor for consistency

    # Compute the clean solution
    u_clean = exact_solution(k, a0, t_vals)

    # Add noise to the clean solution
    noise = torch.normal(mean=0.0, std=noise_std, size=u_clean.shape)
    u_noisy = u_clean + noise

    # Convert tensors back to numpy for saving to CSV
    t_vals_np = t_vals.detach().numpy()  # Detach and convert to NumPy for CSV
    u_noisy_np = u_noisy.detach().numpy()  # Detach and convert to NumPy for CSV

    # Create DataFrame
    df = pd.DataFrame({"t": t_vals_np.flatten(), "u": u_noisy_np.flatten()})

    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"✅ Noisy lab data saved to: {filename}")
    print(df.head())

    return df
