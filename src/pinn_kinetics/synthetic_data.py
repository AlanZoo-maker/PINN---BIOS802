import numpy as np
import torch
import pandas as pd
from pandas import DataFrame
from utils import exact_solution


def generate_noisy_lab_data(
    A0: float = 2.0,  # Initial concentration
    k: float = 10.0,  # Reaction rate constant
    t_max: float = 0.5,  # Maximum time (s)
    n_points: int = 30,  # Number of data points
    noise_std: float = 0.05,  # Noise level (standard deviation)
    seed: int = 42,  # Random seed for reproducibility
    filename: str = "lab_data.csv",
) -> DataFrame:
    # Set the random seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate time values as a PyTorch tensor
    t_vals = torch.linspace(0, t_max, n_points).view(-1, 1)  # 2D Tensor for consistency

    # Compute the clean solution
    u_clean = exact_solution(k, A0, t_vals)

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


# Example usage:
generate_noisy_lab_data(
    A0=2.0,
    k=10.0,
    t_max=0.5,
    n_points=30,
    noise_std=0.1,  # ← Try 0.01 (low noise) to 0.2 (high noise)
    filename="lab_data.csv",
)
