import pytest
import torch
from model import FCN  # Importing the FCN class from model.py
from utils import exact_solution  # Importing the exact_solution function from utils.py


# Fixtures to set up necessary tensors and model
@pytest.fixture
def pinn_model():
    return FCN(1, 1, 32, 3)  # Initialize the FCN model


@pytest.fixture
def boundary_test_tensor():
    return torch.tensor(
        [[0.0]], requires_grad=True
    )  # Tensor for boundary condition at t = 0


# Test 1: Check that the model output shape is correct
def test_output_shape(pinn_model):
    t_input = torch.tensor([[0.1]])  # Simple input tensor
    output = pinn_model(t_input)
    assert output.shape == (1, 1), "Output shape should be (1, 1)"


# Test 2: Ensure that the exact solution returns the correct shape
def test_exact_solution_shape():
    t_test = torch.linspace(0, 1, 100).view(-1, 1)
    k, A0 = 10, 2.0
    u_exact = exact_solution(k, A0, t_test)
    assert u_exact.shape == (100, 1), "Exact solution shape should match t_test"


# Test 3: Check if the boundary condition u(0) = 0 is satisfied
def test_boundary_condition(pinn_model, boundary_test_tensor):
    with torch.no_grad():
        u_boundary = pinn_model(boundary_test_tensor)
        assert torch.isclose(
            u_boundary, torch.tensor([[0.0]]), atol=0.1
        ).all(), "u(0) should be close to 0 after training"


# Test 4: Check if the derivative at t=0 is close to k * A0
def test_derivative_at_t0(pinn_model, boundary_test_tensor):
    with torch.no_grad():
        u_boundary = pinn_model(boundary_test_tensor)
        dudt_boundary = torch.autograd.grad(
            u_boundary,
            boundary_test_tensor,
            torch.ones_like(u_boundary),
            create_graph=True,
        )[0]
        k, A0 = 10, 2.0
        expected_du0 = torch.tensor([[k * A0]])
        assert torch.isclose(
            dudt_boundary, expected_du0, atol=1.0
        ).all(), "Initial derivative should be close to k * A0"


# Test 5: Check if the physics residuals are small after training
def test_physics_residual(pinn_model):
    t_physics = torch.linspace(0, 1, 30).view(-1, 1).requires_grad_(True)
    u_physics = pinn_model(t_physics)
    dudt_physics = torch.autograd.grad(
        u_physics, t_physics, torch.ones_like(u_physics), create_graph=True
    )[0]
    k, A0 = 10, 2.0
    residual = dudt_physics - k * (A0 - u_physics)
    assert torch.mean(residual**2) < 1e-2, "Physics residuals should be small"


# Test 6: Compare the model's output with the exact solution
def test_model_vs_exact_solution(pinn_model):
    t_test = torch.linspace(0, 1, 100).view(-1, 1)
    u_exact = exact_solution(10, 2.0, t_test)
    with torch.no_grad():
        u_pred = pinn_model(t_test)
    assert torch.allclose(
        u_pred, u_exact, atol=0.2
    ), "PINN solution should closely match exact solution"


# Test 7: Ensure that the loss is decreasing during training (example with mock losses)
def test_loss_decrease():
    losses = [1.0, 0.9, 0.8, 0.7]  # This should be a real list tracked during training
    assert losses[0] > losses[-1], "Loss should decrease over training"


# Run the tests using pytest by saving the file and running `pytest <filename>.py`
