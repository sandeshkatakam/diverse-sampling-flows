import math

import torch
from torch import nn

from diverse_sampling_flows.solvers.rk4 import integrate_rk4


class _RotationField(nn.Module):
    def __init__(self, omega: float) -> None:
        super().__init__()
        self.omega = nn.Parameter(torch.tensor(omega, dtype=torch.float64))

    def forward(self, time, state):
        del time
        x, y = state.unbind(dim=1)
        return torch.stack((-self.omega * y, self.omega * x), dim=1)


def test_rk4_returns_exact_requested_checkpoints_and_is_differentiable() -> None:
    field = _RotationField(0.7)
    initial = torch.tensor([[1.0, 0.0]], dtype=torch.float64)

    checkpoints = integrate_rk4(field, initial, [0.0, 0.5, 1.0], 8)
    checkpoints.square().sum().backward()

    assert checkpoints.shape == (1, 3, 2)
    assert torch.equal(checkpoints[:, 0], initial)
    assert field.omega.grad is not None and torch.isfinite(field.omega.grad)


def test_rk4_rotation_error_decreases_at_8_16_32_steps() -> None:
    field = _RotationField(1.3)
    initial = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    exact = torch.tensor([math.cos(1.3), math.sin(1.3)], dtype=torch.float64)
    errors = []
    for steps in (8, 16, 32):
        result = integrate_rk4(field, initial, [0.0, 1.0], steps)[0, -1]
        errors.append(torch.linalg.vector_norm(result - exact).item())

    assert errors[0] > errors[1] > errors[2]


def test_rk4_parameter_gradient_matches_central_difference() -> None:
    field = _RotationField(0.4)
    initial = torch.tensor([[0.8, -0.2]], dtype=torch.float64)

    result = integrate_rk4(field, initial, [0.0, 1.0], 16)[0, -1, 0]
    analytic = torch.autograd.grad(result, field.omega)[0]
    epsilon = 1e-6
    with torch.no_grad():
        field.omega.add_(epsilon)
        plus = integrate_rk4(field, initial, [0.0, 1.0], 16)[0, -1, 0]
        field.omega.sub_(2 * epsilon)
        minus = integrate_rk4(field, initial, [0.0, 1.0], 16)[0, -1, 0]
        field.omega.add_(epsilon)

    assert torch.allclose(analytic, (plus - minus) / (2 * epsilon), atol=1e-8)

