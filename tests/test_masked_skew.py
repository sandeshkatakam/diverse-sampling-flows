import math

import torch
from torch import nn

from diverse_sampling_flows.priors.masked_skew import (
    CoefficientBounds,
    MaskedSkewField,
    bounded_coefficients,
)
from diverse_sampling_flows.priors.masks import MaskGroup


def _inverse_tanh(value: float) -> float:
    return math.atanh(value)


class _ConstantConditioner(nn.Module):
    def __init__(self, raw: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("raw", raw)

    def coefficients(
        self, time, context_tensor, group_id, optional_prompt_context=None
    ):
        del time, group_id, optional_prompt_context
        return self.raw.expand(context_tensor.shape[0], -1, *context_tensor.shape[-2:])


def _one_pair_field(c, alpha, beta, delta, e) -> MaskedSkewField:
    bounds = CoefficientBounds(c_max=math.pi / 4, a_max=0.25, b_max=1.0, e_max=2.0)
    raw = torch.tensor(
        [
            0.0
            if c == math.pi / 2
            else _inverse_tanh((c - math.pi / 2) / bounds.c_max),
            _inverse_tanh(alpha / bounds.a_max) if alpha else 0.0,
            _inverse_tanh(beta / bounds.b_max) if beta else 0.0,
            _inverse_tanh(delta / bounds.b_max) if delta else 0.0,
            _inverse_tanh(e / bounds.e_max) if e else 0.0,
        ],
        dtype=torch.float64,
    ).view(1, 5, 1, 1)
    group = MaskGroup(group_id=0, context_channels=(2, 3), active_pairs=((0, 1),))
    return MaskedSkewField(
        _ConstantConditioner(raw),
        channels=4,
        groups=(group,),
        ridges=1,
        num_slots=4,
        bounds=bounds,
    ).double()


def test_bounded_heads_have_expected_initial_values_and_nonzero_ridge() -> None:
    raw = torch.tensor([0.0, _inverse_tanh(0.08), _inverse_tanh(0.5), 0.0, 0.0])
    c, alpha, beta, delta, e = bounded_coefficients(
        raw.view(1, 1, 5, 1, 1), num_slots=4, bounds=CoefficientBounds()
    )

    assert c.item() == torch.tensor(math.pi / 2).item()
    assert alpha.item() == torch.tensor(0.02).item()
    assert beta.item() == torch.tensor(0.5).item()
    assert delta.item() == 0.0
    assert e.item() == 0.0


def test_constant_head_reproduces_quarter_turn_velocity() -> None:
    field = _one_pair_field(math.pi / 2, 0.0, 0.0, 0.0, 0.0)
    state = torch.tensor([2.0, -3.0, 5.0, 7.0], dtype=torch.float64).reshape(
        1, 4, 1, 1
    )

    velocity = field(torch.tensor(0.0), state)

    assert torch.allclose(velocity[:, 0], -math.pi / 2 * state[:, 1])
    assert torch.allclose(velocity[:, 1], math.pi / 2 * state[:, 0])
    assert torch.count_nonzero(velocity[:, 2:]).item() == 0


def test_analytic_velocity_matches_dense_div_k_minus_kz() -> None:
    field = _one_pair_field(math.pi / 2, 0.1, 0.4, -0.3, 0.2)
    state = torch.randn(1, 4, 1, 1, dtype=torch.float64, requires_grad=True)
    actual = field(torch.tensor(0.3, dtype=torch.float64), state).reshape(4)
    u, v = state.reshape(4)[:2]
    h = math.pi / 2 + 0.1 * torch.tanh(0.4 * u - 0.3 * v + 0.2)
    matrix = torch.zeros(4, 4, dtype=torch.float64)
    matrix[0, 1] = h
    matrix[1, 0] = -h
    divergence_terms = []
    for i in range(4):
        row_divergence = 0.0
        for j in range(4):
            gradient = torch.autograd.grad(
                matrix[i, j], state, retain_graph=True, create_graph=True
            )[0].reshape(4)
            row_divergence = row_divergence + gradient[j]
        divergence_terms.append(row_divergence)
    divergence = torch.stack(divergence_terms)
    expected = divergence - matrix @ state.reshape(4)

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_weighted_divergence_is_floating_point_zero() -> None:
    field = _one_pair_field(math.pi / 2, 0.1, 0.4, -0.3, 0.2)
    state = torch.randn(1, 4, 1, 1, dtype=torch.float64, requires_grad=True)
    velocity = field(torch.tensor(0.4, dtype=torch.float64), state).reshape(4)
    divergence = sum(
        torch.autograd.grad(
            velocity[i], state, retain_graph=True, create_graph=True
        )[0].reshape(4)[i]
        for i in range(4)
    )
    residual = divergence - state.reshape(4).dot(velocity)

    assert residual.abs().item() < 1e-12


def test_tanh_pair_can_have_nonzero_radial_motion() -> None:
    field = _one_pair_field(math.pi / 2, 0.1, 0.4, -0.3, 0.2)
    state = torch.tensor([1.2, -0.7, 0.0, 0.0], dtype=torch.float64).view(1, 4, 1, 1)
    velocity = field(torch.tensor(0.2, dtype=torch.float64), state)

    assert state.flatten().dot(velocity.flatten()).abs().item() > 1e-5
