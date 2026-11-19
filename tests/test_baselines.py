import pytest
import torch

from diverse_sampling_flows.baselines.conditional_rotation import (
    ConditionalRotationCoupling,
)
from diverse_sampling_flows.baselines.fixed_rotation import fixed_rotation_slots
from diverse_sampling_flows.baselines.learned_orthogonal import (
    LearnedOrthogonalCoupling,
)
from diverse_sampling_flows.baselines.sampling import (
    independent_seed_trajectory,
    sample_balanced_gaussian,
    sample_iid,
)


def test_iid_and_balanced_gaussian_match_analytic_covariance() -> None:
    generator = torch.Generator().manual_seed(123)
    iid = sample_iid(20_000, 4, (2,), generator=generator, dtype=torch.float64)
    balanced = sample_balanced_gaussian(
        20_000, 4, (2,), generator=generator, dtype=torch.float64
    )

    assert iid.shape == balanced.shape == (20_000, 4, 2)
    assert abs(iid[:, 0, 0].var().item() - 1) < 0.03
    assert abs(torch.cov(iid[:, :, 0].T)[0, 1].item()) < 0.03
    covariance = torch.cov(balanced[:, :, 0].T)
    assert abs(covariance.diag().mean().item() - 1) < 0.03
    assert abs(covariance[0, 1].item() + 1 / 3) < 0.03
    assert balanced.sum(dim=1).abs().max().item() < 1e-12


def test_fixed_quarter_turns_are_unique_and_norm_preserving() -> None:
    seed = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    slots = fixed_rotation_slots(seed, num_slots=4)

    assert slots.shape == (1, 4, 4)
    assert torch.equal(slots[:, 0], seed)
    assert torch.allclose(slots.norm(dim=-1), seed.norm(dim=-1, keepdim=True))
    assert torch.unique(slots[0], dim=0).shape[0] == 4


def test_learned_orthogonal_transitions_remain_volume_and_norm_preserving() -> None:
    coupling = LearnedOrthogonalCoupling((4,), num_slots=4, layers=2, seed=7).double()
    with torch.no_grad():
        coupling.angles.uniform_(-0.3, 0.3)
    seed = torch.randn(3, 4, dtype=torch.float64)
    slots = coupling(seed)
    jacobian = torch.autograd.functional.jacobian(
        lambda value: coupling.transition(value[None], 0)[0], seed[0]
    )

    assert torch.equal(slots[:, 0], seed)
    assert torch.allclose(
        slots.norm(dim=-1), seed.norm(dim=-1, keepdim=True), atol=1e-12
    )
    assert torch.allclose(
        jacobian.T @ jacobian, torch.eye(4, dtype=torch.float64), atol=1e-12
    )
    assert torch.det(jacobian).item() == pytest.approx(1.0, abs=1e-12)


def test_residual_rotation_layers_are_applied_sequentially() -> None:
    coupling = LearnedOrthogonalCoupling((4,), num_slots=2, layers=2, seed=4).double()
    with torch.no_grad():
        coupling.angles[0, 0].copy_(torch.tensor([0.2, -0.1]))
        coupling.angles[0, 1].copy_(torch.tensor([0.3, 0.4]))
    value = torch.tensor([[0.5, -1.0, 2.0, 0.3]], dtype=torch.float64)

    sequential = coupling.transition(value, 0)
    reversed_layers = coupling.fixed_quarter_turn(value)
    for layer in (1, 0):
        reversed_layers = coupling.residual_layer(reversed_layers, 0, layer)

    assert not torch.allclose(sequential, reversed_layers)


def test_conditional_rotations_preserve_norm_and_gaussian_covariance() -> None:
    coupling = ConditionalRotationCoupling(
        channels=4, num_slots=2, layers_per_transition=4, width=8
    ).double()
    generator = torch.Generator().manual_seed(81)
    seed = torch.randn(6000, 4, 1, 1, generator=generator, dtype=torch.float64)
    transformed = coupling.transition(seed, 0)
    covariance = torch.cov(transformed.reshape(6000, 4).T)

    assert torch.allclose(
        transformed.flatten(1).norm(dim=1), seed.flatten(1).norm(dim=1), atol=1e-11
    )
    assert torch.max(torch.abs(covariance - torch.eye(4, dtype=torch.float64))) < 0.06


def test_independent_seed_negative_control_keeps_row_and_slot_axes() -> None:
    class UnitField(torch.nn.Module):
        def forward(self, time, state):
            del time
            return torch.ones_like(state)

    seeds = torch.zeros(2, 4, 2, dtype=torch.float64)
    result = independent_seed_trajectory(UnitField(), seeds, final_time=1.0, steps=4)

    assert result.shape == seeds.shape
    assert torch.allclose(result, torch.ones_like(result))
