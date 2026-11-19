import pytest
import torch

from diverse_sampling_flows.conditioners.factory import build_conditioner
from diverse_sampling_flows.priors.masks import build_channel_mask_groups


@pytest.mark.parametrize("architecture", ["mlp", "resnet", "transformer"])
def test_conditioner_factory_preserves_spatial_coefficient_shape(
    architecture: str,
) -> None:
    conditioner = build_conditioner(
        architecture,
        context_channels=4,
        pairs=2,
        ridges=1,
        width=16,
        residual_blocks=1,
        transformer_blocks=1,
        transformer_heads=4,
        patch_size=2,
    ).double()
    context = torch.randn(2, 4, 8, 8, dtype=torch.float64)
    time = torch.tensor([0.25, 0.75], dtype=torch.float64)

    output = conditioner.coefficients(time, context, group_id=1)

    assert output.shape == (2, 10, 8, 8)
    assert torch.isfinite(output).all()


def _assert_active_derivative_is_zero(coefficient_function, state, active) -> None:
    output = coefficient_function(state)
    gradient = torch.autograd.grad(output.sum(), state, create_graph=True)[0]
    assert torch.count_nonzero(gradient[:, active]).item() == 0


def test_masking_before_conditioner_excludes_active_inputs_and_detects_leak() -> None:
    group = build_channel_mask_groups(8)[0]
    state = torch.randn(1, 8, 4, 4, dtype=torch.float64, requires_grad=True)
    conditioner = build_conditioner(
        "mlp", context_channels=4, pairs=2, ridges=1, width=8
    ).double()
    time = torch.tensor([0.5], dtype=torch.float64)
    active = [index for pair in group.active_pairs for index in pair]

    _assert_active_derivative_is_zero(
        lambda full: conditioner.coefficients(
            time, full[:, group.context_channels], group.group_id
        ),
        state,
        active,
    )

    with pytest.raises(AssertionError):
        _assert_active_derivative_is_zero(
            lambda full: full[:, active].square(), state, active
        )

