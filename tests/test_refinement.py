import math

import torch
from torch import nn

from diverse_sampling_flows.evaluation.refinement import (
    gaussian_map_identity_residual,
    run_refinement_study,
)


class _Rotation(nn.Module):
    def forward(self, time, state):
        del time
        x, y = state.unbind(-1)
        return torch.stack((-y, x), dim=-1)


def test_refinement_reports_decreasing_normalized_latent_differences() -> None:
    seeds = torch.tensor([[1.0, 0.0], [0.5, -0.3]], dtype=torch.float64)
    report = run_refinement_study(
        _Rotation(),
        seeds,
        checkpoint_times=[0.0, 1.0],
        steps_per_interval=[8, 16, 32],
        score_function=lambda slots: slots[:, -1, 0],
        latent_tolerance=1e-3,
        score_tolerance=0.02,
    )

    assert (
        report.comparisons[0].mean_normalized_latent_difference
        > report.comparisons[1].mean_normalized_latent_difference
    )
    assert report.comparisons[1].mean_normalized_latent_difference < 1e-3
    assert report.converged


def test_gaussian_change_of_variables_residual_detects_rotation_and_drift() -> None:
    angle = 0.7
    rotation = torch.tensor(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=torch.float64,
    )
    seeds = torch.randn(
        5, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(4)
    )

    preserving = gaussian_map_identity_residual(lambda value: rotation @ value, seeds)
    drifting = gaussian_map_identity_residual(lambda value: 1.1 * value, seeds)

    assert preserving.abs().max().item() < 1e-12
    assert drifting.abs().mean().item() > 1e-2
