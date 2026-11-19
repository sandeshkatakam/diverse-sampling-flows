"""Solver-refinement and small-dimensional numerical invariance studies."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from diverse_sampling_flows.solvers.rk4 import integrate_rk4


@dataclass(frozen=True)
class RefinementComparison:
    coarse_steps: int
    fine_steps: int
    mean_normalized_latent_difference: float
    mean_score_difference: float


@dataclass(frozen=True)
class RefinementReport:
    comparisons: list[RefinementComparison]
    converged: bool


def run_refinement_study(
    field: nn.Module,
    seeds: torch.Tensor,
    *,
    checkpoint_times: Sequence[float],
    steps_per_interval: Sequence[int],
    score_function: Callable[[torch.Tensor], torch.Tensor],
    latent_tolerance: float,
    score_tolerance: float,
) -> RefinementReport:
    solutions = [
        integrate_rk4(field, seeds, checkpoint_times, steps)
        for steps in steps_per_interval
    ]
    scores = [score_function(solution) for solution in solutions]
    dimension = math.prod(seeds.shape[1:])
    comparisons = []
    for index in range(len(solutions) - 1):
        normalized = (solutions[index] - solutions[index + 1]).flatten(2).norm(dim=-1)
        latent_difference = float((normalized / math.sqrt(dimension)).mean())
        score_difference = float((scores[index] - scores[index + 1]).abs().mean())
        comparisons.append(
            RefinementComparison(
                int(steps_per_interval[index]),
                int(steps_per_interval[index + 1]),
                latent_difference,
                score_difference,
            )
        )
    finest = comparisons[-1]
    return RefinementReport(
        comparisons,
        finest.mean_normalized_latent_difference < latent_tolerance
        and finest.mean_score_difference < score_tolerance,
    )


def gaussian_map_identity_residual(
    mapping: Callable[[torch.Tensor], torch.Tensor], seeds: torch.Tensor
) -> torch.Tensor:
    residuals = []
    for seed in seeds:
        seed = seed.detach().requires_grad_(True)
        output = mapping(seed)
        jacobian = torch.autograd.functional.jacobian(mapping, seed, create_graph=False)
        _, log_absolute_determinant = torch.linalg.slogdet(jacobian)
        radial_change = 0.5 * (output.square().sum() - seed.square().sum())
        residuals.append(log_absolute_determinant - radial_change)
    return torch.stack(residuals)
