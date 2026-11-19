"""IID, balanced-Gaussian, and independent-seed negative controls."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from diverse_sampling_flows.solvers.rk4 import integrate_rk4


def sample_iid(
    batch: int,
    slots: int,
    shape: Sequence[int],
    *,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.randn(
        (batch, slots, *shape), generator=generator, device=device, dtype=dtype
    )


def sample_balanced_gaussian(
    batch: int,
    slots: int,
    shape: Sequence[int],
    *,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if slots < 2:
        raise ValueError("balanced Gaussian requires at least two slots")
    independent = sample_iid(
        batch, slots, shape, generator=generator, device=device, dtype=dtype
    )
    return math.sqrt(slots / (slots - 1)) * (
        independent - independent.mean(dim=1, keepdim=True)
    )


def independent_seed_trajectory(
    field: nn.Module,
    seeds: torch.Tensor,
    *,
    final_time: float,
    steps: int,
) -> torch.Tensor:
    batch, slots = seeds.shape[:2]
    flattened = seeds.flatten(0, 1)
    final = integrate_rk4(
        field, flattened, [0.0, float(final_time)], steps
    )[:, -1]
    return final.reshape(batch, slots, *seeds.shape[2:])

