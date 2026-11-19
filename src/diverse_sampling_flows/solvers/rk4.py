"""Pure differentiable RK4 with checkpoint-aligned fixed steps."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import torch
from torch import nn


def integrate_rk4(
    field: nn.Module,
    initial: torch.Tensor,
    checkpoint_times: Sequence[float],
    steps_per_interval: int,
) -> torch.Tensor:
    """Integrate in physical time and return ``[B,K,...]`` checkpoints."""
    times = [float(value) for value in checkpoint_times]
    if not times or times[0] != 0.0 or any(b <= a for a, b in pairwise(times)):
        raise ValueError("checkpoint_times must start at zero and strictly increase")
    if steps_per_interval <= 0:
        raise ValueError("steps_per_interval must be positive")
    state = initial
    checkpoints = [state]
    for start, end in pairwise(times):
        exact_steps = (end - start) * steps_per_interval
        steps = round(exact_steps)
        if steps < 1 or abs(steps - exact_steps) > 1e-8:
            raise ValueError("checkpoint interval does not align with fixed step size")
        step_size = (end - start) / steps
        for index in range(steps):
            time = start + index * step_size
            k1 = field(time, state)
            k2 = field(time + step_size / 2, state + step_size * k1 / 2)
            k3 = field(time + step_size / 2, state + step_size * k2 / 2)
            k4 = field(time + step_size, state + step_size * k3)
            state = state + step_size * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        checkpoints.append(state)
    return torch.stack(checkpoints, dim=1)

