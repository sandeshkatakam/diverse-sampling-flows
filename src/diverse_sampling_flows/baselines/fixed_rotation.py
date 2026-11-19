"""Analytic shared-seed pair rotations."""

from __future__ import annotations

import math

import torch


def rotate_pairs(value: torch.Tensor, angle: torch.Tensor | float) -> torch.Tensor:
    flattened = value.flatten(1)
    if flattened.shape[1] % 2:
        raise ValueError("pair rotation requires an even flattened dimension")
    pairs = flattened.reshape(flattened.shape[0], -1, 2)
    angles = torch.as_tensor(angle, device=value.device, dtype=value.dtype)
    while angles.ndim < 2:
        angles = angles.unsqueeze(0)
    cosine, sine = torch.cos(angles), torch.sin(angles)
    first, second = pairs.unbind(dim=-1)
    rotated = torch.stack(
        (cosine * first - sine * second, sine * first + cosine * second), dim=-1
    )
    return rotated.reshape_as(value)


def fixed_rotation_slots(seed: torch.Tensor, num_slots: int = 4) -> torch.Tensor:
    return torch.stack(
        [
            rotate_pairs(seed, 2 * math.pi * slot / num_slots)
            for slot in range(num_slots)
        ],
        dim=1,
    )

