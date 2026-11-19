"""Serializable channel masks, active pairs, and seeded permutations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class MaskGroup:
    group_id: int
    context_channels: tuple[int, ...]
    active_pairs: tuple[tuple[int, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_channel_mask_groups(channels: int) -> tuple[MaskGroup, MaskGroup]:
    if channels < 4 or channels % 4:
        raise ValueError("channels must be positive and divisible by four")
    half = channels // 2
    first_pairs = tuple((index, index + 1) for index in range(0, half, 2))
    second_pairs = tuple(
        (index, index + 1) for index in range(half, channels, 2)
    )
    return (
        MaskGroup(0, tuple(range(half, channels)), first_pairs),
        MaskGroup(1, tuple(range(half)), second_pairs),
    )


def fixed_permutation(size: int, seed: int) -> torch.Tensor:
    if size <= 0:
        raise ValueError("permutation size must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randperm(size, generator=generator)

