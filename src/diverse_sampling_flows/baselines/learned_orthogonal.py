"""Sequential permutation-conjugated learned pair rotations."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from diverse_sampling_flows.baselines.fixed_rotation import rotate_pairs
from diverse_sampling_flows.priors.masks import fixed_permutation


class LearnedOrthogonalCoupling(nn.Module):
    def __init__(
        self,
        latent_shape: Sequence[int],
        *,
        num_slots: int = 4,
        layers: int = 4,
        seed: int = 5101,
    ) -> None:
        super().__init__()
        self.latent_shape = tuple(int(value) for value in latent_shape)
        self.dimension = math.prod(self.latent_shape)
        if self.dimension % 2:
            raise ValueError("orthogonal pair maps require an even dimension")
        self.num_slots = num_slots
        self.layers = layers
        self.angles = nn.Parameter(
            torch.zeros(num_slots - 1, layers, self.dimension // 2)
        )
        permutations = [
            fixed_permutation(self.dimension, seed + transition * layers + layer)
            for transition in range(num_slots - 1)
            for layer in range(layers)
        ]
        self.register_buffer(
            "permutations",
            torch.stack(permutations).reshape(num_slots - 1, layers, self.dimension),
        )

    def fixed_quarter_turn(self, value: torch.Tensor) -> torch.Tensor:
        return rotate_pairs(value, math.pi / 2)

    def residual_layer(
        self, value: torch.Tensor, transition: int, layer: int
    ) -> torch.Tensor:
        flattened = value.flatten(1)
        permutation = self.permutations[transition, layer]
        inverse = torch.argsort(permutation)
        permuted = flattened[:, permutation]
        rotated = rotate_pairs(permuted, self.angles[transition, layer])
        return rotated[:, inverse].reshape_as(value)

    def transition(self, value: torch.Tensor, transition: int) -> torch.Tensor:
        result = self.fixed_quarter_turn(value)
        for layer in range(self.layers):
            result = self.residual_layer(result, transition, layer)
        return result

    def forward(self, seed: torch.Tensor) -> torch.Tensor:
        slots = [seed]
        state = seed
        for transition in range(self.num_slots - 1):
            state = self.transition(state, transition)
            slots.append(state)
        return torch.stack(slots, dim=1)

