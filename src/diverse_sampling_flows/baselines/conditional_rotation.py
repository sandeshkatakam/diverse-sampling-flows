"""Nonlinear conditional channel-pair rotations preserving Gaussian law."""

from __future__ import annotations

import math

import torch
from torch import nn

from diverse_sampling_flows.priors.masks import build_channel_mask_groups


class _AngleNetwork(nn.Module):
    def __init__(self, context_channels: int, pairs: int, width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(context_channels, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, pairs, 1),
        )
        nn.init.normal_(self.network[-1].weight, std=1e-3)
        nn.init.constant_(self.network[-1].bias, math.atanh(0.5))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return (math.pi / 2) * torch.tanh(self.network(context))


class ConditionalRotationCoupling(nn.Module):
    def __init__(
        self,
        *,
        channels: int = 32,
        num_slots: int = 4,
        layers_per_transition: int = 4,
        width: int = 64,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.num_slots = num_slots
        self.layers_per_transition = layers_per_transition
        self.groups = build_channel_mask_groups(channels)
        pairs = channels // 4
        self.angle_networks = nn.ModuleList(
            [
                _AngleNetwork(channels // 2, pairs, width)
                for _ in range((num_slots - 1) * layers_per_transition)
            ]
        )

    def _layer(
        self, state: torch.Tensor, transition: int, layer: int
    ) -> torch.Tensor:
        group = self.groups[layer % len(self.groups)]
        network = self.angle_networks[
            transition * self.layers_per_transition + layer
        ]
        angles = network(state[:, group.context_channels])
        u_indices = [pair[0] for pair in group.active_pairs]
        v_indices = [pair[1] for pair in group.active_pairs]
        u, v = state[:, u_indices], state[:, v_indices]
        cosine, sine = torch.cos(angles), torch.sin(angles)
        rotated = torch.stack(
            (cosine * u - sine * v, sine * u + cosine * v), dim=2
        ).flatten(1, 2)
        indices = torch.tensor(
            [index for pair in group.active_pairs for index in pair],
            device=state.device,
        )
        return state.index_copy(1, indices, rotated)

    def transition(self, state: torch.Tensor, transition: int) -> torch.Tensor:
        result = state
        for layer in range(self.layers_per_transition):
            result = self._layer(result, transition, layer)
        return result

    def forward(self, seed: torch.Tensor) -> torch.Tensor:
        slots = [seed]
        state = seed
        for transition in range(self.num_slots - 1):
            state = self.transition(state, transition)
            slots.append(state)
        return torch.stack(slots, dim=1)
