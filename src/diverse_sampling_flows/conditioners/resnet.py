"""Residual CNN conditioner with time/group FiLM."""

from __future__ import annotations

import torch
from torch import nn

from diverse_sampling_flows.conditioners.base import (
    CoefficientConditioner,
    expand_time,
    initialize_coefficient_head,
)


class _ResidualBlock(nn.Module):
    def __init__(self, width: int, groups: int, epsilon: float) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, width, eps=epsilon)
        self.conv1 = nn.Conv2d(width, width, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, width, eps=epsilon)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        self.activation = nn.SiLU()

    def forward(
        self, value: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.norm1(value) * (1 + scale) + shift
        hidden = self.conv1(self.activation(hidden))
        hidden = self.conv2(self.activation(self.norm2(hidden)))
        return value + hidden


class ResidualCNNConditioner(CoefficientConditioner):
    def __init__(
        self,
        context_channels: int,
        pairs: int,
        ridges: int,
        width: int = 64,
        blocks: int = 4,
        norm_groups: int = 8,
        norm_epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        groups = min(norm_groups, width)
        while width % groups:
            groups -= 1
        self.input = nn.Conv2d(context_channels, width, 3, padding=1)
        self.blocks = nn.ModuleList(
            [_ResidualBlock(width, groups, norm_epsilon) for _ in range(blocks)]
        )
        self.condition = nn.Sequential(nn.Linear(9, width * 2), nn.SiLU())
        self.group_embedding = nn.Embedding(2, 8)
        self.head = nn.Conv2d(width, pairs * (1 + 4 * ridges), 1)
        initialize_coefficient_head(self.head, pairs, ridges)

    def coefficients(
        self,
        t_normalized: torch.Tensor,
        context_tensor: torch.Tensor,
        group_id: int,
        optional_prompt_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del optional_prompt_context
        batch = context_tensor.shape[0]
        time = expand_time(t_normalized, batch).to(context_tensor)
        group = self.group_embedding(
            torch.full((batch,), group_id, device=context_tensor.device)
        ).to(context_tensor)
        scale, shift = self.condition(
            torch.cat((time[:, None], group), dim=1)
        ).chunk(2, dim=1)
        scale, shift = scale[:, :, None, None], shift[:, :, None, None]
        hidden = self.input(context_tensor)
        for block in self.blocks:
            hidden = block(hidden, scale, shift)
        return self.head(hidden)

