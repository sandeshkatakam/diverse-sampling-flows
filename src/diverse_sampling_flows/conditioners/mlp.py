"""Shared per-location patch MLP with context-only global summaries."""

from __future__ import annotations

import torch
from torch import nn

from diverse_sampling_flows.conditioners.base import (
    CoefficientConditioner,
    expand_time,
    initialize_coefficient_head,
)


class PatchMLPConditioner(CoefficientConditioner):
    def __init__(
        self,
        context_channels: int,
        pairs: int,
        ridges: int,
        width: int = 128,
        hidden_layers: int = 3,
    ) -> None:
        super().__init__()
        self.group_embedding = nn.Embedding(2, 8)
        input_channels = context_channels * 2 + 2 + 1 + 8
        layers: list[nn.Module] = [nn.Conv2d(input_channels, width, 1), nn.SiLU()]
        for _ in range(hidden_layers - 1):
            layers.extend((nn.Conv2d(width, width, 1), nn.SiLU()))
        self.backbone = nn.Sequential(*layers)
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
        batch, _, height, width = context_tensor.shape
        time = expand_time(t_normalized, batch).to(context_tensor)
        pooled = context_tensor.mean(dim=(-2, -1), keepdim=True).expand(
            -1, -1, height, width
        )
        y = torch.linspace(
            -1, 1, height, device=context_tensor.device, dtype=context_tensor.dtype
        )
        x = torch.linspace(
            -1, 1, width, device=context_tensor.device, dtype=context_tensor.dtype
        )
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack((yy, xx)).unsqueeze(0).expand(batch, -1, -1, -1)
        group = self.group_embedding(
            torch.full((batch,), group_id, device=context_tensor.device)
        ).to(context_tensor)
        group = group[:, :, None, None].expand(-1, -1, height, width)
        time_map = time[:, None, None, None].expand(-1, 1, height, width)
        inputs = torch.cat(
            (context_tensor, pooled, coordinates, time_map, group), dim=1
        )
        return self.head(self.backbone(inputs))

