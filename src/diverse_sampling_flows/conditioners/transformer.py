"""Small context-only 2x2-patch transformer conditioner."""

from __future__ import annotations

import torch
from torch import nn

from diverse_sampling_flows.conditioners.base import (
    CoefficientConditioner,
    expand_time,
    initialize_coefficient_head,
)


class SmallTransformerConditioner(CoefficientConditioner):
    def __init__(
        self,
        context_channels: int,
        pairs: int,
        ridges: int,
        width: int = 128,
        blocks: int = 4,
        heads: int = 4,
        mlp_ratio: float = 4.0,
        patch_size: int = 2,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patchify = nn.Conv2d(
            context_channels, width, patch_size, stride=patch_size
        )
        layer = nn.TransformerEncoderLayer(
            width,
            heads,
            int(width * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, blocks, enable_nested_tensor=False
        )
        self.group_embedding = nn.Embedding(2, width)
        self.time_projection = nn.Linear(1, width)
        self.head = nn.ConvTranspose2d(
            width,
            pairs * (1 + 4 * ridges),
            patch_size,
            stride=patch_size,
        )
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
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("spatial dimensions must be divisible by patch_size")
        time = expand_time(t_normalized, batch).to(context_tensor)
        features = self.patchify(context_tensor)
        patch_height, patch_width = features.shape[-2:]
        tokens = features.flatten(2).transpose(1, 2)
        group = self.group_embedding(
            torch.full((batch,), group_id, device=context_tensor.device)
        ).to(context_tensor)
        tokens = tokens + group[:, None] + self.time_projection(time[:, None])[:, None]
        tokens = self.transformer(tokens)
        features = tokens.transpose(1, 2).reshape(batch, -1, patch_height, patch_width)
        return self.head(features)

