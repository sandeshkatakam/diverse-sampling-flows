"""Conditioner architecture factory."""

from __future__ import annotations

from diverse_sampling_flows.conditioners.base import CoefficientConditioner
from diverse_sampling_flows.conditioners.mlp import PatchMLPConditioner
from diverse_sampling_flows.conditioners.resnet import ResidualCNNConditioner
from diverse_sampling_flows.conditioners.transformer import SmallTransformerConditioner


def build_conditioner(
    architecture: str,
    *,
    context_channels: int,
    pairs: int,
    ridges: int,
    width: int = 64,
    residual_blocks: int = 4,
    group_norm_groups: int = 8,
    group_norm_epsilon: float = 1e-5,
    mlp_hidden_layers: int = 3,
    transformer_blocks: int = 4,
    transformer_heads: int = 4,
    transformer_mlp_ratio: float = 4.0,
    patch_size: int = 2,
) -> CoefficientConditioner:
    if architecture == "mlp":
        return PatchMLPConditioner(
            context_channels, pairs, ridges, width, mlp_hidden_layers
        )
    if architecture == "resnet":
        return ResidualCNNConditioner(
            context_channels,
            pairs,
            ridges,
            width,
            residual_blocks,
            group_norm_groups,
            group_norm_epsilon,
        )
    if architecture == "transformer":
        return SmallTransformerConditioner(
            context_channels,
            pairs,
            ridges,
            width,
            transformer_blocks,
            transformer_heads,
            transformer_mlp_ratio,
            patch_size,
        )
    raise ValueError(f"unknown conditioner architecture: {architecture}")

