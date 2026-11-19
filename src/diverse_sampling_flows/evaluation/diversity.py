"""Set-level cosine diversity and prompt-alignment summaries."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from diverse_sampling_flows.objectives.clip_logdet import normalize_features


@dataclass(frozen=True)
class PairwiseDiversity:
    distances: torch.Tensor
    mean: torch.Tensor
    minimum: torch.Tensor


@dataclass(frozen=True)
class PromptAlignmentSummary:
    raw_slots: torch.Tensor
    per_slot_mean: torch.Tensor
    set_mean: torch.Tensor
    lower_tail: torch.Tensor


def pairwise_cosine_diversity(features: torch.Tensor) -> PairwiseDiversity:
    if features.ndim != 3:
        raise ValueError("features must be [B,K,D]")
    normalized = normalize_features(features)
    similarity = normalized @ normalized.transpose(-1, -2)
    rows, columns = torch.triu_indices(
        features.shape[1], features.shape[1], offset=1, device=features.device
    )
    distances = 1 - similarity[:, rows, columns]
    return PairwiseDiversity(distances, distances.mean(-1), distances.min(-1).values)


def summarize_prompt_alignment(
    scores: torch.Tensor, *, lower_quantile: float = 0.05
) -> PromptAlignmentSummary:
    if scores.ndim != 2:
        raise ValueError("prompt scores must be [B,K]")
    return PromptAlignmentSummary(
        scores,
        scores.mean(0),
        scores.mean(1),
        torch.quantile(scores.flatten(), lower_quantile),
    )
