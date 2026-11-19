"""Normalized RBF-kernel log-determinant diversity objective."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


class LogDetNumericalError(RuntimeError):
    def __init__(self, message: str, record: dict[str, Any]) -> None:
        super().__init__(message)
        self.record = record


@dataclass(frozen=True)
class BandwidthCalibration:
    bandwidth_sq: float
    median_squared_distance: float
    pair_ids: torch.Tensor
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "bandwidth_sq": self.bandwidth_sq,
            "median_squared_distance": self.median_squared_distance,
            "pair_ids": self.pair_ids.tolist(),
            "seed": self.seed,
        }


def normalize_features(features: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    return features / features.norm(dim=-1, keepdim=True).clamp_min(epsilon)


def rbf_logdet_score(
    features: torch.Tensor,
    *,
    bandwidth_sq: float,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    if bandwidth_sq <= 0 or epsilon <= 0:
        raise ValueError("bandwidth_sq and epsilon must be positive")
    if features.ndim == 2:
        features = features.unsqueeze(0)
    if features.ndim != 3:
        raise ValueError("features must be [B,K,D] or [K,D]")
    normalized = normalize_features(features).to(torch.float64)
    squared_distances = torch.cdist(normalized, normalized).square()
    kernel = torch.exp(-squared_distances / (2 * float(bandwidth_sq)))
    if not torch.isfinite(kernel).all():
        raise LogDetNumericalError(
            "RBF kernel contains nonfinite values",
            {"error": "nonfinite_kernel", "bandwidth_sq": float(bandwidth_sq)},
        )
    slots = features.shape[1]
    identity = torch.eye(slots, device=features.device, dtype=torch.float64)
    stabilized = (kernel + epsilon * identity) / (1 + epsilon)
    try:
        factor = torch.linalg.cholesky(stabilized)
    except RuntimeError as error:
        raise LogDetNumericalError(
            "Cholesky failed for the fixed objective kernel",
            {
                "error": "cholesky_failed",
                "bandwidth_sq": float(bandwidth_sq),
                "epsilon": float(epsilon),
                "cause": str(error),
            },
        ) from error
    return 2 * torch.log(torch.diagonal(factor, dim1=-2, dim2=-1)).sum(-1) / slots


def calibrate_bandwidth(
    features: torch.Tensor,
    *,
    seed: int,
    num_pairs: int,
    minimum_bandwidth_sq: float = 1e-4,
) -> BandwidthCalibration:
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("calibration features must be [N,D] with N >= 2")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    count = features.shape[0]
    pair_ids = []
    while len(pair_ids) < num_pairs:
        candidates = torch.randint(count, (max(16, num_pairs), 2), generator=generator)
        for first, second in candidates.tolist():
            if first != second:
                pair_ids.append((first, second))
                if len(pair_ids) == num_pairs:
                    break
    pairs = torch.tensor(pair_ids, dtype=torch.long)
    normalized = normalize_features(features)
    squared = (normalized[pairs[:, 0]] - normalized[pairs[:, 1]]).square().sum(-1)
    median = float(torch.median(squared).item())
    bandwidth_sq = max(median / (2 * math.log(2)), minimum_bandwidth_sq)
    return BandwidthCalibration(bandwidth_sq, median, pairs, int(seed))

