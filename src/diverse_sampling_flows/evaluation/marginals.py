"""Streaming raw-slot Gaussian prior marginal diagnostics."""

from __future__ import annotations

import math

import torch

from diverse_sampling_flows.evaluation.statistics import fixed_orthonormal_projection


class PriorMarginalAccumulator:
    def __init__(
        self,
        *,
        slots: int,
        dimension: int,
        projection_dimensions: int = 64,
        seed: int = 6101,
    ) -> None:
        self.slots, self.dimension = slots, dimension
        projected = min(projection_dimensions, dimension)
        self.projection = fixed_orthonormal_projection(
            dimension, projected, seed=seed, dtype=torch.float64
        )
        self.count = 0
        self.sum = torch.zeros(slots, dimension, dtype=torch.float64)
        self.sum_square = torch.zeros_like(self.sum)
        self.radial_sum = torch.zeros(slots, dtype=torch.float64)
        self.radial_sum_square = torch.zeros_like(self.radial_sum)
        self.projected_sum = torch.zeros(slots, projected, dtype=torch.float64)
        self.projected_outer = torch.zeros(
            slots, projected, projected, dtype=torch.float64
        )

    def update(self, rows: torch.Tensor) -> None:
        if rows.shape[:2] != (rows.shape[0], self.slots):
            raise ValueError(f"rows must have slot axis of size {self.slots}")
        flattened = rows.detach().to(torch.float64).reshape(
            rows.shape[0], self.slots, -1
        )
        if flattened.shape[2] != self.dimension:
            raise ValueError(f"expected flattened dimension {self.dimension}")
        self.count += flattened.shape[0]
        self.sum += flattened.sum(0).cpu()
        self.sum_square += flattened.square().sum(0).cpu()
        norm_square = flattened.square().sum(-1)
        radial = (norm_square - self.dimension) / math.sqrt(2 * self.dimension)
        self.radial_sum += radial.sum(0).cpu()
        self.radial_sum_square += radial.square().sum(0).cpu()
        projected = flattened.cpu() @ self.projection
        self.projected_sum += projected.sum(0)
        self.projected_outer += torch.einsum("bkp,bkq->kpq", projected, projected)

    def finalize(self) -> dict[str, object]:
        if self.count < 2:
            raise ValueError("at least two rows are required")
        mean = self.sum / self.count
        variance = (self.sum_square - self.count * mean.square()) / (self.count - 1)
        radial_mean = self.radial_sum / self.count
        radial_variance = (
            self.radial_sum_square - self.count * radial_mean.square()
        ) / (self.count - 1)
        projected_mean = self.projected_sum / self.count
        projected_covariance = (
            self.projected_outer
            - self.count * torch.einsum("kp,kq->kpq", projected_mean, projected_mean)
        ) / (self.count - 1)
        return {
            "count": self.count,
            "coordinate_mean": {
                "per_slot": mean,
                "rms_error": mean.square().mean(-1).sqrt(),
            },
            "coordinate_variance": {
                "per_slot": variance,
                "rms_error": (variance - 1).square().mean(-1).sqrt(),
            },
            "radial": {"mean": radial_mean, "variance": radial_variance},
            "projected": {
                "mean": projected_mean,
                "covariance": projected_covariance,
                "projection": self.projection,
            },
        }


def output_feature_marginals(
    slots: torch.Tensor,
    reference: torch.Tensor,
    *,
    bandwidth_sq: float,
) -> dict[str, torch.Tensor]:
    from diverse_sampling_flows.evaluation.statistics import unbiased_rbf_mmd

    return {
        "mmd_per_slot": torch.stack(
            [
                unbiased_rbf_mmd(slots[:, index], reference, bandwidth_sq=bandwidth_sq)
                for index in range(slots.shape[1])
            ]
        )
    }

