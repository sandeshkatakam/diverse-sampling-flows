"""Projection, MMD, sliced-distance, and IID-null statistics."""

from __future__ import annotations

import torch


def fixed_orthonormal_projection(
    dimension: int,
    projected_dimension: int,
    *,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if not 0 < projected_dimension <= dimension:
        raise ValueError("projected_dimension must be in [1, dimension]")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    matrix = torch.randn(
        dimension, projected_dimension, generator=generator, dtype=dtype
    )
    projection, _ = torch.linalg.qr(matrix, mode="reduced")
    return projection


def unbiased_rbf_mmd(
    first: torch.Tensor, second: torch.Tensor, *, bandwidth_sq: float
) -> torch.Tensor:
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]:
        raise ValueError("MMD samples must be [N,D] and [M,D]")
    if first.shape[0] < 2 or second.shape[0] < 2:
        raise ValueError("unbiased MMD requires at least two samples per group")
    kernel_xx = torch.exp(-torch.cdist(first, first).square() / (2 * bandwidth_sq))
    kernel_yy = torch.exp(-torch.cdist(second, second).square() / (2 * bandwidth_sq))
    kernel_xy = torch.exp(-torch.cdist(first, second).square() / (2 * bandwidth_sq))
    count_x, count_y = first.shape[0], second.shape[0]
    xx = (kernel_xx.sum() - kernel_xx.diagonal().sum()) / (count_x * (count_x - 1))
    yy = (kernel_yy.sum() - kernel_yy.diagonal().sum()) / (count_y * (count_y - 1))
    return xx + yy - 2 * kernel_xy.mean()


def sliced_wasserstein(
    first: torch.Tensor, second: torch.Tensor, *, projections: torch.Tensor
) -> torch.Tensor:
    count = min(first.shape[0], second.shape[0])
    projected_first = (first @ projections).sort(dim=0).values[:count]
    projected_second = (second @ projections).sort(dim=0).values[:count]
    return (projected_first - projected_second).abs().mean()


def calibrate_iid_null(
    *, rows: int, slots: int, dimension: int, repeats: int, seed: int
) -> dict[str, object]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    maxima = []
    for _ in range(repeats):
        sample = torch.randn(rows, slots, dimension, generator=generator)
        standardized_means = sample.mean(0).abs() * rows**0.5
        variance_errors = (sample.var(0) - 1).abs() * (rows / 2) ** 0.5
        maxima.append(
            float(torch.maximum(standardized_means.max(), variance_errors.max()))
        )
    values = torch.tensor(maxima)
    return {
        "rows": rows,
        "slots": slots,
        "dimension": dimension,
        "repeats": repeats,
        "maximum_statistics": maxima,
        "maximum_statistic_quantile_0_95": float(torch.quantile(values, 0.95)),
    }

