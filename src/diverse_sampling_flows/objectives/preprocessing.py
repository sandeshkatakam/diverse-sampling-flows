"""Tensor-only differentiable encoder preprocessing."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as functional


def preprocess_images(
    images: torch.Tensor,
    *,
    size: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must be [N,3,H,W]")
    height, width = images.shape[-2:]
    scale = size / min(height, width)
    resized_height = max(size, round(height * scale))
    resized_width = max(size, round(width * scale))
    resized = functional.interpolate(
        images,
        size=(resized_height, resized_width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    top = (resized_height - size) // 2
    left = (resized_width - size) // 2
    cropped = resized[:, :, top : top + size, left : left + size]
    mean_tensor = torch.as_tensor(mean, device=images.device, dtype=images.dtype)[
        None, :, None, None
    ]
    std_tensor = torch.as_tensor(std, device=images.device, dtype=images.dtype)[
        None, :, None, None
    ]
    return (cropped - mean_tensor) / std_tensor

