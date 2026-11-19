"""Common conditioner interface and non-dead ridge initialization."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
from torch import nn


class CoefficientConditioner(nn.Module, ABC):
    @abstractmethod
    def coefficients(
        self,
        t_normalized: torch.Tensor,
        context_tensor: torch.Tensor,
        group_id: int,
        optional_prompt_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return raw maps shaped ``[B, pairs*(1+4R), H, W]``."""


def initial_raw_bias(pairs: int, ridges: int) -> torch.Tensor:
    per_pair = torch.zeros(pairs, 1 + 4 * ridges)
    per_pair[:, 1 : 1 + ridges] = math.atanh(0.02 / 0.25)
    per_pair[:, 1 + ridges : 1 + 2 * ridges] = math.atanh(0.5)
    return per_pair.flatten()


def initialize_coefficient_head(head: nn.Module, pairs: int, ridges: int) -> None:
    if not hasattr(head, "weight") or not hasattr(head, "bias"):
        raise TypeError("coefficient head must expose weight and bias")
    nn.init.normal_(head.weight, mean=0.0, std=1e-3)
    with torch.no_grad():
        head.bias.copy_(initial_raw_bias(pairs, ridges))


def expand_time(t_normalized: torch.Tensor, batch: int) -> torch.Tensor:
    time = torch.as_tensor(t_normalized)
    if time.ndim == 0:
        time = time.expand(batch)
    if time.shape != (batch,):
        raise ValueError(f"time must be scalar or [B], got {tuple(time.shape)}")
    return time

