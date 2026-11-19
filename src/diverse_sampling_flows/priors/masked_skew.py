"""Exact analytic velocity for a context-masked implicit skew field."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from diverse_sampling_flows.conditioners.base import CoefficientConditioner
from diverse_sampling_flows.priors.masks import MaskGroup, build_channel_mask_groups


@dataclass(frozen=True)
class CoefficientBounds:
    c_max: float = math.pi / 4
    a_max: float = 0.25
    b_max: float = 1.0
    e_max: float = 2.0


def bounded_coefficients(
    raw: torch.Tensor,
    *,
    num_slots: int,
    bounds: CoefficientBounds,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if raw.ndim != 5:
        raise ValueError("raw coefficients must be [B,pairs,1+4R,H,W]")
    ridges = (raw.shape[2] - 1) // 4
    if raw.shape[2] != 1 + 4 * ridges or ridges < 1:
        raise ValueError("coefficient axis must have size 1+4R")
    c = 2 * math.pi / num_slots + bounds.c_max * torch.tanh(raw[:, :, 0])
    alpha, beta, delta, offset = raw[:, :, 1:].split(ridges, dim=2)
    return (
        c,
        bounds.a_max * torch.tanh(alpha),
        bounds.b_max * torch.tanh(beta),
        bounds.b_max * torch.tanh(delta),
        bounds.e_max * torch.tanh(offset),
    )


class MaskedSkewField(nn.Module):
    """Evaluate ``Div K - Kz`` without constructing ``K`` or a Jacobian."""

    def __init__(
        self,
        conditioner: CoefficientConditioner,
        *,
        channels: int = 32,
        groups: tuple[MaskGroup, ...] | None = None,
        ridges: int = 1,
        num_slots: int = 4,
        bounds: CoefficientBounds | None = None,
    ) -> None:
        super().__init__()
        self.conditioner = conditioner
        self.channels = channels
        self.groups = groups or build_channel_mask_groups(channels)
        self.ridges = ridges
        self.num_slots = num_slots
        self.bounds = bounds or CoefficientBounds()

    def forward(
        self,
        t: torch.Tensor | float,
        z: torch.Tensor,
        prompt_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z.ndim != 4 or z.shape[1] != self.channels:
            raise ValueError(f"z must be [B,{self.channels},H,W]")
        batch, _, height, width = z.shape
        time = torch.as_tensor(t, device=z.device, dtype=z.dtype)
        if time.ndim == 0:
            time = time.expand(batch)
        normalized_time = time / (self.num_slots - 1)
        velocity = torch.zeros_like(z)
        for group in self.groups:
            context = z[:, group.context_channels]
            raw = self.conditioner.coefficients(
                normalized_time, context, group.group_id, prompt_context
            )
            pairs = len(group.active_pairs)
            raw = raw.reshape(batch, pairs, 1 + 4 * self.ridges, height, width)
            c, alpha, beta, delta, offset = bounded_coefficients(
                raw, num_slots=self.num_slots, bounds=self.bounds
            )
            u_indices = [pair[0] for pair in group.active_pairs]
            v_indices = [pair[1] for pair in group.active_pairs]
            u, v = z[:, u_indices], z[:, v_indices]
            ridge_argument = (
                beta * u[:, :, None]
                + delta * v[:, :, None]
                + offset
            )
            ridge_tanh = torch.tanh(ridge_argument)
            h = c + (alpha * ridge_tanh).sum(dim=2)
            derivative_factor = 1 - ridge_tanh.square()
            w_u = (alpha * delta * derivative_factor).sum(dim=2) - h * v
            w_v = -(alpha * beta * derivative_factor).sum(dim=2) + h * u
            indices = torch.tensor(
                [item for pair in group.active_pairs for item in pair],
                device=z.device,
            )
            contributions = torch.stack((w_u, w_v), dim=2).reshape(
                batch, pairs * 2, height, width
            )
            velocity = velocity.index_add(1, indices, contributions)
        return velocity
