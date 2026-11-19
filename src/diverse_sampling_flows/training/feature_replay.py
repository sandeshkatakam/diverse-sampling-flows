"""Exact first-order feature VJP replay for frozen expensive models."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ReplayResult:
    loss: torch.Tensor
    replayed_slots: list[int]
    feature_forwards: int
    input_vjps: int


def feature_replay_backward(
    prior_checkpoints: Sequence[torch.Tensor],
    feature_function: Callable[[torch.Tensor], torch.Tensor],
    set_loss_function: Callable[[torch.Tensor], torch.Tensor],
    *,
    accumulation_scale: float = 1.0,
) -> ReplayResult:
    if not prior_checkpoints:
        raise ValueError("at least one checkpoint is required")
    with torch.no_grad():
        feature_values = [
            feature_function(checkpoint) for checkpoint in prior_checkpoints
        ]
    feature_leaf = torch.stack(feature_values, dim=1).detach().requires_grad_(True)
    loss = set_loss_function(feature_leaf)
    feature_gradients = torch.autograd.grad(loss, feature_leaf)[0]
    tensors: list[torch.Tensor] = []
    latent_gradients: list[torch.Tensor] = []
    replayed_slots: list[int] = []
    for slot, checkpoint in enumerate(prior_checkpoints):
        if not checkpoint.requires_grad:
            continue
        replay_input = checkpoint.detach().requires_grad_(True)
        replay_features = feature_function(replay_input)
        latent_vjp = torch.autograd.grad(
            replay_features,
            replay_input,
            grad_outputs=feature_gradients[:, slot],
        )[0]
        tensors.append(checkpoint)
        latent_gradients.append(latent_vjp * accumulation_scale)
        replayed_slots.append(slot)
    if tensors:
        torch.autograd.backward(tensors, latent_gradients)
    return ReplayResult(
        loss.detach(),
        replayed_slots,
        feature_forwards=len(prior_checkpoints) + len(replayed_slots),
        input_vjps=len(replayed_slots),
    )

