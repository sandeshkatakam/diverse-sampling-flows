"""AdamW grouping and warmup-cosine schedule."""

from __future__ import annotations

import math

import torch
from torch import nn

_NORMALIZATION_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.LayerNorm,
    nn.LocalResponseNorm,
)


def build_adamw(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.99),
    epsilon: float = 1e-8,
) -> torch.optim.AdamW:
    normalization_parameters = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, _NORMALIZATION_TYPES)
        for parameter in module.parameters(recurse=False)
    }
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith("bias") or id(parameter) in normalization_parameters:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=betas,
        eps=epsilon,
    )


def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_updates: int,
    total_updates: int,
    minimum_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    base_lr = max(group["lr"] for group in optimizer.param_groups)
    minimum_factor = minimum_lr / base_lr

    def multiplier(step: int) -> float:
        if warmup_updates and step <= warmup_updates:
            return step / warmup_updates
        progress = (step - warmup_updates) / max(1, total_updates - warmup_updates)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return minimum_factor + (1 - minimum_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)

