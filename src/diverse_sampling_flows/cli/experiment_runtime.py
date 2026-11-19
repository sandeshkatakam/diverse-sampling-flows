"""Factories and latent recipes shared by training and evaluation workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn

from diverse_sampling_flows.baselines.conditional_rotation import (
    ConditionalRotationCoupling,
)
from diverse_sampling_flows.baselines.fixed_rotation import fixed_rotation_slots
from diverse_sampling_flows.baselines.learned_orthogonal import (
    LearnedOrthogonalCoupling,
)
from diverse_sampling_flows.baselines.sampling import (
    sample_balanced_gaussian,
    sample_iid,
)
from diverse_sampling_flows.conditioners.factory import build_conditioner
from diverse_sampling_flows.priors.masked_skew import (
    CoefficientBounds,
    MaskedSkewField,
)
from diverse_sampling_flows.solvers.rk4 import integrate_rk4
from diverse_sampling_flows.utils.checkpointing import load_checkpoint
from diverse_sampling_flows.utils.seeding import SeedStreams


def build_trainable_method(
    config: DictConfig, method: str, device: torch.device | str
) -> nn.Module:
    shape = tuple(int(value) for value in config.model.standardized_latent_shape)
    channels = shape[0]
    slots = int(config.experiment.num_samples)
    if method == "learned_orthogonal":
        return LearnedOrthogonalCoupling(
            shape,
            num_slots=slots,
            seed=int(config.prior.permutation_seed),
        ).to(device)
    if method == "learned_conditional_rotation":
        return ConditionalRotationCoupling(
            channels=channels, num_slots=slots, width=int(config.conditioner.width)
        ).to(device)
    if method in {
        "skew_initial",
        "skew_trained",
        "independent_seed_negative_control",
    }:
        architecture = str(config.conditioner.architecture)
        width = (
            int(config.conditioner.transformer_width)
            if architecture == "transformer"
            else int(config.conditioner.width)
        )
        conditioner = build_conditioner(
            architecture,
            context_channels=channels // 2,
            pairs=channels // 4,
            ridges=int(config.prior.ridges_per_pair),
            width=width,
            residual_blocks=int(config.conditioner.residual_blocks),
            group_norm_groups=int(config.conditioner.group_norm_groups),
            group_norm_epsilon=float(config.conditioner.group_norm_epsilon),
            mlp_hidden_layers=int(config.conditioner.mlp_hidden_layers),
            transformer_blocks=int(config.conditioner.transformer_blocks),
            transformer_heads=int(config.conditioner.transformer_heads),
            transformer_mlp_ratio=float(config.conditioner.transformer_mlp_ratio),
            patch_size=int(config.conditioner.patch_size),
        )
        return MaskedSkewField(
            conditioner,
            channels=channels,
            ridges=int(config.prior.ridges_per_pair),
            num_slots=slots,
            bounds=CoefficientBounds(
                c_max=float(config.prior.c_max),
                a_max=float(config.prior.a_max),
                b_max=float(config.prior.b_max),
                e_max=float(config.prior.e_max),
            ),
        ).to(device)
    raise ValueError(f"method {method!r} has no trainable model")


def apply_checkpoint(
    model: nn.Module, checkpoint: Path, *, ema: bool, device: torch.device | str
) -> dict[str, Any]:
    payload = load_checkpoint(checkpoint, map_location=device)
    model.load_state_dict(payload["model"])
    if ema:
        with torch.no_grad():
            named = dict(model.named_parameters())
            for name, value in payload["ema"]["shadow"].items():
                named[name].copy_(value.to(device=named[name].device))
    return payload


def sample_method_latents(
    config: DictConfig,
    method: str,
    streams: SeedStreams,
    *,
    batch: int,
    device: torch.device | str,
    model: nn.Module | None = None,
    namespace: str = "training",
) -> torch.Tensor:
    shape = tuple(int(value) for value in config.model.standardized_latent_shape)
    slots = int(config.experiment.num_samples)
    generator = streams.generator(namespace, device)
    streams.counters[namespace] = streams.counters.get(namespace, 0) + 1
    if method == "iid":
        return sample_iid(
            batch, slots, shape, generator=generator, device=device
        )
    if method == "balanced_gaussian":
        return sample_balanced_gaussian(
            batch, slots, shape, generator=generator, device=device
        )
    if method == "fixed_rotation":
        seed = torch.randn((batch, *shape), generator=generator, device=device)
        return fixed_rotation_slots(seed, slots)
    if model is None:
        raise ValueError(f"method {method!r} requires a model")
    if method in {"learned_orthogonal", "learned_conditional_rotation"}:
        seed = torch.randn((batch, *shape), generator=generator, device=device)
        return model(seed)
    if method in {"skew_initial", "skew_trained"}:
        seed = torch.randn((batch, *shape), generator=generator, device=device)
        return integrate_rk4(
            model,
            seed,
            list(config.experiment.checkpoint_times),
            int(config.prior.steps_per_interval),
        )
    if method == "independent_seed_negative_control":
        seeds = sample_iid(
            batch, slots, shape, generator=generator, device=device
        )
        results = []
        for slot, end_time in enumerate(config.experiment.checkpoint_times):
            if slot == 0:
                results.append(seeds[:, slot])
            else:
                results.append(
                    integrate_rk4(
                        model,
                        seeds[:, slot],
                        [0.0, float(end_time)],
                        int(config.prior.steps_per_interval),
                    )[:, -1]
                )
        return torch.stack(results, dim=1)
    raise ValueError(f"unknown method: {method}")
