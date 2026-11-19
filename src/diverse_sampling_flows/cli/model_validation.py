"""Pinned real-model parity, gradient, replay, and RNG validation."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import torch
from diffusers import SanaSprintPipeline
from omegaconf import DictConfig

from diverse_sampling_flows.generators.base import configure_deterministic_backend
from diverse_sampling_flows.generators.sana_sprint import SanaSprintAdapter
from diverse_sampling_flows.objectives.clip_logdet import rbf_logdet_score
from diverse_sampling_flows.objectives.encoders import CLIPImageEncoder
from diverse_sampling_flows.training.feature_replay import feature_replay_backward


def _official_image(
    pipeline,
    noise: torch.Tensor,
    embeddings: torch.Tensor,
    mask: torch.Tensor,
    config: DictConfig,
) -> torch.Tensor:
    return pipeline(
        prompt=None,
        prompt_embeds=embeddings,
        prompt_attention_mask=mask,
        latents=noise.clone(),
        height=int(config.model.height),
        width=int(config.model.width),
        num_inference_steps=1,
        max_timesteps=float(config.model.max_timesteps),
        intermediate_timesteps=None,
        guidance_scale=float(config.model.guidance_scale),
        max_sequence_length=int(config.model.max_sequence_length),
        clean_caption=False,
        use_resolution_binning=False,
        output_type="pt",
    ).images.float()


def run_model_validation(
    config: DictConfig, *, device: str, output_directory: Path
) -> dict[str, Any]:
    del output_directory
    policy = configure_deterministic_backend()
    torch.cuda.reset_peak_memory_stats()
    prompt = str(config.experiment.prompt)
    pipeline = SanaSprintPipeline.from_pretrained(
        str(config.model.model_id),
        revision=str(config.model.revision),
        torch_dtype=torch.bfloat16,
        cache_dir=str(config.paths.cache_root),
        local_files_only=True,
    ).to(device)
    with torch.no_grad():
        embeddings, mask = pipeline.encode_prompt(
            prompt,
            num_images_per_prompt=1,
            device=torch.device(device),
            clean_caption=False,
            max_sequence_length=int(config.model.max_sequence_length),
            complex_human_instruction=None,
        )
    adapter = SanaSprintAdapter(
        transformer=pipeline.transformer,
        vae=pipeline.vae,
        revision=str(config.model.revision),
    )
    adapter.set_conditioning(prompt, embeddings, mask)
    parity = []
    saved_noise = None
    for seed in (17, 29):
        noise = torch.randn(
            (1, *config.model.standardized_latent_shape),
            generator=torch.Generator(device=device).manual_seed(seed),
            device=device,
        )
        saved_noise = noise
        official = _official_image(pipeline, noise, embeddings, mask, config)
        actual = adapter.generate(noise, prompt=prompt, differentiable=False)
        parity.append(
            {
                "seed": seed,
                "max_absolute_error": float((official - actual).abs().max()),
                "mean_absolute_error": float((official - actual).abs().mean()),
            }
        )
    assert saved_noise is not None
    first = adapter.generate(saved_noise, prompt=prompt, differentiable=False)
    torch.randn(2048, device=device)
    second = adapter.generate(saved_noise, prompt=prompt, differentiable=False)
    rng_independent = torch.equal(first, second)
    del pipeline, official, actual, first, second
    gc.collect()
    torch.cuda.empty_cache()

    clip = CLIPImageEncoder.from_pretrained(
        str(config.objective.clip_model_id),
        str(config.objective.clip_revision),
        cache_dir=str(config.paths.cache_root),
        local_files_only=True,
    ).to(device)
    # CUDA lacks deterministic backward implementations for antialiased bicubic
    # resize and memory-efficient attention. Forward determinism remains strict.
    torch.use_deterministic_algorithms(True, warn_only=True)
    noise = saved_noise.detach().clone().requires_grad_(True)
    image = adapter.generate(noise, prompt=prompt, differentiable=True)
    features = clip(image)
    scalar = features[:, :32].sum()
    scalar.backward()
    gradient = noise.grad.detach().clone()
    input_gradient = {
        "finite": bool(torch.isfinite(gradient).all()),
        "norm": float(gradient.norm()),
        "generator_parameter_gradients": sum(
            parameter.grad is not None for parameter in adapter.parameters()
        ),
        "clip_parameter_gradients": sum(
            parameter.grad is not None for parameter in clip.parameters()
        ),
    }
    del image, features, scalar, noise, gradient
    gc.collect()
    torch.cuda.empty_cache()

    bandwidth_sq = 0.5
    anchor = saved_noise.detach()
    parameter = torch.nn.Parameter(torch.tensor(1.0, device=device))

    def feature(value: torch.Tensor) -> torch.Tensor:
        return clip(
            adapter.generate(
                value, prompt=prompt, differentiable=torch.is_grad_enabled()
            )
        )

    with torch.no_grad():
        anchor_feature = feature(anchor)
    dependent = anchor * parameter
    monolithic_features = torch.stack((anchor_feature, feature(dependent)), dim=1)
    monolithic_loss = -rbf_logdet_score(
        monolithic_features,
        bandwidth_sq=bandwidth_sq,
        epsilon=float(config.objective.epsilon),
    ).mean()
    monolithic_gradient = torch.autograd.grad(monolithic_loss, parameter)[0].detach()
    dependent = anchor * parameter

    def set_loss(values: torch.Tensor) -> torch.Tensor:
        return -rbf_logdet_score(
            values,
            bandwidth_sq=bandwidth_sq,
            epsilon=float(config.objective.epsilon),
        ).mean()

    replay = feature_replay_backward([anchor, dependent], feature, set_loss)
    replay_gradient = parameter.grad.detach()
    denominator = monolithic_gradient.abs().clamp_min(1e-12)
    relative = float((replay_gradient - monolithic_gradient).abs() / denominator)
    passed = (
        max(row["max_absolute_error"] for row in parity) <= 1e-5
        and rng_independent
        and input_gradient["finite"]
        and input_gradient["norm"] > 0
        and input_gradient["generator_parameter_gradients"] == 0
        and input_gradient["clip_parameter_gradients"] == 0
        and relative <= 1e-4
    )
    return {
        "status": "passed" if passed else "failed",
        "backend_policy": policy,
        "backward_backend_exception": [
            "upsample_bicubic2d_aa_backward_cuda",
            "memory_efficient_attention_backward_cuda",
        ],
        "parity": parity,
        "rng_independent": rng_independent,
        "input_gradient": input_gradient,
        "replay": {
            "monolithic_gradient": float(monolithic_gradient),
            "replay_gradient": float(replay_gradient),
            "relative_error": relative,
            "feature_forwards": replay.feature_forwards,
        },
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
    }
