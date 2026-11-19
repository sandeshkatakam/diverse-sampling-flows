"""Pinned tensor-only deterministic Sana-Sprint one-step adapter."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from typing import Any, ClassVar

import torch
from torch import nn

from diverse_sampling_flows.generators.base import (
    FrozenGeneratorAdapter,
    configure_deterministic_backend,
)


def _module_dtype(module: nn.Module, fallback: torch.dtype) -> torch.dtype:
    parameter = next(module.parameters(), None)
    return parameter.dtype if parameter is not None else fallback


def sana_one_step_latent(
    standardized_noise: torch.Tensor,
    *,
    transformer: nn.Module,
    prompt_embeddings: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    guidance_scale: float,
    guidance_embed_scale: float,
    max_timestep: float,
    sigma_data: float,
    attention_kwargs: dict[str, Any] | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return standardized denoised latent; ``generator`` is intentionally unused."""
    del generator
    batch = standardized_noise.shape[0]
    device, arithmetic_dtype = standardized_noise.device, standardized_noise.dtype
    timestep_scalar = torch.as_tensor(
        max_timestep, device=device, dtype=arithmetic_dtype
    )
    timestep = timestep_scalar.expand(batch)
    sample = standardized_noise * sigma_data
    standardized_model_input = sample / sigma_data
    scm_timestep = torch.sin(timestep) / (torch.cos(timestep) + torch.sin(timestep))
    expanded = scm_timestep.view(-1, 1, 1, 1)
    scale = torch.sqrt(expanded.square() + (1 - expanded).square())
    model_input = standardized_model_input * scale
    transformer_dtype = _module_dtype(transformer, arithmetic_dtype)
    embeddings = prompt_embeddings
    mask = prompt_attention_mask
    if embeddings.shape[0] == 1 and batch != 1:
        embeddings = embeddings.repeat(batch, 1, 1)
        mask = mask.repeat(batch, 1)
    guidance = torch.full(
        (batch,),
        guidance_scale * guidance_embed_scale,
        device=device,
        dtype=embeddings.dtype,
    )
    raw_output = transformer(
        model_input.to(transformer_dtype),
        encoder_hidden_states=embeddings.to(device=device, dtype=transformer_dtype),
        encoder_attention_mask=mask.to(device),
        guidance=guidance,
        timestep=scm_timestep,
        return_dict=False,
        attention_kwargs=attention_kwargs,
    )[0]
    velocity = (
        (1 - 2 * expanded) * model_input
        + (1 - 2 * expanded + 2 * expanded.square()) * raw_output.to(arithmetic_dtype)
    ) / scale
    velocity = velocity * sigma_data
    denoised = (
        torch.cos(timestep_scalar) * sample
        - torch.sin(timestep_scalar) * velocity
    )
    return denoised / sigma_data


class SanaSprintAdapter(FrozenGeneratorAdapter):
    SOURCE_SHA256: ClassVar[dict[str, str]] = {
        "pipeline": "c682dcf3f30d07d8bd18d95f30b901782f0248738f0a5cc7b832e2f14349428c",
        "scheduler": "5e3407521c573e7df0788cd68802c7acd5ec0180762296d03326d8cb10b46466",
    }

    def __init__(
        self,
        *,
        transformer: nn.Module,
        vae: nn.Module,
        latent_shape: tuple[int, int, int] = (32, 32, 32),
        sigma_data: float = 0.5,
        vae_scaling_factor: float = 0.41407,
        max_timestep: float = 1.57080,
        guidance_scale: float = 4.5,
        guidance_embed_scale: float = 0.1,
        revision: str | None = None,
    ) -> None:
        super().__init__()
        self.transformer, self.vae = transformer, vae
        self.latent_shape = tuple(latent_shape)
        self.sigma_data = float(sigma_data)
        self.vae_scaling_factor = float(vae_scaling_factor)
        self.max_timestep = float(max_timestep)
        self.guidance_scale = float(guidance_scale)
        self.guidance_embed_scale = float(guidance_embed_scale)
        self.revision = revision
        self.conditioning: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.last_standardized_latent: torch.Tensor | None = None
        self.freeze()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        revision: str,
        prompt: str,
        device: torch.device | str = "cuda:0",
        transformer_dtype: torch.dtype = torch.bfloat16,
    ) -> SanaSprintAdapter:
        from diffusers import SanaSprintPipeline

        configure_deterministic_backend()
        pipeline = SanaSprintPipeline.from_pretrained(
            model_id, revision=revision, torch_dtype=transformer_dtype
        ).to(device)
        if float(pipeline.scheduler.config.sigma_data) != 0.5:
            raise ValueError("pinned Sana scheduler sigma_data is not 0.5")
        if abs(float(pipeline.vae.config.scaling_factor) - 0.41407) > 1e-6:
            raise ValueError("pinned Sana VAE scaling factor is not 0.41407")
        with torch.no_grad():
            embeddings, mask = pipeline.encode_prompt(
                prompt,
                num_images_per_prompt=1,
                device=torch.device(device),
                clean_caption=False,
                max_sequence_length=300,
                complex_human_instruction=None,
            )
        adapter = cls(
            transformer=pipeline.transformer,
            vae=pipeline.vae,
            sigma_data=0.5,
            vae_scaling_factor=float(pipeline.vae.config.scaling_factor),
            max_timestep=1.57080,
            guidance_scale=4.5,
            guidance_embed_scale=float(pipeline.transformer.config.guidance_embeds_scale),
            revision=revision,
        )
        adapter.set_conditioning(prompt, embeddings, mask)
        return adapter

    def set_conditioning(
        self, prompt: str, embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> None:
        self.conditioning[prompt] = (embeddings.detach(), attention_mask.detach())

    def cache_conditioning(
        self,
        prompt: str,
        encode: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prompt not in self.conditioning:
            embeddings, mask = encode()
            self.set_conditioning(prompt, embeddings, mask)
        return self.conditioning[prompt]

    def generate(
        self,
        standardized_noise: torch.Tensor,
        *,
        prompt: str,
        differentiable: bool = False,
    ) -> torch.Tensor:
        self.validate_noise(standardized_noise)
        if prompt not in self.conditioning:
            raise KeyError(f"prompt conditioning is not cached: {prompt!r}")
        context = nullcontext() if differentiable else torch.no_grad()
        embeddings, mask = self.conditioning[prompt]
        with context:
            standardized_latent = sana_one_step_latent(
                standardized_noise,
                transformer=self.transformer,
                prompt_embeddings=embeddings,
                prompt_attention_mask=mask,
                guidance_scale=self.guidance_scale,
                guidance_embed_scale=self.guidance_embed_scale,
                max_timestep=self.max_timestep,
                sigma_data=self.sigma_data,
            )
            self.last_standardized_latent = standardized_latent
            decoder_dtype = _module_dtype(self.vae, standardized_latent.dtype)
            decoder_latent = standardized_latent.to(decoder_dtype)
            decoded = self.vae.decode(
                decoder_latent / self.vae_scaling_factor,
                return_dict=False,
            )[0]
            # Diffusers denormalizes in decoder dtype before returning a tensor.
            return (decoded * 0.5 + 0.5).clamp(0, 1).to(torch.float32)

