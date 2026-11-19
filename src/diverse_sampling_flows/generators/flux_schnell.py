"""Access-gated deterministic FLUX.1-schnell replication adapter."""

from __future__ import annotations

import torch

from diverse_sampling_flows.generators.base import (
    FrozenGeneratorAdapter,
    configure_deterministic_backend,
)


def pack_flux_latents(unpacked: torch.Tensor) -> torch.Tensor:
    """Pack ``[B,C,H,W]`` standardized FLUX noise into 2x2 patch tokens."""
    if unpacked.ndim != 4 or unpacked.shape[-2] % 2 or unpacked.shape[-1] % 2:
        raise ValueError("FLUX latents must be [B,C,H,W] with even spatial sizes")
    batch, channels, height, width = unpacked.shape
    return (
        unpacked.reshape(batch, channels, height // 2, 2, width // 2, 2)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch, (height // 2) * (width // 2), channels * 4)
    )


class ModelAccessError(RuntimeError):
    pass


class FluxSchnellAdapter(FrozenGeneratorAdapter):
    latent_shape = (16, 128, 128)

    def __init__(self, pipeline, prompt: str, revision: str) -> None:
        super().__init__()
        self.pipeline, self.prompt, self.revision = pipeline, prompt, revision
        self.freeze()

    @classmethod
    def from_pretrained(
        cls, model_id: str, *, revision: str, prompt: str, device="cuda:0"
    ):
        from diffusers import FluxPipeline
        from huggingface_hub.errors import GatedRepoError

        configure_deterministic_backend()
        try:
            pipeline = FluxPipeline.from_pretrained(
                model_id, revision=revision, torch_dtype=torch.bfloat16
            ).to(device)
        except GatedRepoError as error:
            raise ModelAccessError(
                "FLUX.1-schnell access is gated; accept access and authenticate first"
            ) from error
        if bool(getattr(pipeline.scheduler.config, "stochastic_sampling", False)):
            raise ValueError("FLUX scheduler stochastic_sampling must be false")
        return cls(pipeline, prompt, revision)

    def generate(
        self, standardized_noise, *, prompt: str, differentiable: bool = False
    ):
        self.validate_noise(standardized_noise)
        if differentiable:
            raise NotImplementedError(
                "FLUX differentiable replication requires a separately validated "
                "tensor adapter"
            )
        with torch.no_grad():
            return self.pipeline(
                prompt=prompt,
                latents=pack_flux_latents(standardized_noise),
                height=1024,
                width=1024,
                num_inference_steps=4,
                guidance_scale=0.0,
                max_sequence_length=256,
                output_type="pt",
            ).images
