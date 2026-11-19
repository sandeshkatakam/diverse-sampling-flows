import math

import torch
from torch import nn

from diverse_sampling_flows.generators.sana_sprint import sana_one_step_latent


class _Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings_contiguous = None

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        encoder_attention_mask,
        guidance,
        timestep,
        return_dict,
        attention_kwargs,
    ):
        self.embeddings_contiguous = encoder_hidden_states.is_contiguous()
        del (
            encoder_attention_mask,
            guidance,
            timestep,
            return_dict,
            attention_kwargs,
        )
        return (0.2 * hidden_states + 0.1,)


def test_one_step_tensor_order_matches_hand_formula() -> None:
    noise = torch.tensor([[[[0.3, -0.4]]]], dtype=torch.float64)
    prompt = torch.zeros(1, 2, 3, dtype=torch.float64)
    mask = torch.ones(1, 2, dtype=torch.long)
    result = sana_one_step_latent(
        noise,
        transformer=_Transformer(),
        prompt_embeddings=prompt,
        prompt_attention_mask=mask,
        guidance_scale=4.5,
        guidance_embed_scale=0.1,
        max_timestep=1.57080,
        sigma_data=0.5,
    )
    timestep = torch.tensor(1.57080, dtype=torch.float64)
    ratio = torch.sin(timestep) / (torch.cos(timestep) + torch.sin(timestep))
    scale = torch.sqrt(ratio.square() + (1 - ratio).square())
    model_input = noise * scale
    raw = 0.2 * model_input + 0.1
    velocity = 0.5 * (
        (1 - 2 * ratio) * model_input
        + (1 - 2 * ratio + 2 * ratio.square()) * raw
    ) / scale
    expected = (
        torch.cos(timestep) * (0.5 * noise)
        - torch.sin(timestep) * velocity
    ) / 0.5

    assert torch.allclose(result, expected, atol=1e-12, rtol=1e-12)


def test_one_step_output_does_not_depend_on_generator_state() -> None:
    noise = torch.randn(2, 1, 2, 2)
    prompt = torch.zeros(2, 1, 3)
    mask = torch.ones(2, 1, dtype=torch.long)
    arguments = dict(
        transformer=_Transformer(),
        prompt_embeddings=prompt,
        prompt_attention_mask=mask,
        guidance_scale=4.5,
        guidance_embed_scale=0.1,
        max_timestep=math.pi / 2,
        sigma_data=0.5,
    )

    first = sana_one_step_latent(
        noise, generator=torch.Generator().manual_seed(1), **arguments
    )
    second = sana_one_step_latent(
        noise, generator=torch.Generator().manual_seed(999), **arguments
    )

    assert torch.equal(first, second)


def test_batched_conditioning_matches_official_contiguous_repeat() -> None:
    transformer = _Transformer()
    sana_one_step_latent(
        torch.randn(2, 1, 2, 2),
        transformer=transformer,
        prompt_embeddings=torch.zeros(1, 3, 4),
        prompt_attention_mask=torch.ones(1, 3),
        guidance_scale=4.5,
        guidance_embed_scale=0.1,
        max_timestep=1.57080,
        sigma_data=0.5,
    )

    assert transformer.embeddings_contiguous

