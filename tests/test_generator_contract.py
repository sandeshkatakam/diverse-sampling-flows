import pytest
import torch
from torch import nn

from diverse_sampling_flows.generators.base import configure_deterministic_backend
from diverse_sampling_flows.generators.flux_schnell import pack_flux_latents
from diverse_sampling_flows.generators.sana_sprint import SanaSprintAdapter


class _CountingTransformer(nn.Module):
    guidance_embeds_scale = 0.1

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.2))
        self.calls = 0

    def forward(self, hidden_states, **kwargs):
        del kwargs
        self.calls += 1
        return (self.weight * hidden_states,)


class _VAE(nn.Module):
    scaling_factor = 0.41407

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0, dtype=torch.bfloat16))
        self.last_input = None

    def decode(self, latent, return_dict=False):
        del return_dict
        self.last_input = latent
        image = self.weight * latent[:, :3]
        return (image,)


def _adapter() -> SanaSprintAdapter:
    transformer, vae = _CountingTransformer(), _VAE()
    adapter = SanaSprintAdapter(
        transformer=transformer,
        vae=vae,
        latent_shape=(4, 2, 2),
        sigma_data=0.5,
        vae_scaling_factor=0.41407,
        max_timestep=1.57080,
        guidance_scale=4.5,
    )
    adapter.set_conditioning("prompt", torch.zeros(1, 2, 3), torch.ones(1, 2))
    return adapter


def test_adapter_freezes_components_and_validates_standard_noise_shape() -> None:
    adapter = _adapter()

    assert not adapter.transformer.training and not adapter.vae.training
    assert all(not parameter.requires_grad for parameter in adapter.parameters())
    with pytest.raises(ValueError, match="standardized noise"):
        adapter.generate(torch.randn(1, 4, 3, 3), prompt="prompt")


def test_differentiable_branch_preserves_input_gradient_and_frozen_weights() -> None:
    adapter = _adapter()
    noise = torch.randn(1, 4, 2, 2, requires_grad=True)

    image = adapter.generate(noise, prompt="prompt", differentiable=True)
    image.sum().backward()

    assert noise.grad is not None and torch.isfinite(noise.grad).all()
    assert all(parameter.grad is None for parameter in adapter.parameters())
    expected_decode_input = (
        adapter.last_standardized_latent.to(torch.bfloat16) / 0.41407
    )
    assert torch.equal(adapter.vae.last_input, expected_decode_input)
    expected_image = (adapter.vae.weight * expected_decode_input[:, :3])
    expected_image = (expected_image * 0.5 + 0.5).clamp(0, 1).float()
    assert torch.equal(image, expected_image)


def test_inference_disables_autograd_and_reuses_conditioning_cache() -> None:
    adapter = _adapter()
    calls = 0

    def encode():
        nonlocal calls
        calls += 1
        return torch.ones(1, 2, 3), torch.ones(1, 2)

    first = adapter.cache_conditioning("new", encode)
    second = adapter.cache_conditioning("new", encode)
    output = adapter.generate(
        torch.randn(1, 4, 2, 2), prompt="new", differentiable=False
    )

    assert calls == 1
    assert first[0].data_ptr() == second[0].data_ptr()
    assert not output.requires_grad


def test_deterministic_backend_requires_prelaunch_cublas_workspace(
    monkeypatch,
) -> None:
    previous = torch.are_deterministic_algorithms_enabled()
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_deterministic = torch.backends.cudnn.deterministic
    try:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        with pytest.raises(RuntimeError, match="before Python starts"):
            configure_deterministic_backend()

        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        policy = configure_deterministic_backend()

        assert policy["cublas_workspace_config"] == ":4096:8"
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
    finally:
        torch.use_deterministic_algorithms(previous)
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.backends.cudnn.deterministic = previous_deterministic


def test_flux_packing_is_deterministic_and_matches_patch_layout() -> None:
    unpacked = torch.arange(1 * 2 * 4 * 6).reshape(1, 2, 4, 6)

    packed = pack_flux_latents(unpacked)

    expected_first = unpacked[0, :, 0:2, 0:2].reshape(-1)
    assert packed.shape == (1, 6, 8)
    assert torch.equal(packed[0, 0], expected_first)
    assert torch.equal(packed, pack_flux_latents(unpacked.clone()))
