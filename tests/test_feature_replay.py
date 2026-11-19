import torch

from diverse_sampling_flows.training.feature_replay import feature_replay_backward


def _feature(value: torch.Tensor) -> torch.Tensor:
    return torch.stack((value.square().sum(-1), value.sin().sum(-1)), dim=-1)


def _set_loss(features: torch.Tensor) -> torch.Tensor:
    centered = features - features.mean(dim=1, keepdim=True)
    return centered.square().sum() + features.prod(dim=1).sum()


def test_feature_replay_matches_monolithic_gradient_and_skips_anchor() -> None:
    seed = torch.tensor([[0.2, -0.7]], dtype=torch.float64)
    parameter = torch.nn.Parameter(torch.tensor(0.4, dtype=torch.float64))
    checkpoints = [seed, seed * parameter, seed * parameter.square()]
    monolithic_features = torch.stack([_feature(value) for value in checkpoints], dim=1)
    monolithic = torch.autograd.grad(
        _set_loss(monolithic_features), parameter, retain_graph=True
    )[0]

    replay = feature_replay_backward(checkpoints, _feature, _set_loss)

    assert replay.replayed_slots == [1, 2]
    assert torch.allclose(parameter.grad, monolithic, atol=1e-12, rtol=1e-12)


def test_feature_replay_applies_accumulation_scale_once() -> None:
    seed = torch.tensor([[0.3, 0.8]], dtype=torch.float64)
    parameter = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float64))
    checkpoints = [seed, seed * parameter]

    feature_replay_backward(checkpoints, _feature, _set_loss, accumulation_scale=0.25)
    scaled_gradient = parameter.grad.clone()
    parameter.grad = None
    fresh_checkpoints = [seed, seed * parameter]
    feature_replay_backward(
        fresh_checkpoints, _feature, _set_loss, accumulation_scale=1.0
    )

    assert torch.allclose(scaled_gradient, parameter.grad * 0.25, atol=1e-12)

