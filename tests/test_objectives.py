import math

import pytest
import torch

from diverse_sampling_flows.objectives.clip_logdet import (
    LogDetNumericalError,
    calibrate_bandwidth,
    normalize_features,
    rbf_logdet_score,
)
from diverse_sampling_flows.objectives.preprocessing import preprocess_images


def test_normalize_features_preserves_set_axes_and_unit_norms() -> None:
    features = torch.randn(3, 4, 7)
    normalized = normalize_features(features)

    assert normalized.shape == (3, 4, 7)
    assert torch.allclose(normalized.norm(dim=-1), torch.ones(3, 4), atol=1e-6)


def test_rbf_logdet_matches_hand_computed_two_slot_kernel() -> None:
    features = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    epsilon = 1e-4
    off_diagonal = math.exp(-2.0 / 2.0)
    normalized_off_diagonal = off_diagonal / (1 + epsilon)
    expected = 0.5 * math.log(1 - normalized_off_diagonal**2)

    score = rbf_logdet_score(features, bandwidth_sq=1.0, epsilon=epsilon)

    assert score.dtype == torch.float64
    assert score.item() == pytest.approx(expected, abs=1e-12)
    assert score.item() <= 0.0


def test_logdet_numerical_failure_contains_record() -> None:
    features = torch.tensor([[[float("nan"), 0.0], [0.0, 1.0]]])

    with pytest.raises(LogDetNumericalError) as caught:
        rbf_logdet_score(features, bandwidth_sq=1.0, epsilon=1e-4)

    assert caught.value.record["error"] == "nonfinite_kernel"
    assert caught.value.record["bandwidth_sq"] == 1.0


def test_bandwidth_calibration_is_seeded_and_saves_distinct_pair_ids() -> None:
    features = normalize_features(
        torch.randn(30, 8, generator=torch.Generator().manual_seed(4))
    )

    first = calibrate_bandwidth(features, seed=17, num_pairs=40)
    second = calibrate_bandwidth(features, seed=17, num_pairs=40)

    assert first.bandwidth_sq == second.bandwidth_sq
    assert torch.equal(first.pair_ids, second.pair_ids)
    assert torch.all(first.pair_ids[:, 0] != first.pair_ids[:, 1])
    assert first.bandwidth_sq >= 1e-4


def test_tensor_preprocessing_is_differentiable_and_center_crops() -> None:
    images = torch.rand(2, 3, 10, 16, requires_grad=True)
    processed = preprocess_images(
        images, size=8, mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)
    )
    processed.sum().backward()

    assert processed.shape == (2, 3, 8, 8)
    assert images.grad is not None and torch.isfinite(images.grad).all()

