import torch

from diverse_sampling_flows.evaluation.diversity import (
    pairwise_cosine_diversity,
    summarize_prompt_alignment,
)


def test_pairwise_cosine_metrics_use_six_unique_pairs_for_k4() -> None:
    features = torch.tensor(
        [
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
            ]
        ]
    )
    metrics = pairwise_cosine_diversity(features)

    assert metrics.distances.shape == (1, 6)
    assert metrics.mean.item() == torch.tensor(4 / 3).item()
    assert metrics.minimum.item() == 1.0


def test_prompt_alignment_keeps_raw_slots_and_lower_tail() -> None:
    scores = torch.tensor([[0.2, 0.4, 0.6, 0.8], [0.1, 0.3, 0.5, 0.7]])
    summary = summarize_prompt_alignment(scores, lower_quantile=0.25)

    assert torch.allclose(
        summary.per_slot_mean, torch.tensor([0.15, 0.35, 0.55, 0.75])
    )
    assert torch.allclose(summary.set_mean, torch.tensor([0.5, 0.4]))
    assert summary.lower_tail.item() == torch.quantile(scores.flatten(), 0.25).item()
