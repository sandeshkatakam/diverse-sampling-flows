import torch

from diverse_sampling_flows.priors.masks import (
    build_channel_mask_groups,
    fixed_permutation,
)


def test_two_groups_cover_every_channel_once_as_active() -> None:
    groups = build_channel_mask_groups(32)

    active = [
        channel
        for group in groups
        for pair in group.active_pairs
        for channel in pair
    ]
    assert sorted(active) == list(range(32))
    assert len(active) == len(set(active))
    assert groups[0].context_channels == tuple(range(16, 32))
    assert groups[1].context_channels == tuple(range(16))
    assert groups[0].active_pairs[0] == (0, 1)
    assert groups[1].active_pairs[-1] == (30, 31)


def test_fixed_permutation_is_seeded_and_serializable() -> None:
    first = fixed_permutation(24, seed=91)
    second = fixed_permutation(24, seed=91)
    different = fixed_permutation(24, seed=92)

    assert torch.equal(first, second)
    assert not torch.equal(first, different)
    assert sorted(first.tolist()) == list(range(24))

