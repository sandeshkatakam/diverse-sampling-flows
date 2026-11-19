from __future__ import annotations

from pathlib import Path

import pytest
import torch

from diverse_sampling_flows.utils.checkpointing import (
    CheckpointError,
    atomic_torch_save,
    load_checkpoint,
    save_checkpoint,
)
from diverse_sampling_flows.utils.seeding import SeedStreams


class _Unserializable:
    def __reduce__(self):
        raise RuntimeError("intentional serialization failure")


def test_atomic_save_replaces_target_and_leaves_no_partial_file(tmp_path: Path) -> None:
    target = tmp_path / "checkpoint.pt"
    torch.save({"value": "old"}, target)

    atomic_torch_save({"value": "new"}, target)

    assert torch.load(target, weights_only=False) == {"value": "new"}
    assert list(tmp_path.glob(".checkpoint.pt.*.tmp")) == []


def test_atomic_save_failure_preserves_previous_target(tmp_path: Path) -> None:
    target = tmp_path / "checkpoint.pt"
    torch.save({"value": "old"}, target)

    with pytest.raises(RuntimeError, match="intentional serialization failure"):
        atomic_torch_save({"bad": _Unserializable()}, target)

    assert torch.load(target, weights_only=False) == {"value": "old"}
    assert list(tmp_path.glob(".checkpoint.pt.*.tmp")) == []


def test_checkpoint_round_trip_contains_complete_resume_state(tmp_path: Path) -> None:
    path = tmp_path / "run" / "latest.pt"
    seeds = SeedStreams(17)
    _ = seeds.standard_normal("training", (2,))

    save_checkpoint(
        path,
        model_state={"weight": torch.tensor([1.0])},
        ema_state={"weight": torch.tensor([0.9])},
        optimizer_state={"state": {1: {"step": 3}}},
        scheduler_state={"last_epoch": 3},
        global_update=3,
        seed_streams=seeds,
        best_validation={"update": 2, "score": -0.4, "eligible": True},
        call_counters={"generator": 28, "field": 192},
        extra={"resolved_config": {"training": {"updates": 1000}}},
    )
    loaded = load_checkpoint(path)

    assert set(loaded) == {
        "format_version",
        "model",
        "ema",
        "optimizer",
        "scheduler",
        "global_update",
        "seed_streams",
        "best_validation",
        "call_counters",
        "extra",
        "torch_rng_state",
        "cuda_rng_states",
    }
    assert loaded["global_update"] == 3
    assert loaded["seed_streams"]["counters"] == {"training": 1}
    assert loaded["call_counters"] == {"generator": 28, "field": 192}


def test_load_checkpoint_rejects_incomplete_payload(tmp_path: Path) -> None:
    path = tmp_path / "broken.pt"
    torch.save({"model": {}}, path)

    with pytest.raises(CheckpointError, match="missing required keys"):
        load_checkpoint(path)
