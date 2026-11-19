"""Atomic, complete checkpoints for exact training resumption."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from diverse_sampling_flows.utils.seeding import SeedStreams


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is missing required resume state."""


REQUIRED_CHECKPOINT_KEYS = {
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


def atomic_torch_save(payload: Any, path: Path) -> None:
    """Write via a same-directory temporary and atomically replace ``path``."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(
    path: Path,
    *,
    model_state: dict[str, Any],
    ema_state: dict[str, Any],
    optimizer_state: dict[str, Any],
    scheduler_state: dict[str, Any],
    global_update: int,
    seed_streams: SeedStreams,
    best_validation: dict[str, Any] | None,
    call_counters: dict[str, int],
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "format_version": 1,
        "model": model_state,
        "ema": ema_state,
        "optimizer": optimizer_state,
        "scheduler": scheduler_state,
        "global_update": int(global_update),
        "seed_streams": seed_streams.state_dict(),
        "best_validation": best_validation,
        "call_counters": dict(call_counters),
        "extra": dict(extra or {}),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else [],
    }
    atomic_torch_save(payload, path)


def load_checkpoint(
    path: Path, *, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise CheckpointError("checkpoint payload must be a mapping")
    missing = REQUIRED_CHECKPOINT_KEYS.difference(payload)
    if missing:
        raise CheckpointError(f"checkpoint missing required keys: {sorted(missing)}")
    return payload

