"""Shared contracts for strict, auditable command-line workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from diverse_sampling_flows.config import load_config, validate_executable_config
from diverse_sampling_flows.utils.metadata import make_run_id


@dataclass(frozen=True)
class MethodSpec:
    learned: bool
    requires_checkpoint: bool
    shared_seed: bool


MANDATORY_METHODS: dict[str, MethodSpec] = {
    "iid": MethodSpec(False, False, False),
    "fixed_rotation": MethodSpec(False, False, True),
    "balanced_gaussian": MethodSpec(False, False, True),
    "learned_orthogonal": MethodSpec(True, True, True),
    "learned_conditional_rotation": MethodSpec(True, True, True),
    "skew_initial": MethodSpec(False, False, True),
    "skew_trained": MethodSpec(True, True, True),
    "independent_seed_negative_control": MethodSpec(False, True, False),
}


@dataclass(frozen=True)
class ReportState:
    status: str
    reason: str

    @classmethod
    def unavailable(cls, reason: str) -> ReportState:
        return cls("unavailable", reason)

    @classmethod
    def not_run(cls, reason: str) -> ReportState:
        return cls("not_run", reason)

    def to_dict(self) -> dict[str, str]:
        return {"status": self.status, "reason": self.reason}


def load_cli_config(path: Path, overrides: list[str]) -> DictConfig:
    config = load_config(path, overrides)
    validate_executable_config(config)
    return config


def validate_stage_device(stage: str, device: str) -> None:
    if stage not in {"math", "model"}:
        raise ValueError(f"unknown validation stage: {stage}")
    if stage == "model" and not device.startswith("cuda:"):
        raise ValueError("model validation requires CUDA")


def validate_checkpoint_argument(method: str, checkpoint: Path | None) -> None:
    if method not in MANDATORY_METHODS:
        raise ValueError(f"unknown method: {method}")
    if MANDATORY_METHODS[method].requires_checkpoint and checkpoint is None:
        raise ValueError(f"method {method!r} requires --checkpoint")
    if checkpoint is not None and not checkpoint.is_file():
        raise ValueError(f"checkpoint does not exist: {checkpoint}")


def create_run_directory(
    output_root: Path, kind: str, *, run_id: str | None = None
) -> Path:
    identifier = run_id or make_run_id(kind, 0)
    destination = Path(output_root) / kind / identifier
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def write_json(path: Path, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return path


def write_raw_records(
    path: Path, records: list[dict[str, Any]], *, num_slots: int
) -> Path:
    grouped: dict[tuple[Any, ...], set[int]] = {}
    for row in records:
        if "set_id" not in row or "slot" not in row:
            raise ValueError("raw records require set_id and slot")
        key = (row.get("method"), row.get("prompt_id"), row["set_id"])
        grouped.setdefault(key, set()).add(int(row["slot"]))
    expected = set(range(num_slots))
    if any(slots != expected for slots in grouped.values()):
        raise ValueError("every generated set must retain complete raw slots")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(destination)
    return destination
