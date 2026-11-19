"""Immutable run identifiers and environment provenance."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import re
import socket
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from diverse_sampling_flows.utils.logging import redact_secrets


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "run"


def make_run_id(
    experiment_name: str,
    seed: int,
    *,
    now: datetime | None = None,
) -> str:
    timestamp = (now or datetime.now(UTC)).astimezone(UTC)
    return (
        f"{timestamp:%Y%m%dT%H%M%SZ}-{_slug(experiment_name)}-s{int(seed)}-"
        f"{uuid.uuid4().hex[:8]}"
    )


def prepare_run_directory(
    output_root: Path, experiment_name: str, seed: int
) -> tuple[str, Path]:
    run_id = make_run_id(experiment_name, seed)
    run_directory = Path(output_root) / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_id, run_directory


def _git_output(code_root: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=code_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip() if process.returncode == 0 else ""


def _package_versions() -> dict[str, str]:
    names = (
        "torch",
        "torchvision",
        "diffusers",
        "transformers",
        "accelerate",
        "omegaconf",
    )
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _redact_environment_values(text: str, environment: Mapping[str, str]) -> str:
    redacted = text
    for key, value in environment.items():
        redacted_value = redact_secrets({key: value})[key]
        if value and redacted_value == "[REDACTED]":
            redacted = redacted.replace(value, "[REDACTED]")
    return redacted


def capture_runtime_metadata(
    *,
    resolved_config: Mapping[str, Any],
    command: Sequence[str],
    selected_gpu_ids: Sequence[int],
    gpu_records: Sequence[Mapping[str, Any]],
    checkpoint_revisions: Mapping[str, str],
    code_root: Path,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    environment = dict(os.environ if environ is None else environ)
    prompt = str(resolved_config.get("experiment", {}).get("prompt", ""))
    root = Path(code_root).resolve()
    dirty_diff = _redact_environment_values(
        _git_output(root, "diff", "--binary", "HEAD"), environment
    )
    metadata = {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "hostname": socket.gethostname(),
        "process_id": os.getpid(),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "packages": _package_versions(),
        "launch_command": list(command),
        "selected_server_gpu_ids": [int(index) for index in selected_gpu_ids],
        "gpus": [dict(record) for record in gpu_records],
        "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
        "logical_device_mapping": {
            f"cuda:{logical}": int(server)
            for logical, server in enumerate(selected_gpu_ids)
        },
        "checkpoint_revisions": dict(checkpoint_revisions),
        "prompt_sha256": prompt_sha256(prompt),
        "resolved_config": dict(resolved_config),
        "git": {
            "commit": _git_output(root, "rev-parse", "HEAD"),
            "branch": _git_output(root, "branch", "--show-current"),
            "dirty": bool(_git_output(root, "status", "--short")),
            "dirty_diff": dirty_diff,
            "dirty_diff_sha256": hashlib.sha256(dirty_diff.encode()).hexdigest(),
        },
    }
    return redact_secrets(metadata)
