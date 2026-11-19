from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import torch

from diverse_sampling_flows.utils.logging import JsonlLogger
from diverse_sampling_flows.utils.metadata import (
    capture_runtime_metadata,
    make_run_id,
    prompt_sha256,
)
from diverse_sampling_flows.utils.seeding import SeedStreams


def test_seed_namespaces_are_deterministic_and_disjoint() -> None:
    first = SeedStreams(17)
    second = SeedStreams(17)

    first_training = first.standard_normal("training", (8,), dtype=torch.float64)
    second_training = second.standard_normal("training", (8,), dtype=torch.float64)
    calibration = first.standard_normal("calibration", (8,), dtype=torch.float64)

    assert torch.equal(first_training, second_training)
    assert not torch.equal(first_training, calibration)
    assert first.counters == {"training": 1, "calibration": 1}


def test_parameter_initialization_seed_is_recorded_without_consuming_rows() -> None:
    first = SeedStreams(17)
    second = SeedStreams(17)

    initialization = first.seed_for("parameter_initialization")

    assert initialization == second.seed_for("parameter_initialization")
    assert initialization != first.seed_for("training")
    assert first.counters["parameter_initialization"] == 0
    assert (
        first.manifest()["namespace_seeds"]["parameter_initialization"]
        == initialization
    )


def test_seed_stream_state_round_trip_continues_exact_sequence() -> None:
    uninterrupted = SeedStreams(91)
    _ = uninterrupted.standard_normal("training", (3,))
    state = uninterrupted.state_dict()
    expected_next = uninterrupted.standard_normal("training", (3,))

    resumed = SeedStreams(0)
    resumed.load_state_dict(state)
    actual_next = resumed.standard_normal("training", (3,))

    assert torch.equal(actual_next, expected_next)
    assert resumed.counters["training"] == 2


def test_jsonl_logger_redacts_nested_secrets(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"

    with JsonlLogger(log_path) as logger:
        logger.log(
            "launch",
            token="hf_secret",
            nested={"authorization": "Bearer private", "safe": "visible"},
        )

    event = json.loads(log_path.read_text(encoding="utf-8"))
    assert event["event"] == "launch"
    assert event["token"] == "[REDACTED]"
    assert event["nested"]["authorization"] == "[REDACTED]"
    assert event["nested"]["safe"] == "visible"
    assert "hf_secret" not in log_path.read_text(encoding="utf-8")


def test_run_ids_are_unique_and_path_safe() -> None:
    now = datetime(2026, 11, 18, 12, 30, tzinfo=UTC)

    first = make_run_id("Sana prompt-specific / K=4", 17, now=now)
    second = make_run_id("Sana prompt-specific / K=4", 17, now=now)

    assert first != second
    assert re.fullmatch(
        r"20261118T123000Z-sana-prompt-specific-k-4-s17-[0-9a-f]{8}", first
    )


def test_runtime_metadata_records_exact_launch_and_prompt() -> None:
    prompt = "A photograph of a golden retriever."
    metadata = capture_runtime_metadata(
        resolved_config={"experiment": {"prompt": prompt}},
        command=["bash", "launchers/pilot.sh", "--gpu", "7"],
        selected_gpu_ids=[7],
        gpu_records=[{"index": 7, "uuid": "GPU-test", "name": "NVIDIA A40"}],
        checkpoint_revisions={"generator": "a" * 40},
        code_root=Path.cwd(),
        environ={"CUDA_VISIBLE_DEVICES": "7", "HF_TOKEN": "must-not-appear"},
    )

    assert metadata["launch_command"] == [
        "bash",
        "launchers/pilot.sh",
        "--gpu",
        "7",
    ]
    assert metadata["selected_server_gpu_ids"] == [7]
    assert metadata["gpus"][0]["uuid"] == "GPU-test"
    assert metadata["cuda_visible_devices"] == "7"
    assert metadata["prompt_sha256"] == prompt_sha256(prompt)
    assert metadata["checkpoint_revisions"] == {"generator": "a" * 40}
    assert "must-not-appear" not in json.dumps(metadata)
    assert metadata["git"]["branch"] == "scrape"

