import json
from pathlib import Path

import pytest

from diverse_sampling_flows.cli import (
    aggregate,
    cache,
    evaluate,
    pilot,
    train,
    validate,
)
from diverse_sampling_flows.cli.common import (
    MANDATORY_METHODS,
    ReportState,
    create_run_directory,
    load_cli_config,
    validate_checkpoint_argument,
    validate_stage_device,
    write_raw_records,
)
from diverse_sampling_flows.config import ConfigError


def test_method_registry_contains_every_prespecified_comparison() -> None:
    assert set(MANDATORY_METHODS) == {
        "iid",
        "fixed_rotation",
        "balanced_gaussian",
        "learned_orthogonal",
        "learned_conditional_rotation",
        "skew_initial",
        "skew_trained",
        "independent_seed_negative_control",
    }
    assert MANDATORY_METHODS["skew_trained"].requires_checkpoint
    assert not MANDATORY_METHODS["iid"].requires_checkpoint


def test_cli_config_is_strict_and_cli_override_is_last() -> None:
    root = Path(__file__).parents[1]
    config = load_cli_config(
        root / "configs/main.yaml", ["experiment.training_seed=29"]
    )
    assert config.experiment.training_seed == 29
    with pytest.raises(ConfigError):
        load_cli_config(root / "configs/main.yaml", ["training.typo=1"])


def test_stage_device_and_checkpoint_contracts() -> None:
    validate_stage_device("math", "cpu")
    validate_stage_device("model", "cuda:0")
    with pytest.raises(ValueError, match="requires CUDA"):
        validate_stage_device("model", "cpu")
    with pytest.raises(ValueError, match="requires --checkpoint"):
        validate_checkpoint_argument("skew_trained", None)
    validate_checkpoint_argument("iid", None)


def test_run_directories_are_isolated_and_raw_slots_are_preserved(tmp_path) -> None:
    first = create_run_directory(tmp_path, "pilot", run_id="same")
    with pytest.raises(FileExistsError):
        create_run_directory(tmp_path, "pilot", run_id="same")

    rows = [
        {"set_id": 0, "slot": slot, "score": float(slot)} for slot in range(4)
    ]
    destination = write_raw_records(first / "raw.jsonl", rows, num_slots=4)
    loaded = [json.loads(line) for line in destination.read_text().splitlines()]
    assert [row["slot"] for row in loaded] == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="complete raw slots"):
        write_raw_records(first / "bad.jsonl", rows[:-1], num_slots=4)


def test_report_states_do_not_fabricate_results() -> None:
    unavailable = ReportState.unavailable("gated checkpoint")
    not_run = ReportState.not_run("full sweep is opt-in")
    assert unavailable.to_dict() == {
        "status": "unavailable",
        "reason": "gated checkpoint",
    }
    assert not_run.to_dict()["status"] == "not_run"


def test_all_workflow_parsers_expose_help() -> None:
    for module in (cache, validate, pilot, train, evaluate, aggregate):
        assert module.build_parser().format_help().startswith("usage:")


def test_toy_cpu_workflows_create_auditable_artifacts(tmp_path) -> None:
    root = Path(__file__).parents[1]
    config = root / "configs/main.yaml"
    assert (
        validate.main(
            [
                "--config",
                str(config),
                "--stage",
                "math",
                "--device",
                "cpu",
                "--output-root",
                str(tmp_path),
                "--run-id",
                "math",
                "--toy",
            ]
        )
        == 0
    )
    assert (
        train.main(
            [
                "--config",
                str(config),
                "--method",
                "skew_trained",
                "--device",
                "cpu",
                "--output-root",
                str(tmp_path),
                "--run-id",
                "train",
                "--updates",
                "2",
                "--toy",
            ]
        )
        == 0
    )
    assert (tmp_path / "train" / "train" / "latest.pt").is_file()
    assert (
        evaluate.main(
            [
                "--config",
                str(config),
                "--method",
                "iid",
                "--device",
                "cpu",
                "--output-root",
                str(tmp_path),
                "--run-id",
                "evaluate",
                "--sets",
                "2",
                "--toy",
            ]
        )
        == 0
    )
    raw = tmp_path / "evaluate" / "evaluate" / "raw.jsonl"
    assert len(raw.read_text().splitlines()) == 8
    assert (
        aggregate.main(
            [
                "--input",
                str(raw),
                "--output",
                str(tmp_path / "table.json"),
                "--bootstrap-replicates",
                "8",
            ]
        )
        == 0
    )
