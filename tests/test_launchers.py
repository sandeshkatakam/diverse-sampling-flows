import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
LAUNCHERS = (
    "setup",
    "validate",
    "pilot",
    "train",
    "evaluate",
    "aggregate",
    "sweep",
)


def _run(name: str, *arguments: str, env=None):
    return subprocess.run(
        ["bash", str(ROOT / "launchers" / f"{name}.sh"), *arguments],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("name", LAUNCHERS)
def test_launchers_have_valid_syntax_and_help(name: str) -> None:
    syntax = subprocess.run(
        ["bash", "-n", str(ROOT / "launchers" / f"{name}.sh")], check=False
    )
    assert syntax.returncode == 0
    result = _run(name, "--help")
    assert result.returncode == 0
    assert "Usage:" in result.stdout


def test_math_validation_dry_run_uses_shared_venv_without_cuda() -> None:
    result = _run(
        "validate", "--stage", "math", "--device", "cpu", "--dry-run"
    )
    assert result.returncode == 0
    assert ".venv/bin/python" in result.stdout
    assert "--stage math --device cpu" in result.stdout
    assert "CUDA_VISIBLE_DEVICES=" not in result.stdout


def test_setup_dry_run_uses_uv_to_avoid_ensurepip_bootstrap() -> None:
    result = _run("setup", "--dry-run")
    assert result.returncode == 0
    assert "uv venv --python python3.11" in result.stdout
    assert "python3.11 -m venv" not in result.stdout


def test_pipeline_dry_run_resolves_the_pilot_checkpoint() -> None:
    result = subprocess.run(
        ["bash", str(ROOT / "launch.sh"), "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--run-id pilot-s17" in result.stdout
    assert f"--checkpoint {ROOT}/outputs/train/pilot-s17/best.pt" in result.stdout
    assert "<run>" not in result.stdout


def test_model_launch_requires_gpu_and_rejects_visibility_widening() -> None:
    missing = _run("validate", "--stage", "model", "--dry-run")
    assert missing.returncode != 0
    assert "--gpu" in missing.stderr

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "4,5"
    rejected = _run(
        "validate",
        "--stage",
        "model",
        "--gpu",
        "3",
        "--dry-run",
        env=environment,
    )
    assert rejected.returncode != 0
    assert "not permitted" in rejected.stderr


def test_model_dry_run_narrows_visibility_and_records_uuid(tmp_path) -> None:
    result = _run(
        "validate",
        "--stage",
        "model",
        "--gpu",
        "0",
        "--output-root",
        str(tmp_path),
        "--dry-run",
    )
    assert result.returncode == 0
    assert "CUDA_VISIBLE_DEVICES=0" in result.stdout
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in result.stdout
    assert "GPU-" in result.stdout


def test_sweep_is_dry_by_default_and_caps_concurrency() -> None:
    planned = _run("sweep", "--gpus", "0,1,2", "--max-concurrency", "3")
    assert planned.returncode == 0
    assert "DRY RUN" in planned.stdout
    assert "gpu=0" in planned.stdout and "gpu=2" in planned.stdout
    assert "outputs/sweep/" in planned.stdout

    rejected = _run("sweep", "--gpus", "0,1,2,3,4,5", "--max-concurrency", "6")
    assert rejected.returncode != 0
    assert "at most 5" in rejected.stderr
