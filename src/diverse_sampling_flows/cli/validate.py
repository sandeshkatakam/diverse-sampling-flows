"""Run staged CPU mathematical or selected-GPU model validation."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from diverse_sampling_flows.cli.common import (
    create_run_directory,
    load_cli_config,
    validate_stage_device,
    write_json,
)
from diverse_sampling_flows.config import save_resolved_config

_MATH_TESTS = [
    "tests/test_masked_skew.py",
    "tests/test_conditioners.py",
    "tests/test_rk4.py",
    "tests/test_baselines.py",
    "tests/test_feature_replay.py",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/main.yaml"))
    parser.add_argument("--stage", choices=("math", "model"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--run-id")
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("overrides", nargs="*")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_stage_device(args.stage, args.device)
    config = load_cli_config(args.config, args.overrides)
    run = create_run_directory(
        args.output_root, "validation", run_id=args.run_id
    )
    save_resolved_config(config, run / "resolved_config.yaml")
    started = time.perf_counter()
    if args.toy:
        result = {"status": "passed", "checks": ["toy_contract"]}
    elif args.stage == "math":
        process = subprocess.run(
            [sys.executable, "-m", "pytest", *_MATH_TESTS, "-q"],
            check=False,
            capture_output=True,
            text=True,
        )
        (run / "pytest.stdout.log").write_text(process.stdout)
        (run / "pytest.stderr.log").write_text(process.stderr)
        result = {
            "status": "passed" if process.returncode == 0 else "failed",
            "exit_status": process.returncode,
            "test_files": _MATH_TESTS,
        }
    else:
        from diverse_sampling_flows.cli.model_validation import run_model_validation

        result = run_model_validation(config, device=args.device, output_directory=run)
    result["elapsed_seconds"] = time.perf_counter() - started
    write_json(run / "validation.json", result)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
