"""Evaluate one method and retain every raw set member."""

from __future__ import annotations

import argparse
from pathlib import Path

from diverse_sampling_flows.cli.common import (
    MANDATORY_METHODS,
    create_run_directory,
    load_cli_config,
    validate_checkpoint_argument,
    write_json,
    write_raw_records,
)
from diverse_sampling_flows.config import save_resolved_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/main.yaml"))
    parser.add_argument("--method", choices=tuple(MANDATORY_METHODS), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--sets", type=int)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--run-id")
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("overrides", nargs="*")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.toy:
        validate_checkpoint_argument(args.method, args.checkpoint)
    config = load_cli_config(args.config, args.overrides)
    config.experiment.method = args.method
    if not args.toy and not args.device.startswith("cuda:"):
        raise ValueError("real frozen-generator evaluation requires CUDA")
    run = create_run_directory(args.output_root, "evaluate", run_id=args.run_id)
    save_resolved_config(config, run / "resolved_config.yaml")
    sets = int(args.sets or config.evaluation.test_sets)
    if args.toy:
        rows = [
            {
                "method": args.method,
                "prompt_id": str(config.experiment.prompt_id),
                "set_id": set_id,
                "slot": slot,
                "score": -float(set_id) / 10,
                "slot_score": -float(set_id + slot) / 10,
                "status": "complete",
            }
            for set_id in range(sets)
            for slot in range(int(config.experiment.num_samples))
        ]
    else:
        from diverse_sampling_flows.cli.evaluation_runtime import evaluate_real_method

        rows = evaluate_real_method(
            config,
            method=args.method,
            checkpoint=args.checkpoint,
            device=args.device,
            sets=sets,
            output_directory=run,
        )
    write_raw_records(
        run / "raw.jsonl", rows, num_slots=int(config.experiment.num_samples)
    )
    write_json(
        run / "evaluation_summary.json",
        {"status": "complete", "rows": len(rows)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
