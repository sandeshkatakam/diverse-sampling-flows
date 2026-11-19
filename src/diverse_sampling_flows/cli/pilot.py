"""Run the bounded 300-update pilot recipe."""

from __future__ import annotations

import argparse
from pathlib import Path

from diverse_sampling_flows.cli.train import main as train_main
from diverse_sampling_flows.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/main.yaml"))
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--run-id")
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("overrides", nargs="*")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config, args.overrides)
    forwarded = [
        "--config",
        str(args.config),
        "--method",
        "skew_trained",
        "--device",
        args.device,
        "--output-root",
        str(args.output_root),
        "--updates",
        str(config.training.pilot_updates),
    ]
    if args.run_id:
        forwarded.extend(["--run-id", args.run_id])
    if args.toy:
        forwarded.append("--toy")
    forwarded.extend(args.overrides)
    return train_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
