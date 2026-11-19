"""Train or resume one learned preserving prior method."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from diverse_sampling_flows.cli.common import (
    MANDATORY_METHODS,
    create_run_directory,
    load_cli_config,
    write_json,
)
from diverse_sampling_flows.config import save_resolved_config
from diverse_sampling_flows.training.ema import ExponentialMovingAverage
from diverse_sampling_flows.training.optim import build_cosine_scheduler
from diverse_sampling_flows.utils.checkpointing import save_checkpoint
from diverse_sampling_flows.utils.seeding import SeedStreams


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/main.yaml"))
    parser.add_argument("--method", choices=tuple(MANDATORY_METHODS), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--run-id")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("overrides", nargs="*")
    return parser


def _toy_train(run: Path, updates: int, seed: int) -> None:
    torch.manual_seed(seed)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = build_cosine_scheduler(
        optimizer, warmup_updates=0, total_updates=updates, minimum_lr=1e-4
    )
    ema = ExponentialMovingAverage(model, decay=0.9)
    streams = SeedStreams(seed)
    losses = []
    for _ in range(updates):
        optimizer.zero_grad(set_to_none=True)
        sample = streams.standard_normal("training", (4, 2))
        loss = model(sample).square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema.update(model)
        losses.append(float(loss))
    save_checkpoint(
        run / "latest.pt",
        model_state=model.state_dict(),
        ema_state=ema.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(),
        global_update=updates,
        seed_streams=streams,
        best_validation=None,
        call_counters={"generator": 0, "field": updates, "feature": 0},
        extra={"toy": True, "losses": losses},
    )
    write_json(run / "training_summary.json", {"status": "complete", "losses": losses})


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_cli_config(args.config, args.overrides)
    config.experiment.method = args.method
    if not MANDATORY_METHODS[args.method].learned:
        raise ValueError(f"method {args.method!r} has no trainable parameters")
    if not args.toy and not args.device.startswith("cuda:"):
        raise ValueError("real frozen-generator training requires CUDA")
    run = create_run_directory(args.output_root, "train", run_id=args.run_id)
    save_resolved_config(config, run / "resolved_config.yaml")
    updates = int(args.updates or config.training.updates)
    if args.toy:
        _toy_train(run, updates, int(config.experiment.training_seed))
    else:
        from diverse_sampling_flows.cli.training_runtime import train_real_method

        train_real_method(
            config,
            method=args.method,
            device=args.device,
            output_directory=run,
            updates=updates,
            resume=args.resume,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
