"""Generic finite-safe trainer with exact checkpoint resumption."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn

from diverse_sampling_flows.training.ema import ExponentialMovingAverage
from diverse_sampling_flows.utils.checkpointing import load_checkpoint, save_checkpoint
from diverse_sampling_flows.utils.seeding import SeedStreams


class NonFiniteTrainingError(RuntimeError):
    pass


class Trainer:
    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        ema: ExponentialMovingAverage,
        seed_streams: SeedStreams,
        output_directory: Path,
        gradient_clip_norm: float,
        max_consecutive_nonfinite: int,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ema = ema
        self.seed_streams = seed_streams
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.max_consecutive_nonfinite = int(max_consecutive_nonfinite)
        self.global_update = 0
        self.best_validation: dict[str, Any] | None = None
        self.call_counters = {"generator": 0, "field": 0, "feature": 0}
        self.failure_records: list[dict[str, Any]] = []
        self.gradient_norms: list[float] = []

    def train(
        self,
        update_function: Callable[[Trainer], torch.Tensor],
        *,
        updates: int,
    ) -> None:
        consecutive_nonfinite = 0
        while self.global_update < updates:
            self.optimizer.zero_grad(set_to_none=True)
            loss = update_function(self)
            if not torch.isfinite(loss):
                consecutive_nonfinite += 1
                record = {
                    "reason": "nonfinite_loss",
                    "consecutive": consecutive_nonfinite,
                    "global_update": self.global_update,
                }
                self.failure_records.append(record)
                if consecutive_nonfinite >= self.max_consecutive_nonfinite:
                    raise NonFiniteTrainingError(str(record))
                continue
            consecutive_nonfinite = 0
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip_norm
            )
            self.gradient_norms.append(float(norm))
            if not torch.isfinite(norm):
                raise NonFiniteTrainingError("nonfinite_gradient_norm")
            self.optimizer.step()
            self.scheduler.step()
            self.ema.update(self.model)
            self.global_update += 1

    def consider_validation(self, *, score: float, eligible: bool) -> bool:
        if not eligible:
            return False
        if self.best_validation is None or score > self.best_validation["score"]:
            self.best_validation = {
                "score": float(score),
                "update": self.global_update,
                "eligible": True,
            }
            return True
        return False

    def save(self, filename: str) -> Path:
        destination = self.output_directory / filename
        save_checkpoint(
            destination,
            model_state=self.model.state_dict(),
            ema_state=self.ema.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict(),
            global_update=self.global_update,
            seed_streams=self.seed_streams,
            best_validation=self.best_validation,
            call_counters=self.call_counters,
            extra={"failure_records": self.failure_records},
        )
        return destination

    def resume(self, path: Path) -> None:
        payload = load_checkpoint(path)
        self.model.load_state_dict(payload["model"])
        self.ema.load_state_dict(payload["ema"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.global_update = int(payload["global_update"])
        self.seed_streams.load_state_dict(payload["seed_streams"])
        self.best_validation = payload["best_validation"]
        self.call_counters = dict(payload["call_counters"])
        self.failure_records = list(payload["extra"].get("failure_records", []))
        torch.set_rng_state(payload["torch_rng_state"])
        if payload["cuda_rng_states"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["cuda_rng_states"])
