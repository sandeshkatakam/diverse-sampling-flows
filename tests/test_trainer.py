from pathlib import Path

import pytest
import torch
from torch import nn

from diverse_sampling_flows.training.ema import ExponentialMovingAverage
from diverse_sampling_flows.training.optim import build_adamw, build_cosine_scheduler
from diverse_sampling_flows.training.trainer import NonFiniteTrainingError, Trainer
from diverse_sampling_flows.utils.seeding import SeedStreams


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.norm = nn.LayerNorm(2)

    def forward(self, value):
        return self.norm(self.linear(value))


def test_optimizer_excludes_bias_and_normalization_from_weight_decay() -> None:
    model = _ToyModel()
    optimizer = build_adamw(model, lr=1e-3, weight_decay=0.1)
    decay_by_parameter = {
        id(parameter): group["weight_decay"]
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    assert decay_by_parameter[id(model.linear.weight)] == 0.1
    assert decay_by_parameter[id(model.linear.bias)] == 0.0
    assert decay_by_parameter[id(model.norm.weight)] == 0.0
    assert decay_by_parameter[id(model.norm.bias)] == 0.0


def test_cosine_scheduler_warms_up_and_reaches_declared_minimum() -> None:
    model = nn.Linear(1, 1)
    optimizer = build_adamw(model, lr=1e-3, weight_decay=0.0)
    scheduler = build_cosine_scheduler(
        optimizer, warmup_updates=2, total_updates=10, minimum_lr=1e-4
    )
    rates = []
    for _ in range(10):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])

    assert rates[0] < rates[1]
    assert rates[-1] == pytest.approx(1e-4)


def test_ema_updates_and_context_restores_live_parameters() -> None:
    model = nn.Linear(1, 1, bias=False)
    ema = ExponentialMovingAverage(model, decay=0.5)
    original = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(2)
    ema.update(model)
    live = model.weight.detach().clone()

    with ema.average_parameters(model):
        assert torch.allclose(model.weight, original + 1)
    assert torch.equal(model.weight, live)


def _make_trainer(path: Path, seed: int = 31) -> Trainer:
    torch.manual_seed(99)
    model = nn.Linear(2, 1)
    optimizer = build_adamw(model, lr=1e-2, weight_decay=0.0)
    scheduler = build_cosine_scheduler(
        optimizer, warmup_updates=1, total_updates=4, minimum_lr=1e-3
    )
    return Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ExponentialMovingAverage(model, decay=0.9),
        seed_streams=SeedStreams(seed),
        output_directory=path,
        gradient_clip_norm=1.0,
        max_consecutive_nonfinite=2,
    )


def _update(trainer: Trainer) -> torch.Tensor:
    inputs = trainer.seed_streams.standard_normal("training", (4, 2))
    return trainer.model(inputs).square().mean()


def test_resume_matches_uninterrupted_optimizer_scheduler_and_seed_progression(
    tmp_path: Path,
) -> None:
    uninterrupted = _make_trainer(tmp_path / "uninterrupted")
    uninterrupted.train(_update, updates=2)

    first_half = _make_trainer(tmp_path / "resumed")
    first_half.train(_update, updates=1)
    checkpoint = first_half.save("latest.pt")
    resumed = _make_trainer(tmp_path / "resumed")
    resumed.resume(checkpoint)
    resumed.train(_update, updates=2)

    for expected, actual in zip(
        uninterrupted.model.parameters(), resumed.model.parameters(), strict=True
    ):
        assert torch.equal(expected, actual)
    assert uninterrupted.global_update == resumed.global_update == 2
    assert torch.equal(
        uninterrupted.seed_streams.standard_normal("training", (3,)),
        resumed.seed_streams.standard_normal("training", (3,)),
    )


def test_validation_selection_rejects_ineligible_and_never_uses_test_scores(
    tmp_path: Path,
) -> None:
    trainer = _make_trainer(tmp_path)

    assert not trainer.consider_validation(score=10.0, eligible=False)
    assert trainer.consider_validation(score=-0.5, eligible=True)
    assert not trainer.consider_validation(score=-0.7, eligible=True)
    assert trainer.best_validation["score"] == -0.5
    assert not hasattr(trainer, "consider_test")


def test_repeated_nonfinite_loss_stops_and_records_failure(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path)

    with pytest.raises(NonFiniteTrainingError):
        trainer.train(
            lambda _: torch.tensor(float("nan"), requires_grad=True), updates=3
        )

    assert trainer.failure_records[-1]["reason"] == "nonfinite_loss"
    assert trainer.failure_records[-1]["consecutive"] == 2
