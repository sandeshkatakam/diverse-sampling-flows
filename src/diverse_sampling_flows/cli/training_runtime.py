"""Real frozen-stack feature-replay training used by the train CLI."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from omegaconf import DictConfig

from diverse_sampling_flows.cli.common import write_json
from diverse_sampling_flows.cli.experiment_runtime import (
    build_trainable_method,
    sample_method_latents,
)
from diverse_sampling_flows.generators.base import configure_training_backward_backend
from diverse_sampling_flows.generators.sana_sprint import SanaSprintAdapter
from diverse_sampling_flows.objectives.clip_logdet import (
    calibrate_bandwidth,
    rbf_logdet_score,
)
from diverse_sampling_flows.objectives.encoders import CLIPImageEncoder
from diverse_sampling_flows.training.ema import ExponentialMovingAverage
from diverse_sampling_flows.training.feature_replay import feature_replay_backward
from diverse_sampling_flows.training.optim import build_adamw, build_cosine_scheduler
from diverse_sampling_flows.training.trainer import NonFiniteTrainingError, Trainer
from diverse_sampling_flows.utils.logging import JsonlLogger
from diverse_sampling_flows.utils.seeding import SeedStreams


def _load_stack(config: DictConfig, device: str):
    generator = SanaSprintAdapter.from_pretrained(
        str(config.model.model_id),
        revision=str(config.model.revision),
        prompt=str(config.experiment.prompt),
        device=device,
    )
    encoder = CLIPImageEncoder.from_pretrained(
        str(config.objective.clip_model_id),
        str(config.objective.clip_revision),
        cache_dir=str(config.paths.cache_root),
        local_files_only=True,
    ).to(device)
    return generator, encoder


def _feature_function(generator, encoder, prompt: str):
    def feature(latents: torch.Tensor) -> torch.Tensor:
        images = generator.generate(
            latents, prompt=prompt, differentiable=torch.is_grad_enabled()
        )
        return encoder(images)

    return feature


def _bandwidth_path(config: DictConfig) -> Path:
    return Path(
        str(config.objective.bandwidth_file).format(
            prompt_id=str(config.experiment.prompt_id)
        )
    )


def _load_or_calibrate_bandwidth(
    config: DictConfig,
    generator: SanaSprintAdapter,
    encoder: CLIPImageEncoder,
    *,
    device: str,
) -> float:
    path = _bandwidth_path(config)
    if path.is_file():
        return float(json.loads(path.read_text())["bandwidth_sq"])
    streams = SeedStreams(int(config.experiment.training_seed))
    requested = int(config.evaluation.calibration_images)
    features = []
    feature = _feature_function(generator, encoder, str(config.experiment.prompt))
    with torch.no_grad():
        for start in range(0, requested, 4):
            batch = min(4, requested - start)
            latents = streams.standard_normal(
                "calibration",
                (batch, *config.model.standardized_latent_shape),
                device=device,
            )
            features.append(feature(latents).cpu())
    calibration = calibrate_bandwidth(
        torch.cat(features),
        seed=int(config.experiment.training_seed) + 7001,
        num_pairs=min(4096, requested * (requested - 1)),
        minimum_bandwidth_sq=float(config.objective.minimum_bandwidth_sq),
    )
    record = calibration.to_dict() | {
        "model_id": str(config.objective.clip_model_id),
        "revision": str(config.objective.clip_revision),
        "images": requested,
        "seed_namespace": "calibration",
    }
    write_json(path, record)
    return calibration.bandwidth_sq


@torch.no_grad()
def _validation_score(
    config: DictConfig,
    method: str,
    model: torch.nn.Module,
    generator: SanaSprintAdapter,
    encoder: CLIPImageEncoder,
    bandwidth_sq: float,
    *,
    device: str,
) -> float:
    streams = SeedStreams(int(config.experiment.training_seed))
    scores = []
    feature = _feature_function(generator, encoder, str(config.experiment.prompt))
    for _ in range(int(config.evaluation.validation_sets)):
        latents = sample_method_latents(
            config,
            method,
            streams,
            batch=1,
            device=device,
            model=model,
            namespace="validation",
        )
        flat = latents.flatten(0, 1)
        encoded = feature(flat).reshape(1, int(config.experiment.num_samples), -1)
        scores.append(
            rbf_logdet_score(
                encoded,
                bandwidth_sq=bandwidth_sq,
                epsilon=float(config.objective.epsilon),
            )
        )
    return float(torch.cat(scores).mean())


def train_real_method(
    config: DictConfig,
    *,
    method: str,
    device: str,
    output_directory: Path,
    updates: int,
    resume: Path | None,
) -> None:
    streams = SeedStreams(int(config.experiment.training_seed))
    initialization_seed = streams.seed_for("parameter_initialization")
    torch.manual_seed(initialization_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(initialization_seed)
    model = build_trainable_method(config, method, device)
    optimizer = build_adamw(
        model,
        lr=float(config.training.lr),
        weight_decay=float(config.training.weight_decay),
        betas=tuple(config.training.betas),
        epsilon=float(config.training.optimizer_epsilon),
    )
    scheduler = build_cosine_scheduler(
        optimizer,
        warmup_updates=int(config.training.warmup_updates),
        total_updates=updates,
        minimum_lr=float(config.training.min_lr),
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ExponentialMovingAverage(model, float(config.training.ema_decay)),
        seed_streams=streams,
        output_directory=output_directory,
        gradient_clip_norm=float(config.training.gradient_clip_norm),
        max_consecutive_nonfinite=int(config.training.max_consecutive_nonfinite),
    )
    if resume is not None:
        trainer.resume(resume)
    else:
        trainer.save("initial.pt")
    generator, encoder = _load_stack(config, device)
    configure_training_backward_backend()
    bandwidth_sq = _load_or_calibrate_bandwidth(
        config, generator, encoder, device=device
    )
    feature = _feature_function(generator, encoder, str(config.experiment.prompt))
    effective_sets = int(config.training.effective_sets_per_update)
    microbatch = int(config.training.microbatch_sets)
    if effective_sets % microbatch:
        raise ValueError("effective sets must be divisible by microbatch sets")
    field_calls_per_set = (
        4
        * int(config.prior.steps_per_interval)
        * (len(config.experiment.checkpoint_times) - 1)
        if method == "skew_trained"
        else 1
    )
    nonfinite = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with JsonlLogger(output_directory / "training.jsonl") as logger:
        while trainer.global_update < updates:
            update_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for _ in range(effective_sets // microbatch):
                latent_sets = sample_method_latents(
                    config,
                    method,
                    trainer.seed_streams,
                    batch=microbatch,
                    device=device,
                    model=model,
                    namespace="training",
                )
                checkpoints = [latent_sets[:, 0].detach()] + [
                    latent_sets[:, slot]
                    for slot in range(1, int(config.experiment.num_samples))
                ]

                def set_loss(features: torch.Tensor) -> torch.Tensor:
                    return -rbf_logdet_score(
                        features,
                        bandwidth_sq=bandwidth_sq,
                        epsilon=float(config.objective.epsilon),
                    ).mean()

                replay = feature_replay_backward(
                    checkpoints,
                    feature,
                    set_loss,
                    accumulation_scale=microbatch / effective_sets,
                )
                losses.append(float(replay.loss))
                trainer.call_counters["generator"] += replay.feature_forwards
                trainer.call_counters["feature"] += replay.feature_forwards
                trainer.call_counters["field"] += field_calls_per_set * microbatch
            mean_loss = sum(losses) / len(losses)
            if not math.isfinite(mean_loss):
                nonfinite += 1
                trainer.failure_records.append(
                    {
                        "reason": "nonfinite_loss",
                        "consecutive": nonfinite,
                        "global_update": trainer.global_update,
                    }
                )
                if nonfinite >= trainer.max_consecutive_nonfinite:
                    raise NonFiniteTrainingError("repeated nonfinite feature loss")
                continue
            nonfinite = 0
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), trainer.gradient_clip_norm
            )
            if not torch.isfinite(norm):
                raise NonFiniteTrainingError("nonfinite_gradient_norm")
            optimizer.step()
            scheduler.step()
            trainer.ema.update(model)
            trainer.global_update += 1
            logger.log(
                "update",
                update=trainer.global_update,
                loss=mean_loss,
                gradient_norm=float(norm),
                learning_rate=optimizer.param_groups[0]["lr"],
                peak_gpu_memory=torch.cuda.max_memory_allocated()
                if torch.cuda.is_available()
                else 0,
                wall_seconds=time.perf_counter() - update_started,
            )
            should_validate = (
                trainer.global_update % int(config.training.validate_every) == 0
                or trainer.global_update == updates
            )
            if should_validate:
                validation_started = time.perf_counter()
                with trainer.ema.average_parameters(model):
                    score = _validation_score(
                        config,
                        method,
                        model,
                        generator,
                        encoder,
                        bandwidth_sq,
                        device=device,
                    )
                improved = trainer.consider_validation(
                    score=score, eligible=math.isfinite(score)
                )
                logger.log(
                    "validation",
                    update=trainer.global_update,
                    score=score,
                    eligible=math.isfinite(score),
                    wall_seconds=time.perf_counter() - validation_started,
                )
                if improved:
                    trainer.save("best.pt")
            trainer.save("latest.pt")
    write_json(
        output_directory / "training_summary.json",
        {
            "status": "complete",
            "updates": trainer.global_update,
            "best_validation": trainer.best_validation,
            "call_counters": trainer.call_counters,
            "bandwidth_sq": bandwidth_sq,
        },
    )
