"""Real multi-encoder evaluation for one configured method."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torchvision.utils import save_image

from diverse_sampling_flows.cli.common import write_json
from diverse_sampling_flows.cli.experiment_runtime import (
    apply_checkpoint,
    build_trainable_method,
    sample_method_latents,
)
from diverse_sampling_flows.evaluation.diversity import pairwise_cosine_diversity
from diverse_sampling_flows.evaluation.marginals import PriorMarginalAccumulator
from diverse_sampling_flows.generators.sana_sprint import SanaSprintAdapter
from diverse_sampling_flows.objectives.clip_logdet import rbf_logdet_score
from diverse_sampling_flows.objectives.encoders import (
    CLIPImageEncoder,
    DINOImageEncoder,
    SigLIPPromptEncoder,
)
from diverse_sampling_flows.solvers.rk4 import integrate_rk4
from diverse_sampling_flows.utils.seeding import SeedStreams


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def finalize_marginals_or_unavailable(
    accumulator: PriorMarginalAccumulator,
) -> dict[str, Any]:
    try:
        return {"status": "complete", "diagnostics": accumulator.finalize()}
    except ValueError as error:
        return {"status": "unavailable", "reason": str(error)}


def _load_bandwidth(config: DictConfig) -> float:
    import json

    path = Path(
        str(config.objective.bandwidth_file).format(
            prompt_id=str(config.experiment.prompt_id)
        )
    )
    if not path.is_file():
        raise ValueError(f"calibrated bandwidth is missing: {path}")
    return float(json.loads(path.read_text())["bandwidth_sq"])


def evaluate_real_method(
    config: DictConfig,
    *,
    method: str,
    checkpoint: Path | None,
    device: str,
    sets: int,
    output_directory: Path,
    run_refinement: bool = False,
    refinement_sets: int | None = None,
) -> list[dict[str, Any]]:
    streams = SeedStreams(int(config.experiment.training_seed))
    initialization_seed = streams.seed_for("parameter_initialization")
    torch.manual_seed(initialization_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(initialization_seed)
    model = None
    if method not in {"iid", "fixed_rotation", "balanced_gaussian"}:
        model = build_trainable_method(config, method, device)
        if checkpoint is not None:
            apply_checkpoint(model, checkpoint, ema=True, device=device)
        model.eval()
    generator = SanaSprintAdapter.from_pretrained(
        str(config.model.model_id),
        revision=str(config.model.revision),
        prompt=str(config.experiment.prompt),
        device=device,
    )
    common = {
        "cache_dir": str(config.paths.cache_root),
        "local_files_only": True,
    }
    clip = CLIPImageEncoder.from_pretrained(
        str(config.objective.clip_model_id),
        str(config.objective.clip_revision),
        **common,
    ).to(device)
    dino = DINOImageEncoder.from_pretrained(
        str(config.objective.dino_model_id),
        str(config.objective.dino_revision),
        **common,
    ).to(device)
    siglip = SigLIPPromptEncoder.from_pretrained(
        str(config.objective.siglip_model_id),
        str(config.objective.siglip_revision),
        prompt=str(config.experiment.prompt),
        **common,
    ).to(device)
    bandwidth_sq = _load_bandwidth(config)
    slots = int(config.experiment.num_samples)
    dimension = math.prod(config.model.standardized_latent_shape)
    marginals = PriorMarginalAccumulator(
        slots=slots,
        dimension=dimension,
        projection_dimensions=int(config.evaluation.projection_dimensions),
        seed=int(config.experiment.training_seed) + 6101,
    )
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for set_id in range(sets):
            latents = sample_method_latents(
                config,
                method,
                streams,
                batch=1,
                device=device,
                model=model,
                namespace="test",
            )
            marginals.update(latents.cpu())
            images = generator.generate(
                latents.flatten(0, 1),
                prompt=str(config.experiment.prompt),
                differentiable=False,
            )
            clip_features = clip(images).reshape(1, slots, -1)
            dino_features = dino(images).reshape(1, slots, -1)
            alignment = siglip(images).reshape(1, slots)
            clip_score = float(
                rbf_logdet_score(
                    clip_features,
                    bandwidth_sq=bandwidth_sq,
                    epsilon=float(config.objective.epsilon),
                )[0]
            )
            clip_pair = pairwise_cosine_diversity(clip_features)
            dino_pair = pairwise_cosine_diversity(dino_features)
            image_directory = (
                output_directory
                / "images"
                / str(config.experiment.prompt_id)
                / f"set-{set_id:05d}"
            )
            image_directory.mkdir(parents=True, exist_ok=True)
            for slot in range(slots):
                image_path = image_directory / f"slot-{slot}.png"
                save_image(images[slot].cpu(), image_path)
                rows.append(
                    {
                        "method": method,
                        "prompt_id": str(config.experiment.prompt_id),
                        "set_id": set_id,
                        "slot": slot,
                        "score": clip_score,
                        "clip_logdet": clip_score,
                        "clip_pairwise_mean": float(clip_pair.mean[0]),
                        "clip_pairwise_minimum": float(clip_pair.minimum[0]),
                        "dino_pairwise_mean": float(dino_pair.mean[0]),
                        "dino_pairwise_minimum": float(dino_pair.minimum[0]),
                        "siglip_alignment": float(alignment[0, slot]),
                        "image_path": str(image_path.resolve()),
                        "status": "complete",
                    }
                )
    write_json(
        output_directory / "prior_marginals.json",
        _jsonable(finalize_marginals_or_unavailable(marginals)),
    )
    write_json(
        output_directory / "evaluation_provenance.json",
        {
            "bandwidth_sq": bandwidth_sq,
            "seed_manifest": streams.manifest(),
            "raw_slots_retained": True,
            "checkpoint": str(checkpoint) if checkpoint else None,
        },
    )
    if run_refinement:
        if model is None or method not in {"skew_initial", "skew_trained"}:
            raise ValueError("solver refinement requires a skew field method")
        _evaluate_refinement(
            config,
            model,
            generator,
            clip,
            bandwidth_sq,
            device=device,
            sets=int(refinement_sets or config.evaluation.refinement_validation_sets),
            output_directory=output_directory,
        )
    return rows


@torch.no_grad()
def _evaluate_refinement(
    config: DictConfig,
    model: torch.nn.Module,
    generator: SanaSprintAdapter,
    clip: CLIPImageEncoder,
    bandwidth_sq: float,
    *,
    device: str,
    sets: int,
    output_directory: Path,
) -> None:
    started = time.perf_counter()
    steps = [int(value) for value in config.evaluation.refinement_steps_per_interval]
    streams = SeedStreams(int(config.experiment.training_seed))
    slots = int(config.experiment.num_samples)
    dimension = math.prod(config.model.standardized_latent_shape)
    latent_sums = [0.0 for _ in range(len(steps) - 1)]
    score_sums = [0.0 for _ in range(len(steps) - 1)]
    score_totals = [0.0 for _ in steps]
    accumulators = [
        PriorMarginalAccumulator(
            slots=slots,
            dimension=dimension,
            projection_dimensions=int(config.evaluation.projection_dimensions),
            seed=int(config.experiment.training_seed) + 6201,
        )
        for _ in steps
    ]
    for _ in range(sets):
        seed = streams.standard_normal(
            "refinement_validation",
            (1, *config.model.standardized_latent_shape),
            device=device,
        )
        solutions = [
            integrate_rk4(
                model,
                seed,
                list(config.experiment.checkpoint_times),
                resolution,
            )
            for resolution in steps
        ]
        scores = []
        for index, solution in enumerate(solutions):
            accumulators[index].update(solution.cpu())
            images = generator.generate(
                solution.flatten(0, 1),
                prompt=str(config.experiment.prompt),
                differentiable=False,
            )
            features = clip(images).reshape(1, slots, -1)
            score = float(
                rbf_logdet_score(
                    features,
                    bandwidth_sq=bandwidth_sq,
                    epsilon=float(config.objective.epsilon),
                )[0]
            )
            scores.append(score)
            score_totals[index] += score
        for index in range(len(steps) - 1):
            difference = (solutions[index] - solutions[index + 1]).flatten(2)
            latent_sums[index] += float(
                (difference.norm(dim=-1) / math.sqrt(dimension)).mean()
            )
            score_sums[index] += abs(scores[index] - scores[index + 1])
    comparisons = [
        {
            "coarse_steps": steps[index],
            "fine_steps": steps[index + 1],
            "mean_normalized_latent_difference": latent_sums[index] / sets,
            "mean_clip_logdet_difference": score_sums[index] / sets,
        }
        for index in range(len(steps) - 1)
    ]
    finest = comparisons[-1]
    converged = (
        finest["mean_normalized_latent_difference"]
        < float(config.evaluation.refinement_latent_tolerance)
        and finest["mean_clip_logdet_difference"]
        < float(config.evaluation.refinement_clip_tolerance)
    )
    write_json(
        output_directory / "refinement.json",
        _jsonable(
            {
                "status": "complete",
                "sets": sets,
                "steps_per_interval": steps,
                "comparisons": comparisons,
                "mean_clip_logdet": [value / sets for value in score_totals],
                "converged": converged,
                "latent_tolerance": float(
                    config.evaluation.refinement_latent_tolerance
                ),
                "clip_tolerance": float(config.evaluation.refinement_clip_tolerance),
                "marginals": [
                    finalize_marginals_or_unavailable(accumulator)
                    for accumulator in accumulators
                ],
                "field_calls": sum(4 * value * (slots - 1) for value in steps)
                * sets,
                "generator_calls": len(steps) * sets,
                "generated_images": len(steps) * sets * slots,
                "wall_seconds": time.perf_counter() - started,
            }
        ),
    )
