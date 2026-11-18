"""Strict, composable experiment configuration."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException


class ConfigError(ValueError):
    """Raised when an experiment configuration violates its schema."""


@dataclass
class ExperimentConfig:
    name: str = "sana_prompt_specific_k4"
    method: str = "skew_trained"
    prompt_id: str = "golden_retriever"
    prompt: str = "A photograph of a golden retriever."
    training_seed: int = 17
    num_samples: int = 4
    checkpoint_times: list[float] = field(default_factory=lambda: [0.0, 1.0, 2.0, 3.0])


@dataclass
class ModelConfig:
    model_id: str = "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers"
    revision: str = "aa76e7f4f4928f378716b6716a2130fba3caf5b1"
    transformer_revision: str = "aa76e7f4f4928f378716b6716a2130fba3caf5b1"
    vae_revision: str = "aa76e7f4f4928f378716b6716a2130fba3caf5b1"
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 1
    max_timesteps: float = 1.57080
    intermediate_timesteps: list[float] | None = None
    schedule: list[float] = field(default_factory=lambda: [1.57080, 0.0])
    guidance_scale: float = 4.5
    max_sequence_length: int = 300
    standardized_latent_shape: list[int] = field(default_factory=lambda: [32, 32, 32])
    transformer_dtype: str = "bfloat16"
    sampler_arithmetic_dtype: str = "float32"
    decoder_dtype: str = "bfloat16"
    sigma_data: float = 0.5
    vae_scaling_factor: float = 0.41407
    attention_backend: str = "torch_sdpa"
    deterministic: bool = True


@dataclass
class PriorConfig:
    mask_groups: int = 2
    ridges_per_pair: int = 1
    solver: str = "rk4"
    steps_per_interval: int = 8
    dtype: str = "float32"
    c_max: float = 0.7853981633974483
    a_max: float = 0.25
    b_max: float = 1.0
    e_max: float = 2.0
    mask_seed: int = 3101
    permutation_seed: int = 3102


@dataclass
class ConditionerConfig:
    architecture: str = "resnet"
    width: int = 64
    residual_blocks: int = 4
    group_norm_groups: int = 8
    group_norm_epsilon: float = 1e-5
    mlp_hidden_layers: int = 3
    mlp_hidden_width: int = 128
    patch_size: int = 2
    transformer_width: int = 128
    transformer_blocks: int = 4
    transformer_heads: int = 4
    transformer_mlp_ratio: float = 4.0
    dropout: float = 0.0


@dataclass
class ObjectiveConfig:
    clip_model_id: str = "openai/clip-vit-large-patch14"
    clip_revision: str = "32bd64288804d66eefd0ccbe215aa642df71cc41"
    dino_model_id: str = "facebook/dinov2-base"
    dino_revision: str = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
    siglip_model_id: str = "google/siglip-base-patch16-224"
    siglip_revision: str = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
    epsilon: float = 1e-4
    minimum_bandwidth_sq: float = 1e-4
    bandwidth_file: str = "artifacts/bandwidths/{prompt_id}.json"
    feature_dtype: str = "float32"
    kernel_dtype: str = "float64"


@dataclass
class TrainingConfig:
    updates: int = 1000
    pilot_updates: int = 300
    effective_sets_per_update: int = 4
    microbatch_sets: int = 1
    gradient_mode: str = "feature_replay"
    lr: float = 1e-4
    min_lr: float = 1e-5
    betas: list[float] = field(default_factory=lambda: [0.9, 0.99])
    optimizer_epsilon: float = 1e-8
    weight_decay: float = 1e-4
    warmup_updates: int = 50
    ema_decay: float = 0.995
    gradient_clip_norm: float = 1.0
    validate_every: int = 100
    training_initializations: list[int] = field(default_factory=lambda: [17, 18, 19])
    max_consecutive_nonfinite: int = 2


@dataclass
class EvaluationConfig:
    calibration_images: int = 512
    validation_sets: int = 64
    refinement_validation_sets: int = 256
    test_sets: int = 256
    iid_reference_images: int = 2048
    prior_diagnostic_rows: int = 8192
    refinement_steps_per_interval: list[int] = field(
        default_factory=lambda: [8, 16, 32]
    )
    raw_slot_metrics: bool = True
    refinement_latent_tolerance: float = 1e-3
    refinement_clip_tolerance: float = 0.02
    projection_dimensions: int = 64
    bootstrap_replicates: int = 2000


@dataclass
class ExecutionConfig:
    backend: str = "direct"
    available_gpu_ids: list[int] = field(default_factory=lambda: list(range(8)))
    selected_gpu_id: int = 0
    device: str = "cuda:0"
    gpus_per_run: int = 1
    max_parallel_runs: int = 1
    allow_cpu_mathematical_tests: bool = True
    require_cuda_for_model_runs: bool = True
    validate_during_implementation: bool = True
    run_pilot_after_validation: bool = True
    launch_full_sweep_automatically: bool = False


@dataclass
class PathsConfig:
    output_root: str = "outputs"
    cache_root: str = "cache/huggingface"
    artifact_root: str = "artifacts"


@dataclass
class RootConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    prior: PriorConfig = field(default_factory=PriorConfig)
    conditioner: ConditionerConfig = field(default_factory=ConditionerConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def _load_with_includes(path: Path, seen: set[Path]) -> DictConfig:
    resolved_path = path.expanduser().resolve()
    if resolved_path in seen:
        raise ConfigError(f"cyclic config include: {resolved_path}")
    if not resolved_path.is_file():
        raise ConfigError(f"configuration file does not exist: {resolved_path}")
    seen.add(resolved_path)
    loaded = OmegaConf.load(resolved_path)
    includes = list(loaded.pop("includes", []))
    merged = OmegaConf.create()
    for include in includes:
        merged = OmegaConf.merge(
            merged, _load_with_includes(resolved_path.parent / str(include), seen)
        )
    seen.remove(resolved_path)
    return OmegaConf.merge(merged, loaded)


def load_config(path: Path, overrides: Sequence[str] = ()) -> DictConfig:
    """Load relative includes, merge in order, then apply strict CLI overrides."""
    try:
        schema = OmegaConf.structured(RootConfig)
        OmegaConf.set_struct(schema, True)
        merged = OmegaConf.merge(schema, _load_with_includes(Path(path), set()))
        if overrides:
            merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(list(overrides)))
        OmegaConf.resolve(merged)
        return merged
    except (OmegaConfBaseException, TypeError, ValueError) as error:
        if isinstance(error, ConfigError):
            raise
        raise ConfigError(str(error)) from error


_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")


def validate_executable_config(config: DictConfig) -> None:
    """Reject mutable or missing model revisions before downloads/model runs."""
    for field_name in ("revision", "transformer_revision", "vae_revision"):
        value = str(config.model[field_name])
        if not _IMMUTABLE_REVISION.fullmatch(value):
            raise ConfigError(
                f"model.{field_name} must be an immutable 40-character "
                f"commit SHA; got {value!r}"
            )
    for field_name in ("clip_revision", "dino_revision", "siglip_revision"):
        value = str(config.objective[field_name])
        if not _IMMUTABLE_REVISION.fullmatch(value):
            raise ConfigError(
                f"objective.{field_name} must be an immutable 40-character "
                f"commit SHA; got {value!r}"
            )


def save_resolved_config(config: DictConfig, path: Path) -> None:
    """Atomically persist the fully merged configuration as plain YAML."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    OmegaConf.save(config=config, f=temporary, resolve=True)
    temporary.replace(destination)
