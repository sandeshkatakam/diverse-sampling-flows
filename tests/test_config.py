from pathlib import Path

import pytest

from diverse_sampling_flows.config import (
    ConfigError,
    load_config,
    save_resolved_config,
    validate_executable_config,
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_relative_includes_merge_in_order_and_cli_overrides_last(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "modules" / "first.yaml", "training:\n  updates: 100\n")
    _write(tmp_path / "modules" / "second.yaml", "training:\n  updates: 200\n")
    main = _write(
        tmp_path / "main.yaml",
        "includes:\n"
        "  - modules/first.yaml\n"
        "  - modules/second.yaml\n"
        "experiment:\n"
        "  name: test\n",
    )

    config = load_config(main, ["training.updates=300"])

    assert config.training.updates == 300
    assert config.experiment.name == "test"
    assert "includes" not in config


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    main = _write(
        tmp_path / "main.yaml",
        "experiment:\n  name: test\n  misspelled_option: true\n",
    )

    with pytest.raises(ConfigError, match="misspelled_option"):
        load_config(main)


def test_main_config_contains_complete_module_sections() -> None:
    config = load_config(Path("configs/main.yaml"))

    assert set(config.keys()) == {
        "experiment",
        "model",
        "prior",
        "conditioner",
        "objective",
        "training",
        "evaluation",
        "execution",
        "paths",
    }
    assert config.experiment.num_samples == 4
    assert list(config.experiment.checkpoint_times) == [0.0, 1.0, 2.0, 3.0]
    assert list(config.model.standardized_latent_shape) == [32, 32, 32]
    assert config.prior.steps_per_interval == 8
    assert config.objective.epsilon == pytest.approx(1e-4)
    assert config.evaluation.refinement_steps_per_interval == [8, 16, 32]


@pytest.mark.parametrize("field", ["revision", "transformer_revision", "vae_revision"])
def test_executable_model_config_requires_immutable_revisions(
    tmp_path: Path, field: str
) -> None:
    main = _write(
        tmp_path / "main.yaml",
        "experiment:\n  name: test\n"
        "model:\n"
        "  model_id: org/model\n"
        "  revision: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
        "  transformer_revision: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
        "  vae_revision: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n",
    )
    config = load_config(main)
    config.model[field] = "main"

    with pytest.raises(ConfigError, match=field):
        validate_executable_config(config)


def test_resolved_config_serialization_omits_includes(tmp_path: Path) -> None:
    _write(tmp_path / "module.yaml", "training:\n  updates: 41\n")
    main = _write(
        tmp_path / "main.yaml",
        "includes:\n  - module.yaml\nexperiment:\n  name: serialized\n",
    )
    config = load_config(main, ["training.updates=42"])
    output = tmp_path / "run" / "resolved.yaml"

    save_resolved_config(config, output)
    reloaded = load_config(output)

    assert reloaded.training.updates == 42
    assert reloaded.experiment.name == "serialized"
    assert "includes" not in output.read_text(encoding="utf-8")
