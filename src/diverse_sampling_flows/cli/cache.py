"""Cache immutable generator and encoder revisions in the project cache."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from diverse_sampling_flows.cli.common import load_cli_config, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/main.yaml"))
    parser.add_argument("--all-encoders", action="store_true")
    parser.add_argument("overrides", nargs="*")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_cli_config(args.config, args.overrides)
    cache = Path(config.paths.cache_root)
    requested = [
        (config.model.model_id, config.model.revision),
        (config.objective.clip_model_id, config.objective.clip_revision),
    ]
    if args.all_encoders:
        requested.extend(
            [
                (config.objective.dino_model_id, config.objective.dino_revision),
                (config.objective.siglip_model_id, config.objective.siglip_revision),
            ]
        )
    records = []
    for model_id, revision in requested:
        location = snapshot_download(
            str(model_id), revision=str(revision), cache_dir=cache
        )
        records.append(
            {"model_id": str(model_id), "revision": str(revision), "path": location}
        )
    write_json(Path(config.paths.artifact_root) / "cache_manifest.json", records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
