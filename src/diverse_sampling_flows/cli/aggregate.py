"""Deterministic aggregation from raw per-set records."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image

from diverse_sampling_flows.cli.common import MANDATORY_METHODS, write_json


def hierarchical_bootstrap_summary(
    rows: list[dict[str, Any]], *, replicates: int, seed: int
) -> dict[str, dict[str, Any]]:
    set_scores: dict[tuple[str, str, int], float] = {}
    for row in rows:
        key = (str(row["method"]), str(row["prompt_id"]), int(row["set_id"]))
        score = float(row["score"])
        if key in set_scores and set_scores[key] != score:
            raise ValueError(f"inconsistent duplicated set score for {key}")
        set_scores[key] = score
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (method, prompt, _), score in set_scores.items():
        grouped[method][prompt].append(score)
    rng = random.Random(seed)
    result: dict[str, dict[str, Any]] = {}
    for method, prompts in sorted(grouped.items()):
        prompt_means = {
            prompt: statistics.fmean(values)
            for prompt, values in sorted(prompts.items())
        }
        prompt_names = list(prompt_means)
        bootstrap: list[float] = []
        for _ in range(replicates):
            sampled_prompt_means = []
            for _ in prompt_names:
                prompt = rng.choice(prompt_names)
                values = prompts[prompt]
                sampled = [rng.choice(values) for _ in values]
                sampled_prompt_means.append(statistics.fmean(sampled))
            bootstrap.append(statistics.fmean(sampled_prompt_means))
        sorted_bootstrap = sorted(bootstrap)
        result[method] = {
            "mean": statistics.fmean(prompt_means.values()),
            "prompt_means": prompt_means,
            "set_count": sum(len(values) for values in prompts.values()),
            "bootstrap_means": bootstrap,
            "ci_low": sorted_bootstrap[int(0.025 * (replicates - 1))],
            "ci_high": sorted_bootstrap[int(0.975 * (replicates - 1))],
        }
    return result


def build_complete_method_table(
    summaries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    table = []
    for method in MANDATORY_METHODS:
        if method in summaries:
            table.append(
                {"method": method, "status": "complete", **summaries[method]}
            )
        else:
            table.append(
                {
                    "method": method,
                    "status": "not_run",
                    "reason": "no raw records supplied",
                }
            )
    return table


def _read_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    return rows


def _write_artifacts(
    output: Path, rows: list[dict[str, Any]], table: list[dict[str, Any]]
) -> None:
    complete = [row for row in table if row["status"] == "complete"]
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["method", "status", "mean", "ci_low", "ci_high"]
        )
        writer.writeheader()
        for row in table:
            writer.writerow(
                {
                    key: row.get(key)
                    for key in ("method", "status", "mean", "ci_low", "ci_high")
                }
            )
    if complete:
        figure, axis = plt.subplots(figsize=(max(6, len(complete)), 4))
        axis.bar([row["method"] for row in complete], [row["mean"] for row in complete])
        axis.set_ylabel("mean CLIP logdet")
        axis.tick_params(axis="x", rotation=35)
        figure.tight_layout()
        figure.savefig(output.with_name("method_means.png"), dpi=160)
        plt.close(figure)
    grouped_images: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("image_path"):
            grouped_images[(str(row["method"]), str(row["prompt_id"]))].append(row)
    grid_directory = output.parent / "grids"
    grid_directory.mkdir(parents=True, exist_ok=True)
    manifest = []
    for method in MANDATORY_METHODS:
        matches = [key for key in grouped_images if key[0] == method]
        if not matches:
            manifest.append(
                {"method": method, "status": "not_run", "reason": "no images"}
            )
            continue
        for key in sorted(matches):
            selected = sorted(
                grouped_images[key], key=lambda row: (row["set_id"], row["slot"])
            )[:16]
            paths = [Path(row["image_path"]) for row in selected]
            if not all(path.is_file() for path in paths):
                manifest.append(
                    {
                        "method": method,
                        "prompt_id": key[1],
                        "status": "unavailable",
                        "reason": "one or more image files are missing",
                    }
                )
                continue
            images = [
                Image.open(path).convert("RGB").resize((256, 256)) for path in paths
            ]
            columns = 4
            grid = Image.new("RGB", (columns * 256, ((len(images) + 3) // 4) * 256))
            for index, image in enumerate(images):
                grid.paste(image, ((index % columns) * 256, (index // columns) * 256))
            destination = grid_directory / f"{method}-{key[1]}.png"
            grid.save(destination)
            manifest.append(
                {
                    "method": method,
                    "prompt_id": key[1],
                    "status": "complete",
                    "path": str(destination),
                    "images": len(images),
                }
            )
    write_json(output.parent / "grid_manifest.json", manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=9107)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = _read_jsonl(args.input)
    summaries = hierarchical_bootstrap_summary(
        rows,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    table = build_complete_method_table(summaries)
    write_json(args.output, table)
    _write_artifacts(args.output, rows, table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
