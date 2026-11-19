import json

import pytest

from diverse_sampling_flows.cli.aggregate import (
    build_complete_method_table,
    hierarchical_bootstrap_summary,
)


def test_hierarchical_summary_averages_sets_within_prompt_first() -> None:
    rows = [
        {"method": "iid", "prompt_id": "a", "set_id": 0, "score": 0.0},
        {"method": "iid", "prompt_id": "a", "set_id": 1, "score": 2.0},
        {"method": "iid", "prompt_id": "b", "set_id": 0, "score": 5.0},
    ]

    summary = hierarchical_bootstrap_summary(rows, replicates=32, seed=7)

    assert summary["iid"]["mean"] == pytest.approx(3.0)
    assert summary["iid"]["prompt_means"] == {"a": 1.0, "b": 5.0}
    assert summary["iid"]["set_count"] == 3
    assert len(summary["iid"]["bootstrap_means"]) == 32


def test_complete_table_marks_missing_methods_not_run() -> None:
    summaries = {"iid": {"mean": -1.0}}

    table = build_complete_method_table(summaries)

    by_method = {row["method"]: row for row in table}
    assert by_method["iid"] == {"method": "iid", "status": "complete", "mean": -1.0}
    assert by_method["skew_trained"]["status"] == "not_run"
    json.dumps(table)
