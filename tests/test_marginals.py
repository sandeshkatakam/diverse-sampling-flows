import torch

from diverse_sampling_flows.evaluation.marginals import PriorMarginalAccumulator
from diverse_sampling_flows.evaluation.statistics import (
    calibrate_iid_null,
    fixed_orthonormal_projection,
    unbiased_rbf_mmd,
)


def test_streaming_prior_statistics_keep_raw_slots_separate() -> None:
    generator = torch.Generator().manual_seed(9)
    rows = torch.randn(6000, 4, 6, generator=generator, dtype=torch.float64)
    rows[:, 2, 0] += 0.25
    accumulator = PriorMarginalAccumulator(
        slots=4, dimension=6, projection_dimensions=3, seed=8
    )
    for batch in rows.split(257):
        accumulator.update(batch)
    report = accumulator.finalize()

    assert report["count"] == 6000
    assert report["coordinate_mean"]["per_slot"].shape == (4, 6)
    assert report["radial"]["mean"].shape == (4,)
    assert report["projected"]["covariance"].shape == (4, 3, 3)
    assert report["coordinate_mean"]["per_slot"][2, 0] > 0.2
    assert report["coordinate_mean"]["per_slot"][0, 0].abs() < 0.05


def test_fixed_projection_is_seeded_and_orthonormal() -> None:
    first = fixed_orthonormal_projection(12, 5, seed=77, dtype=torch.float64)
    second = fixed_orthonormal_projection(12, 5, seed=77, dtype=torch.float64)

    assert torch.equal(first, second)
    assert torch.allclose(
        first.T @ first, torch.eye(5, dtype=torch.float64), atol=1e-12
    )


def test_unbiased_mmd_retains_negative_estimate() -> None:
    values = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float64)
    estimate = unbiased_rbf_mmd(values, values.clone(), bandwidth_sq=1.0)

    assert estimate.item() < 0.0


def test_iid_null_calibration_reports_maximum_statistic_alarm() -> None:
    calibration = calibrate_iid_null(
        rows=128, slots=4, dimension=5, repeats=12, seed=23
    )

    assert calibration["repeats"] == 12
    assert calibration["maximum_statistic_quantile_0_95"] > 0
    assert len(calibration["maximum_statistics"]) == 12

