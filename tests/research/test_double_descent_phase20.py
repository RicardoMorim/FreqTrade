from __future__ import annotations

import math

import pytest

from research.double_descent.phase20 import (
    PHASE20_EFFECTIVE_RANK_DEFINITION,
    aggregate_effective_dimensions,
    build_curve_summaries,
    build_phase10_contrasts,
    dimension_metrics,
)


def _row(
    representation: str,
    feature_count: int,
    effective_rank: float,
    seed: int,
) -> dict[str, object]:
    effective_n = 100
    metrics = dimension_metrics(
        feature_count=feature_count,
        effective_n=effective_n,
        effective_rank=effective_rank,
        algebraic_rank=min(feature_count, effective_n - 1),
    )
    return {
        "source_phase": 10,
        "study": "representation_controls_1h",
        "asset": "btc",
        "asset_label": "BTC",
        "representation": representation,
        "seed": seed,
        "gamma": 0.5,
        "feature_count": feature_count,
        "nominal_complexity": "finite_features",
        "effective_n": effective_n,
        "actual_pn_ratio": feature_count / effective_n,
        "effective_rank": effective_rank,
        "algebraic_rank": min(feature_count, effective_n - 1),
        "condition_number": 10.0,
        "training_window_count": 13,
        "dtype": "float64",
        **metrics,
    }


def test_dimension_metrics_keep_nominal_algebraic_and_spectral_dimensions_separate() -> None:
    metrics = dimension_metrics(
        feature_count=1_000_000,
        effective_n=2_159,
        effective_rank=25.0,
        algebraic_rank=2_158,
    )
    assert metrics["available_rank"] == 2_158
    assert metrics["nominal_to_effective_dimension"] == 40_000
    assert metrics["algebraic_to_effective_dimension"] == pytest.approx(86.32)
    assert metrics["effective_rank_fraction_of_n"] == pytest.approx(25 / 2_159)
    assert "participation ratio" in PHASE20_EFFECTIVE_RANK_DEFINITION


def test_dimension_metrics_reject_impossible_effective_rank() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        dimension_metrics(
            feature_count=20,
            effective_n=100,
            effective_rank=21.0,
            algebraic_rank=20,
        )


def test_aggregation_reports_seed_uncertainty_without_inventing_replicates() -> None:
    rows = [_row("market_rff", 50, rank, seed) for rank, seed in ((10, 1), (12, 2), (14, 3))]
    aggregate = aggregate_effective_dimensions(rows)[0]
    assert aggregate["seed_count"] == 3
    assert aggregate["effective_rank_median"] == 12
    assert aggregate["effective_rank_standard_deviation"] == 2
    assert aggregate["effective_rank_ci95_low"] < 12 < aggregate["effective_rank_ci95_high"]


def test_curve_summary_measures_sublinear_effective_rank_growth() -> None:
    rows = [
        _row("market_rff", feature_count, effective_rank, 1)
        for feature_count, effective_rank in ((10, 8), (100, 10), (1_000, 12))
    ]
    curve = build_curve_summaries(aggregate_effective_dimensions(rows))[0]
    assert curve["nominal_growth"] == 100
    assert curve["effective_rank_growth"] == 1.5
    assert curve["log_effective_rank_elasticity"] == pytest.approx(
        math.log(1.5) / math.log(100)
    )
    assert curve["largest_nominal_to_effective_dimension"] == pytest.approx(1_000 / 12)


def test_phase10_contrasts_require_three_matched_representations() -> None:
    rows = [
        _row(representation, 100, effective_rank, seed)
        for representation, effective_rank in (
            ("market_rff", 10),
            ("pure_noise", 90),
            ("market_plus_noise", 85),
        )
        for seed in (1, 2, 3)
    ]
    contrast = build_phase10_contrasts(aggregate_effective_dimensions(rows))[0]
    assert contrast["market_rff_to_noise_effective_rank_ratio"] == pytest.approx(1 / 9)
    assert contrast["market_plus_noise_to_noise_effective_rank_ratio"] == pytest.approx(85 / 90)
    with pytest.raises(ValueError, match="not matched"):
        build_phase10_contrasts(
            aggregate_effective_dimensions(
                [row for row in rows if row["representation"] != "market_plus_noise"]
            )
        )
