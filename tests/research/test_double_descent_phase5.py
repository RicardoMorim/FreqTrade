from pathlib import Path

import pytest

from research.double_descent.phase5 import (
    AGGREGATE_METRICS,
    PHASE5_SEEDS,
    Phase5Config,
    aggregate_seed_summaries,
    assess_multi_seed,
    evaluate_phase5_gate,
    phase4_config,
    summarize_values,
)


def _seed_summary(seed: int, extreme_mse: float = 0.3) -> dict:
    def row(ratio: float, mse: float, interpolated: float) -> dict:
        result = {
            "target_pn_ratio": ratio,
            "actual_pn_ratio": ratio,
            "feature_count": round(ratio * 100),
            "seed": seed,
            "success": True,
            "oos_observation_count": 100,
            "zero_mse": 0.35,
            "oos_mse": mse,
            "oos_r2": 0.1,
            "oos_information_coefficient": 0.05,
            "oos_spearman_information_coefficient": 0.04,
            "oos_directional_accuracy": 0.52,
            "train_train_mse_mean": 1e-8 if interpolated else 0.1,
            "train_condition_number_maximum": 100.0,
            "train_effective_rank_mean": 20.0,
            "trading_sharpe": 0.7,
            "trading_profit_factor": 1.1,
            "trading_profit_total": 0.05,
            "trading_max_drawdown_account": 0.1,
        }
        for metric in AGGREGATE_METRICS:
            result.setdefault(metric, 1.0)
        return result

    return {
        "phase": 4,
        "config": {"seed": seed},
        "gate": {"passed": True},
        "results": [row(0.5, 0.4, 0), row(1.0, 1.2, 1), row(10.0, extreme_mse, 1)],
        "curve_assessment": {
            "status": "preliminary_single_seed",
            "global_peak_ratio": 1.0,
            "global_peak_is_near_threshold": True,
            "interpolation_onset_ratio": 1.0,
            "recovery_fraction": 1.125,
            "double_descent_pattern_detected": True,
            "largest_model_beats_zero_mse": extreme_mse < 0.35,
            "second_descent_beats_best_underparameterized": extreme_mse < 0.4,
        },
    }


def test_phase5_seeds_are_frozen_and_unique() -> None:
    assert PHASE5_SEEDS == (20260810, 1898170439, 3432960257, 2841638297, 2713191886)
    assert len(set(PHASE5_SEEDS)) == 5


def test_phase5_changes_only_seed_and_output_from_frozen_phase4(tmp_path: Path) -> None:
    config = Phase5Config(
        data_directory=tmp_path,
        cuda_python_executable=str(tmp_path / "cuda-python.exe"),
    )
    first = phase4_config(config, 1, tmp_path / "first")
    second = phase4_config(config, 2, tmp_path / "second")

    assert first.seed == 1 and second.seed == 2
    assert first.output_directory != second.output_directory
    assert first.ratios == second.ratios
    assert first.gamma == second.gamma == 0.2
    assert first.ridge == second.ridge == 0
    assert first.timerange == second.timerange


def test_seed_summary_uses_student_t_interval() -> None:
    summary = summarize_values([1, 2, 3, 4, 5])

    assert summary["mean"] == 3
    assert summary["median"] == 3
    assert summary["standard_deviation"] == pytest.approx(1.58113883)
    assert summary["ci95_low"] == pytest.approx(1.0367568)
    assert summary["ci95_high"] == pytest.approx(4.9632432)


def test_seed_summary_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        summarize_values([1.0, float("nan")])


def test_multi_seed_assessment_separates_shape_from_benign_overfitting() -> None:
    summaries = [_seed_summary(seed, extreme_mse=0.5) for seed in range(5)]
    aggregates = aggregate_seed_summaries(summaries)

    assessment = assess_multi_seed(summaries, aggregates)

    assert assessment["robust_double_descent_shape"] is True
    assert assessment["benign_overfitting_evidence"] is False
    assert assessment["largest_model_beats_zero_seed_count"] == 0


def test_phase5_gate_requires_complete_matched_seed_grid(tmp_path: Path) -> None:
    seeds = (10, 11, 12)
    summaries = [_seed_summary(seed) for seed in seeds]
    aggregates = aggregate_seed_summaries(summaries)
    config = Phase5Config(
        data_directory=tmp_path,
        cuda_python_executable=str(tmp_path / "cuda-python.exe"),
        ratios=(0.5, 1.0, 10.0),
        seeds=seeds,
        minimum_seed_count=3,
    )

    gate = evaluate_phase5_gate(config, summaries, aggregates)

    assert len(aggregates) == len(config.ratios) * len(AGGREGATE_METRICS)
    assert gate["passed"] is True
