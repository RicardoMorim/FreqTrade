from pathlib import Path

import pytest

from research.double_descent.phase4 import Phase4Config
from research.double_descent.phase8 import (
    PHASE8_MAIN_RATIOS,
    PHASE8_ROBUSTNESS_RATIOS,
    PHASE8_SEEDS,
    PHASE8_TRAINING_WINDOWS,
    Phase8Config,
    _task_grid,
    aggregate_endpoint_robustness,
    assess_training_window_effect,
)


def _row(window: int, seed: int, ratio: float, mse: float) -> dict:
    row = {
        "train_period_days": window,
        "effective_n": {30: 719, 60: 1439, 90: 2159, 180: 4319, 365: 8759}[window],
        "seed": seed,
        "target_pn_ratio": ratio,
        "actual_pn_ratio": ratio,
        "feature_count": round(ratio * {30: 719, 60: 1439, 90: 2159, 180: 4319, 365: 8759}[window]),
        "oos_mse": mse,
        "zero_mse": 1.0,
        "oos_mae": mse,
        "oos_r2": -mse,
        "oos_information_coefficient": 0.0,
        "oos_directional_accuracy": 0.5,
        "oos_prediction_standard_deviation": 0.1,
        "train_train_mse_mean": 0.01,
        "train_interpolated_window_fraction": float(ratio >= 1),
        "train_effective_rank_mean": 10.0,
        "train_condition_number_maximum": 100.0,
        "trading_total_trades": 100,
        "trading_profit_total": -0.2,
        "trading_sharpe": -1.0,
        "trading_sortino": -1.0,
        "trading_profit_factor": 0.5,
        "trading_max_drawdown_account": 0.2,
        "trading_turnover_multiple": 10.0,
    }
    return row


def test_phase8_grid_changes_only_window_and_bounds_extreme_compute() -> None:
    assert PHASE8_TRAINING_WINDOWS == (30, 60, 90, 180, 365)
    assert PHASE8_MAIN_RATIOS == (0.1, 0.5, 0.9, 0.98, 1.0, 1.02, 1.1, 2.0, 5.0)
    assert PHASE8_ROBUSTNESS_RATIOS == (0.1, 1.0, 1.02, 5.0)


def test_phase4_allows_only_exact_phase3_window_measurements(tmp_path: Path) -> None:
    python = tmp_path / "python.exe"
    cuda = tmp_path / "cuda.exe"
    python.touch()
    cuda.touch()
    valid = Phase4Config(
        data_directory=tmp_path,
        python_executable=str(python),
        cuda_python_executable=str(cuda),
        train_period_days=365,
        effective_n=8759,
        ratios=(1.0,),
        allow_phase3_training_window_variation=True,
    )
    valid.validate()

    with pytest.raises(ValueError, match="exactly match"):
        Phase4Config(
            data_directory=tmp_path,
            python_executable=str(python),
            cuda_python_executable=str(cuda),
            train_period_days=365,
            effective_n=8760,
            ratios=(1.0,),
            allow_phase3_training_window_variation=True,
        ).validate()


def test_phase8_task_grid_reuses_main_seed_and_replicates_only_endpoints(tmp_path: Path) -> None:
    config = Phase8Config(data_directory=tmp_path)

    tasks = _task_grid(config)

    expected_case_count = 5 * 9 + 2 * 2 * 4
    assert sum(len(ratios) for _, _, ratios in tasks) == expected_case_count
    assert sum(seed == PHASE8_SEEDS[0] for _, seed, _ in tasks) == 5
    assert {window for window, seed, _ in tasks if seed != PHASE8_SEEDS[0]} == {30, 365}


def test_window_assessment_distinguishes_persistent_shape_from_useful_prediction(
    tmp_path: Path,
) -> None:
    config = Phase8Config(data_directory=tmp_path)
    rows = []
    main_mse = {
        0.1: 0.4,
        0.5: 0.5,
        0.9: 1.0,
        0.98: 1.1,
        1.0: 1.2,
        1.02: 1.1,
        1.1: 0.9,
        2.0: 0.7,
        5.0: 0.6,
    }
    robust_mse = {0.1: 0.4, 1.0: 1.2, 1.02: 1.1, 5.0: 0.6}
    for window in PHASE8_TRAINING_WINDOWS:
        rows.extend(_row(window, PHASE8_SEEDS[0], ratio, mse) for ratio, mse in main_mse.items())
    for window in (30, 365):
        for seed in PHASE8_SEEDS[1:]:
            rows.extend(_row(window, seed, ratio, mse) for ratio, mse in robust_mse.items())

    assessment = assess_training_window_effect(config, rows)
    aggregates = aggregate_endpoint_robustness(config, rows)

    assert assessment["reference_seed_shape_window_count"] == 5
    assert assessment["double_descent_persists_at_longest_window"] is True
    assert assessment["predictive_alpha_evidence"] is False
    assert len(aggregates) == 2 * 4 * 17
