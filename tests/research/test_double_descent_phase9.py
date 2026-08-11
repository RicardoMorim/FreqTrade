from pathlib import Path

import numpy as np

from research.double_descent.phase9 import (
    PHASE9_GAMMAS,
    PHASE9_MAP_RATIOS,
    PHASE9_ROBUSTNESS_RATIOS,
    PHASE9_SEEDS,
    PHASE9_SELECTION_RATIOS,
    Phase9Config,
    _task_grid,
    select_gamma,
)
from research.double_descent.rff_cuda_worker import iter_nested_rff_parameters


def _row(gamma: float, seed: int, ratio: float, mse_over_zero: float) -> dict:
    return {
        "gamma": gamma,
        "seed": seed,
        "target_pn_ratio": ratio,
        "actual_pn_ratio": ratio,
        "oos_mse": mse_over_zero,
        "zero_mse": 1.0,
    }


def _curve(gamma: float, seed: int, detected: bool = True) -> dict:
    return {"gamma": gamma, "seed": seed, "double_descent_pattern_detected": detected}


def test_phase9_grid_and_selection_controls_are_frozen() -> None:
    assert PHASE9_GAMMAS == (0.05, 0.1, 0.2, 0.5)
    assert PHASE9_MAP_RATIOS == (0.1, 0.5, 0.9, 0.98, 1.0, 1.02, 1.1, 2.0, 5.0, 10.0, 50.0)
    assert PHASE9_ROBUSTNESS_RATIOS == (0.1, 1.0, 1.02, 5.0, 50.0)
    assert PHASE9_SELECTION_RATIOS == (0.1, 5.0, 50.0)


def test_gamma_rescales_the_same_nested_gaussian_draws() -> None:
    small = list(
        iter_nested_rff_parameters(
            input_dimension=5,
            feature_count=17,
            chunk_size=7,
            gamma=0.05,
            seed=42,
            dtype=np.dtype("float64"),
        )
    )
    large = list(
        iter_nested_rff_parameters(
            input_dimension=5,
            feature_count=17,
            chunk_size=7,
            gamma=0.2,
            seed=42,
            dtype=np.dtype("float64"),
        )
    )
    small_weights = np.vstack([weights for weights, _ in small])
    large_weights = np.vstack([weights for weights, _ in large])
    small_phases = np.hstack([phases for _, phases in small])
    large_phases = np.hstack([phases for _, phases in large])

    np.testing.assert_allclose(large_weights, 2 * small_weights, rtol=0, atol=0)
    np.testing.assert_array_equal(large_phases, small_phases)


def test_phase9_task_grid_has_84_predeclared_cases(tmp_path: Path) -> None:
    config = Phase9Config(data_directory=tmp_path)
    tasks = _task_grid(config)

    assert sum(len(ratios) for _, _, ratios in tasks) == 84
    assert len(tasks) == 12


def test_gamma_challenger_requires_material_score_and_matched_wins(tmp_path: Path) -> None:
    config = Phase9Config(data_directory=tmp_path)
    rows = []
    curves = []
    for gamma in PHASE9_GAMMAS:
        for seed in PHASE9_SEEDS:
            curves.append(_curve(gamma, seed))
            for ratio in PHASE9_SELECTION_RATIOS:
                score = 1.0
                if gamma == 0.1:
                    score = 0.9
                elif gamma in (0.05, 0.5):
                    score = 1.1
                rows.append(_row(gamma, seed, ratio, score))

    selection = select_gamma(config, rows, curves)

    assert selection["raw_best_gamma"] == 0.1
    assert selection["promote_challenger"] is True
    assert selection["selected_gamma"] == 0.1
    assert selection["trading_metrics_used_for_selection"] is False


def test_gamma_retains_baseline_when_improvement_is_too_small(tmp_path: Path) -> None:
    config = Phase9Config(data_directory=tmp_path)
    rows = []
    curves = []
    for gamma in PHASE9_GAMMAS:
        for seed in PHASE9_SEEDS:
            curves.append(_curve(gamma, seed))
            for ratio in PHASE9_SELECTION_RATIOS:
                score = 0.97 if gamma == 0.1 else (1.0 if gamma == 0.2 else 1.1)
                rows.append(_row(gamma, seed, ratio, score))

    selection = select_gamma(config, rows, curves)

    assert selection["raw_best_gamma"] == 0.1
    assert selection["promote_challenger"] is False
    assert selection["selected_gamma"] == 0.2
