from pathlib import Path

import numpy as np
import pytest

from research.double_descent.benchmark import discover_cuda_python
from research.double_descent.freqai.Phase4CudaRFFRegressor import _CudaWorkerClient
from research.double_descent.phase7 import (
    RIDGE_LAMBDAS,
    ROBUSTNESS_RATIOS,
    Phase7Config,
    aggregate_robustness,
    assess_regularization_map,
    evaluate_regularization_diagnostics,
)
from research.double_descent.rff_cuda_worker import iter_nested_rff_parameters


def _row(seed: int, ridge: float, ratio: float) -> dict:
    shrinkage = 1 / (1 + ridge)
    mse = 0.2 * shrinkage + 0.1
    return {
        "seed": seed,
        "ridge": ridge,
        "target_pn_ratio": ratio,
        "actual_pn_ratio": ratio,
        "feature_count": round(ratio * 100),
        "oos_mse": mse,
        "oos_mae": mse,
        "oos_r2": -1.0,
        "oos_information_coefficient": 0.0,
        "oos_spearman_information_coefficient": 0.0,
        "oos_directional_accuracy": 0.5,
        "oos_prediction_standard_deviation": shrinkage,
        "zero_mse": 0.1,
        "train_train_mse_mean": 0.01 + ridge,
        "train_interpolated_window_fraction": float(ridge == 0),
        "train_normalized_feature_coefficient_norm_mean": shrinkage,
        "train_ridge_effective_degrees_of_freedom_mean": 20 * shrinkage,
        "train_ridge_system_condition_number_mean": 100 * shrinkage,
        "trading_total_trades": 100,
        "trading_profit_total": -0.2,
        "trading_sharpe": -1.0,
        "trading_sortino": -1.0,
        "trading_profit_factor": 0.5,
        "trading_max_drawdown_account": 0.2,
        "trading_turnover_multiple": 10.0,
    }


def test_phase7_grid_includes_ridgeless_and_strong_controls() -> None:
    assert RIDGE_LAMBDAS == (0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 100.0)
    assert ROBUSTNESS_RATIOS == (0.1, 1.0, 1.02, 50.0)


def test_regularization_diagnostics_enforce_theoretical_monotonicity(tmp_path: Path) -> None:
    phase5 = tmp_path / "phase5.json"
    phase5.write_text("{}", encoding="utf-8")
    config = Phase7Config(
        data_directory=tmp_path,
        phase5_summary=phase5,
        cuda_python_executable=str(tmp_path / "cuda.exe"),
        map_ratios=(0.1, 1.0),
        robustness_ratios=(0.1, 1.0),
        ridge_lambdas=(0.0, 1.0),
        seeds=(1, 2, 3),
    )
    rows = [
        _row(seed, ridge, ratio)
        for seed in config.seeds
        for ridge in config.ridge_lambdas
        for ratio in config.robustness_ratios
    ]

    diagnostics = evaluate_regularization_diagnostics(config, rows)

    assert diagnostics["passed"] is True
    rows[-1]["train_ridge_effective_degrees_of_freedom_mean"] = 1_000
    assert evaluate_regularization_diagnostics(config, rows)["passed"] is False


def test_assessment_separates_shrinkage_from_predictive_alpha(tmp_path: Path) -> None:
    phase5 = tmp_path / "phase5.json"
    phase5.write_text("{}", encoding="utf-8")
    config = Phase7Config(
        data_directory=tmp_path,
        phase5_summary=phase5,
        cuda_python_executable=str(tmp_path / "cuda.exe"),
        map_ratios=(0.1, 1.0),
        robustness_ratios=(0.1, 1.0),
        ridge_lambdas=(0.0, 1.0),
        seeds=(1, 2, 3),
    )
    rows = [
        _row(seed, ridge, ratio)
        for seed in config.seeds
        for ridge in config.ridge_lambdas
        for ratio in config.robustness_ratios
    ]
    aggregates = aggregate_robustness(config, rows)

    assessment = assess_regularization_map(config, rows, aggregates)

    assert assessment["strong_regularization_suppresses_interpolation_catastrophe"] is False
    assert assessment["best_cell_robustly_beats_zero"] is False
    assert assessment["predictive_alpha_evidence"] is False
    assert assessment["shrinkage_only_explanation_supported"] is True


def test_external_cuda_ridge_worker_matches_numpy_solution(tmp_path: Path) -> None:
    cuda_python = discover_cuda_python()
    if not cuda_python:
        pytest.skip("no external CUDA-enabled Python interpreter is available")
    rng = np.random.default_rng(71)
    inputs = rng.normal(size=(32, 5))
    inference_inputs = rng.normal(size=(7, 5))
    target = rng.normal(size=32)
    feature_count = 64
    ridge = 0.01
    client = _CudaWorkerClient(cuda_python, tmp_path / "ridge-worker.log")
    try:
        train_prediction, diagnostics = client.fit(
            inputs,
            target,
            {
                "feature_count": feature_count,
                "chunk_size": 17,
                "gamma": 0.2,
                "seed": 42,
                "ridge": ridge,
                "rcond": 1e-12,
                "dtype": "float64",
            },
        )
        prediction = client.predict(inference_inputs)
    finally:
        client.close()

    chunks = list(
        iter_nested_rff_parameters(
            input_dimension=5,
            feature_count=feature_count,
            chunk_size=11,
            gamma=0.2,
            seed=42,
            dtype=np.dtype("float64"),
        )
    )
    weights = np.vstack([item[0] for item in chunks])
    phases = np.hstack([item[1] for item in chunks])
    train_rff = np.sqrt(2.0) * np.cos(inputs @ weights.T + phases)
    inference_rff = np.sqrt(2.0) * np.cos(inference_inputs @ weights.T + phases)
    feature_mean = train_rff.mean(axis=0)
    centered_train = train_rff - feature_mean
    centered_inference = inference_rff - feature_mean
    gram = centered_train @ centered_train.T / feature_count
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    threshold = eigenvalues[-1] * max(1e-24, np.finfo(np.float64).eps * len(inputs))
    retained = eigenvalues > threshold
    retained_values = eigenvalues[retained]
    retained_vectors = eigenvectors[:, retained]
    centered_target = target - target.mean()
    projected = retained_vectors.T @ centered_target
    alpha = retained_vectors @ (projected / (retained_values + ridge))
    expected_train = target.mean() + gram @ alpha
    expected = target.mean() + centered_inference @ centered_train.T @ alpha / feature_count
    expected_df = np.sum(retained_values / (retained_values + ridge))
    expected_norm = np.sqrt(alpha @ gram @ alpha)

    np.testing.assert_allclose(train_prediction, expected_train, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(prediction, expected, rtol=1e-8, atol=1e-8)
    assert diagnostics["ridge_effective_degrees_of_freedom"] == pytest.approx(expected_df)
    assert diagnostics["normalized_feature_coefficient_norm"] == pytest.approx(expected_norm)
