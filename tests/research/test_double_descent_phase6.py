from pathlib import Path

import numpy as np
import pytest

from research.double_descent.benchmark import discover_cuda_python
from research.double_descent.freqai.Phase4CudaRFFRegressor import _CudaWorkerClient
from research.double_descent.phase6 import (
    CONVERGENCE_METRICS,
    EXTENSION_FEATURE_COUNTS,
    Phase6Config,
    aggregate_convergence,
    assess_kernel_limit,
    build_kernel_freqtrade_config,
    evaluate_phase6_gate,
)
from research.double_descent.synthetic import rbf_kernel


def _comparison(seed: int, feature_count: int, relative_error: float) -> dict:
    result = {
        "seed": seed,
        "feature_count": feature_count,
        "pn_ratio": feature_count / 100,
        "aligned_prediction_count": 8_760,
        "prediction_relative_l2_error": relative_error,
        "oos_mse": 0.2 + relative_error * 0.1,
        "oos_r2": -1.0,
        "oos_information_coefficient": 0.0,
        "oos_directional_accuracy": 0.5,
        "train_train_mse_mean": 0.0,
        "train_effective_rank_mean": 12.0,
        "train_condition_number_maximum": 100.0,
        "trading_sharpe": -1.0,
        "trading_profit_factor": 0.5,
        "trading_profit_total": -0.2,
    }
    for metric in CONVERGENCE_METRICS:
        result.setdefault(metric, relative_error)
    return result


def _kernel_result() -> dict:
    return {
        "training": {"training_seconds_total": 0.5},
        "oos": {
            "model": {
                "mse": 0.2,
                "r2": -1.0,
                "information_coefficient": 0.0,
                "directional_accuracy": 0.5,
            },
            "zero_baseline": {"mse": 0.1},
        },
        "trading": {"sharpe": -1.0, "profit_factor": 0.5, "profit_total": -0.2},
    }


def test_kernel_config_has_no_finite_rff_dimension(tmp_path: Path) -> None:
    phase5 = tmp_path / "phase5.json"
    phase5.write_text("{}", encoding="utf-8")
    config = Phase6Config(
        data_directory=tmp_path,
        phase5_summary=phase5,
        cuda_python_executable=str(tmp_path / "cuda-python.exe"),
    )

    generated = build_kernel_freqtrade_config(
        config,
        identifier="kernel-test",
        metrics_path=tmp_path / "metrics.jsonl",
        run_id="test",
    )

    parameters = generated["freqai"]["model_training_parameters"]
    assert parameters["gamma"] == 0.2
    assert parameters["ridge"] == 0
    assert parameters["dtype"] == "float64"
    assert "feature_count" not in parameters
    assert "seed" not in parameters


def test_phase6_convergence_grid_extends_to_one_million() -> None:
    assert EXTENSION_FEATURE_COUNTS == (250_000, 500_000, 1_000_000)


def test_phase6_gate_distinguishes_raw_predictions_from_forward_return_count(
    tmp_path: Path,
) -> None:
    phase5 = tmp_path / "phase5.json"
    phase5.write_text("{}", encoding="utf-8")
    config = Phase6Config(
        data_directory=tmp_path,
        phase5_summary=phase5,
        cuda_python_executable=str(tmp_path / "cuda-python.exe"),
    )
    feature_counts = (*config.reference_feature_counts, *config.extension_feature_counts)
    comparisons = [
        _comparison(seed, count, relative_error=0.1)
        for seed in config.seeds
        for count in feature_counts
    ]
    aggregates = aggregate_convergence(comparisons)
    kernel = _kernel_result()
    kernel.update(
        {
            "success": True,
            "training": {"cuda_devices": ["test GPU"], "peak_vram_mib": 1.0},
        }
    )

    gate = evaluate_phase6_gate(
        config,
        {"gate": {"passed": True}},
        kernel,
        [{"gate": {"passed": True}} for _ in config.seeds],
        comparisons,
        aggregates,
    )

    assert gate["passed"] is True


def test_convergence_assessment_separates_numerical_limit_from_useful_prediction(
    tmp_path: Path,
) -> None:
    phase5 = tmp_path / "phase5.json"
    phase5.write_text("{}", encoding="utf-8")
    config = Phase6Config(
        data_directory=tmp_path,
        phase5_summary=phase5,
        cuda_python_executable=str(tmp_path / "cuda-python.exe"),
        reference_feature_counts=(10_795, 53_975, 107_950),
        extension_feature_counts=(250_000, 500_000, 1_000_000),
    )
    feature_counts = (*config.reference_feature_counts, *config.extension_feature_counts)
    rows = [
        _comparison(seed, count, relative_error=0.5 / (index + 1))
        for seed in range(5)
        for index, count in enumerate(feature_counts)
    ]
    aggregates = aggregate_convergence(rows)
    phase5_summary = {"multi_seed_assessment": {"best_underparameterized_mean_mse": 0.15}}

    assessment = assess_kernel_limit(config, _kernel_result(), phase5_summary, aggregates)

    assert assessment["convergence_trend_detected"] is True
    assert assessment["practical_convergence_at_one_million"] is True
    assert assessment["kernel_supports_useful_benign_overfitting"] is False


def test_external_cuda_kernel_worker_matches_numpy_centered_kernel(tmp_path: Path) -> None:
    cuda_python = discover_cuda_python()
    if not cuda_python:
        pytest.skip("no external CUDA-enabled Python interpreter is available")
    rng = np.random.default_rng(17)
    inputs = rng.normal(size=(32, 5))
    inference_inputs = rng.normal(size=(7, 5))
    target = rng.normal(size=32)
    client = _CudaWorkerClient(
        cuda_python,
        tmp_path / "kernel-worker.log",
        worker_module="research.double_descent.kernel_cuda_worker",
        temporary_prefix="test-kernel-",
    )
    try:
        train_prediction, diagnostics = client.fit(
            inputs,
            target,
            {"gamma": 0.2, "ridge": 0.0, "rcond": 1e-12, "dtype": "float64"},
        )
        prediction = client.predict(inference_inputs)
    finally:
        client.close()

    train_kernel = rbf_kernel(inputs, inputs, gamma=0.2)
    test_kernel = rbf_kernel(inference_inputs, inputs, gamma=0.2)
    column_mean = train_kernel.mean(axis=0, keepdims=True)
    grand_mean = train_kernel.mean()
    centered_train = (
        train_kernel - train_kernel.mean(axis=1, keepdims=True) - column_mean + grand_mean
    )
    centered_test = test_kernel - test_kernel.mean(axis=1, keepdims=True) - column_mean + grand_mean
    target_mean = target.mean()
    eigenvalues, eigenvectors = np.linalg.eigh(centered_train)
    threshold = eigenvalues[-1] * max(1e-24, np.finfo(np.float64).eps * len(inputs))
    retained = eigenvalues > threshold
    alpha = eigenvectors[:, retained] @ (
        (eigenvectors[:, retained].T @ (target - target_mean)) / eigenvalues[retained]
    )
    expected_train = target_mean + centered_train @ alpha
    expected = target_mean + centered_test @ alpha

    assert diagnostics["kernel_limit"] is True
    assert diagnostics["rank"] == 31
    assert diagnostics["peak_vram_mib"] > 0
    np.testing.assert_allclose(train_prediction, expected_train, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(prediction, expected, rtol=1e-8, atol=1e-8)
