from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.double_descent.benchmark import discover_cuda_python
from research.double_descent.freqai.Phase4CudaRFFRegressor import _CudaWorkerClient
from research.double_descent.metrics import prediction_metrics
from research.double_descent.phase4 import (
    Phase4Config,
    assess_curve,
    build_freqtrade_config,
    evaluate_oos_predictions,
    feature_count_for_ratio,
)
from research.double_descent.rff_cuda_worker import iter_nested_rff_parameters


def _parameters(feature_count: int, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    chunks = list(
        iter_nested_rff_parameters(
            input_dimension=5,
            feature_count=feature_count,
            chunk_size=chunk_size,
            gamma=0.2,
            seed=42,
            dtype=np.dtype("float64"),
        )
    )
    return np.vstack([weights for weights, _ in chunks]), np.hstack(
        [phases for _, phases in chunks]
    )


def test_phase4_rff_parameters_are_nested_and_chunk_independent() -> None:
    small_weights, small_phases = _parameters(17, 5)
    large_weights, large_phases = _parameters(41, 13)

    np.testing.assert_array_equal(small_weights, large_weights[:17])
    np.testing.assert_array_equal(small_phases, large_phases[:17])


def test_prediction_metrics_report_perfect_forecast() -> None:
    actual = np.array([-0.02, -0.01, 0.01, 0.03])

    metrics = prediction_metrics(actual, actual)

    assert metrics["mse"] == 0
    assert metrics["r2"] == 1
    assert metrics["r2_vs_zero"] == 1
    assert metrics["information_coefficient"] == pytest.approx(1)
    assert metrics["directional_accuracy"] == 1


def test_phase4_config_freezes_ridgeless_freqai_controls(tmp_path: Path) -> None:
    config = Phase4Config(
        data_directory=tmp_path,
        cuda_python_executable=str(tmp_path / "cuda-python.exe"),
        ratios=(1.0,),
    )
    generated = build_freqtrade_config(
        config,
        feature_count=feature_count_for_ratio(1.0, config.effective_n),
        identifier="phase4-test",
        metrics_path=tmp_path / "metrics.jsonl",
        run_id="test-run",
    )

    freqai = generated["freqai"]
    features = freqai["feature_parameters"]
    model = freqai["model_training_parameters"]
    assert freqai["data_split_parameters"] == {"test_size": 0, "shuffle": False}
    assert model["feature_count"] == 2_159
    assert model["ridge"] == 0
    assert model["dtype"] == "float64"
    assert features["weight_factor"] == 0
    assert features["include_shifted_candles"] == 0
    assert features["principal_component_analysis"] is False
    assert generated["fee"] == 0.001


def test_oos_evaluation_aligns_predictions_to_next_hour_returns() -> None:
    dates = pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC")
    realized = np.array([0.01, -0.02, 0.03, -0.01, np.nan])
    market = pd.DataFrame(
        {
            "date": dates,
            "realized_forward_return": realized,
            "momentum_1h_prediction": np.array([np.nan, 0.01, -0.02, 0.03, -0.01]),
        }
    )
    predictions = pd.DataFrame(
        {
            "date": dates[:4],
            "&-forward_return": realized[:4],
            "do_predict": 1,
            "prediction_window": "window-1",
        }
    )

    result = evaluate_oos_predictions(predictions, market, "20250101-20250102")

    assert result["valid_prediction_rows"] == 4
    assert result["duplicate_prediction_dates"] == 0
    assert result["model"]["mse"] == 0
    assert result["model"]["directional_accuracy"] == 1


def test_curve_assessment_distinguishes_pattern_from_robust_evidence() -> None:
    def row(ratio: float, mse: float, interpolated: float) -> dict:
        return {
            "success": True,
            "actual_pn_ratio": ratio,
            "oos": {
                "model": {"mse": mse},
                "zero_baseline": {"mse": 1.0},
            },
            "training": {"interpolated_window_fraction": interpolated},
        }

    results = [
        row(0.5, 0.4, 0.0),
        row(1.0, 1.2, 1.0),
        row(2.0, 0.6, 1.0),
        row(10.0, 0.3, 1.0),
    ]

    assessment = assess_curve(results)

    assert assessment["double_descent_pattern_detected"] is True
    assert assessment["robust_evidence"] is False
    assert assessment["interpolation_onset_ratio"] == 1.0


def test_external_cuda_worker_interpolates_small_problem(tmp_path: Path) -> None:
    cuda_python = discover_cuda_python()
    if not cuda_python:
        pytest.skip("no external CUDA-enabled Python interpreter is available")
    rng = np.random.default_rng(7)
    inputs = rng.normal(size=(32, 5))
    inference_inputs = rng.normal(size=(4, 5))
    target = rng.normal(size=32)
    client = _CudaWorkerClient(cuda_python, tmp_path / "worker.log")
    try:
        train_prediction, diagnostics = client.fit(
            inputs,
            target,
            {
                "feature_count": 64,
                "chunk_size": 17,
                "gamma": 0.2,
                "seed": 42,
                "ridge": 0.0,
                "rcond": 1e-12,
                "dtype": "float64",
            },
        )
        prediction = client.predict(inference_inputs)
    finally:
        client.close()

    assert np.mean((train_prediction - target) ** 2) < 1e-20
    assert diagnostics["rank"] == 31
    assert diagnostics["peak_vram_mib"] > 0
    assert np.isfinite(prediction).all()

    weights, phases = _parameters(64, 11)
    train_rff = np.sqrt(2.0) * np.cos(inputs @ weights.T + phases)
    inference_rff = np.sqrt(2.0) * np.cos(inference_inputs @ weights.T + phases)
    feature_mean = np.mean(train_rff, axis=0)
    target_mean = np.mean(target)
    coefficients = np.linalg.lstsq(train_rff - feature_mean, target - target_mean, rcond=1e-12)[0]
    cpu_prediction = target_mean + (inference_rff - feature_mean) @ coefficients
    np.testing.assert_allclose(prediction, cpu_prediction, rtol=1e-7, atol=1e-7)
