from pathlib import Path

import numpy as np
import pytest

from research.double_descent.benchmark import discover_cuda_python
from research.double_descent.freqai.Phase4CudaRFFRegressor import _CudaWorkerClient
from research.double_descent.phase10 import (
    PHASE10_CURVE_REPRESENTATIONS,
    PHASE10_EXPECTED_CASE_COUNT,
    PHASE10_GAMMA,
    PHASE10_MAP_RATIOS,
    PHASE10_REPRESENTATIONS,
    PHASE10_ROBUSTNESS_RATIOS,
    PHASE10_SEEDS,
    Phase10Config,
    _composition_counts,
    _task_grid,
)
from research.double_descent.rff_cuda_worker import iter_stateless_noise_features


def _noise(
    sample_ids: np.ndarray,
    feature_count: int,
    chunk_size: int,
    seed: int = 42,
    feature_offset: int = 0,
) -> np.ndarray:
    return np.hstack(
        list(
            iter_stateless_noise_features(
                sample_ids,
                feature_count,
                chunk_size,
                seed,
                np.dtype("float64"),
                feature_offset=feature_offset,
            )
        )
    )


def test_phase10_design_is_frozen() -> None:
    assert PHASE10_REPRESENTATIONS == (
        "market_linear",
        "market_rff",
        "pure_noise",
        "market_plus_noise",
    )
    assert PHASE10_CURVE_REPRESENTATIONS == PHASE10_REPRESENTATIONS[1:]
    assert PHASE10_GAMMA == 0.5
    assert PHASE10_MAP_RATIOS == (
        0.1,
        0.5,
        0.9,
        0.98,
        1.0,
        1.02,
        1.1,
        2.0,
        5.0,
        10.0,
        50.0,
    )
    assert PHASE10_ROBUSTNESS_RATIOS == (0.1, 1.0, 1.02, 5.0, 50.0)


def test_phase10_task_grid_has_64_cases(tmp_path: Path) -> None:
    config = Phase10Config(data_directory=tmp_path)
    tasks = _task_grid(config)

    assert len(tasks) == 10
    assert sum(len(ratios) for _, _, ratios in tasks) == PHASE10_EXPECTED_CASE_COUNT
    assert tasks[0] == ("market_linear", PHASE10_SEEDS[0], (25 / 2_159,))


def test_noise_features_are_nested_chunk_independent_and_sample_keyed() -> None:
    sample_ids = np.array([1_700_000_000, 1_700_003_600, 1_700_007_200], dtype=np.int64)
    small = _noise(sample_ids, 17, 5)
    large = _noise(sample_ids, 41, 13)
    reordered = _noise(sample_ids[::-1], 17, 7)

    np.testing.assert_array_equal(small, large[:, :17])
    np.testing.assert_array_equal(reordered, small[::-1])


def test_mixed_control_replaces_exactly_25_matched_noise_columns() -> None:
    sample_ids = np.array([1_700_000_000, 1_700_003_600], dtype=np.int64)
    pure = _noise(sample_ids, 64, 11)
    mixed_noise = _noise(sample_ids, 39, 7, feature_offset=25)

    np.testing.assert_array_equal(mixed_noise, pure[:, 25:])


def test_noise_features_have_sane_standard_normal_moments() -> None:
    sample_ids = np.arange(400, dtype=np.int64) * 3_600 + 1_700_000_000
    features = _noise(sample_ids, 200, 37)

    assert abs(float(np.mean(features))) < 0.02
    assert float(np.std(features)) == pytest.approx(1.0, abs=0.02)


def test_feature_composition_preserves_total_p() -> None:
    assert _composition_counts("market_linear", 25) == (25, 0)
    assert _composition_counts("market_rff", 2_159) == (0, 0)
    assert _composition_counts("pure_noise", 2_159) == (0, 2_159)
    assert _composition_counts("market_plus_noise", 2_159) == (25, 2_134)


def test_cuda_noise_worker_interpolates_and_is_repeatable(tmp_path: Path) -> None:
    cuda_python = discover_cuda_python()
    if not cuda_python:
        pytest.skip("no external CUDA-enabled Python interpreter is available")
    rng = np.random.default_rng(7)
    train_inputs = rng.normal(size=(32, 25))
    inference_inputs = rng.normal(size=(5, 25))
    target = rng.normal(size=32)
    train_ids = np.arange(32, dtype=np.int64) + 1_700_000_000
    inference_ids = np.arange(5, dtype=np.int64) + 1_800_000_000
    parameters = {
        "feature_count": 64,
        "chunk_size": 17,
        "gamma": PHASE10_GAMMA,
        "seed": 42,
        "ridge": 0.0,
        "rcond": 1e-12,
        "dtype": "float64",
        "representation": "pure_noise",
    }
    client = _CudaWorkerClient(cuda_python, tmp_path / "phase10-worker.log")
    try:
        train_prediction, diagnostics = client.fit(
            train_inputs,
            target,
            parameters,
            sample_ids=train_ids,
        )
        first = client.predict(inference_inputs, sample_ids=inference_ids)
        second = client.predict(inference_inputs, sample_ids=inference_ids)
    finally:
        client.close()

    assert np.mean((train_prediction - target) ** 2) < 1e-20
    assert diagnostics["representation"] == "pure_noise"
    assert diagnostics["noise_feature_count"] == 64
    np.testing.assert_array_equal(first, second)
