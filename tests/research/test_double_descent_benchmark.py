from pathlib import Path

import numpy as np
import pytest

from research.double_descent.benchmark import (
    BenchmarkCase,
    Phase2Config,
    benchmark_case,
    discover_cuda_python,
    iter_rff_parameter_chunks,
    run_isolated_case,
    run_phase2,
)


def _parameters(feature_count: int, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    chunks = list(
        iter_rff_parameter_chunks(
            input_dimension=6,
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


def test_streamed_rff_parameters_are_nested_and_chunk_size_independent() -> None:
    small_weights, small_phases = _parameters(feature_count=17, chunk_size=5)
    large_weights, large_phases = _parameters(feature_count=41, chunk_size=13)

    np.testing.assert_array_equal(small_weights, large_weights[:17])
    np.testing.assert_array_equal(small_phases, large_phases[:17])


def test_primal_and_streamed_dual_predictions_match() -> None:
    cases = (
        BenchmarkCase(64, "primal_svd"),
        BenchmarkCase(64, "streamed_dual"),
    )
    config = Phase2Config(
        n_train=32,
        n_inference=48,
        input_dimension=6,
        cases=cases,
        chunk_size=19,
        base_seed=123,
    )
    primal = benchmark_case(cases[0], config)
    dual = benchmark_case(cases[1], config)

    np.testing.assert_allclose(
        primal["prediction_probe"],
        dual["prediction_probe"],
        rtol=1e-8,
        atol=1e-8,
    )
    assert np.isclose(primal["oos_mse"], dual["oos_mse"], rtol=1e-8, atol=1e-8)


def test_isolated_benchmark_writes_artifacts(tmp_path: Path) -> None:
    cases = (
        BenchmarkCase(32, "primal_svd"),
        BenchmarkCase(32, "streamed_dual"),
    )
    config = Phase2Config(
        n_train=32,
        n_inference=48,
        input_dimension=6,
        cases=cases,
        chunk_size=16,
        base_seed=456,
        memory_sample_interval_seconds=0.001,
    )

    isolated = run_isolated_case(cases[0], config)
    assert isolated["success"] is True
    assert isolated["peak_rss_mib"] > 0

    summary = run_phase2(config, tmp_path)
    assert all(row["success"] for row in summary["results"])
    assert summary["gate"]["checks"]["primal_dual_overlap_matches"] is True
    assert (tmp_path / "benchmark.csv").is_file()
    assert (tmp_path / "summary.json").is_file()


def test_cuda_streamed_predictions_match_cpu_when_available() -> None:
    cuda_python = discover_cuda_python()
    if not cuda_python:
        pytest.skip("no CUDA-enabled PyTorch interpreter is available")
    cases = (
        BenchmarkCase(64, "streamed_dual"),
        BenchmarkCase(64, "torch_cuda_dual"),
    )
    config = Phase2Config(
        n_train=32,
        n_inference=48,
        input_dimension=6,
        cases=cases,
        chunk_size=17,
        base_seed=789,
        cuda_python_executable=cuda_python,
    )

    cpu = run_isolated_case(cases[0], config)
    cuda = run_isolated_case(cases[1], config)

    assert cpu["success"] is True
    assert cuda["success"] is True, cuda.get("stderr")
    assert cuda["vram_measured"] is True
    assert cuda["peak_vram_mib"] > 0
    np.testing.assert_allclose(
        cpu["prediction_probe"],
        cuda["prediction_probe"],
        rtol=config.overlap_relative_tolerance,
        atol=config.overlap_relative_tolerance,
    )
