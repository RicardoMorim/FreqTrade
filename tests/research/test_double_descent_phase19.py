from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from research.double_descent.freqai.Phase19CaptureRegressor import _ZeroCapturePredictor
from research.double_descent.phase19 import (
    PHASE19_CPU_REFERENCE_RATIO,
    PHASE19_EXPECTED_CASES,
    PHASE19_RATIOS,
    PHASE19_SEEDS,
    Phase19Config,
    _audit_capture_window,
    _comparison,
    _cpu_primal_svd_reference,
    assess_numerical_stability,
    phase19_cases,
    prepare_phase19,
)
from research.double_descent.rff_cuda_worker import iter_nested_rff_parameters


def _summaries(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    data = tmp_path / "data"
    data.mkdir()
    freqtrade_python = tmp_path / "freqtrade-python.exe"
    cuda_python = tmp_path / "cuda-python.exe"
    freqtrade_python.write_text("placeholder", encoding="utf-8")
    cuda_python.write_text("placeholder", encoding="utf-8")
    phase10 = tmp_path / "phase10.json"
    phase18 = tmp_path / "phase18.json"
    phase10.write_text(
        json.dumps(
            {
                "phase": 10,
                "gate": {"passed": True},
                "design": {"holdout_used": False},
                "config": {
                    "effective_n": 2_159,
                    "gamma": 0.5,
                    "ridge": 0.0,
                    "rcond": 1e-12,
                    "dtype": "float64",
                    "seeds": list(PHASE19_SEEDS),
                    "holdout_start": "20260101",
                    "cuda_python_executable": str(cuda_python),
                },
            }
        ),
        encoding="utf-8",
    )
    phase18.write_text(
        json.dumps(
            {
                "phase": 18,
                "gate": {"passed": True},
                "design": {"holdout_used": False},
            }
        ),
        encoding="utf-8",
    )
    return data, freqtrade_python, phase10, phase18


def _config(tmp_path: Path) -> Phase19Config:
    data, freqtrade_python, phase10, phase18 = _summaries(tmp_path)
    return Phase19Config(
        data_directory=data,
        phase10_summary=phase10,
        phase18_summary=phase18,
        output_directory=tmp_path / "phase19",
        project_root=tmp_path,
        python_executable=str(freqtrade_python),
    )


def test_phase19_protocol_is_frozen() -> None:
    assert PHASE19_RATIOS == (0.90, 0.98, 1.00, 1.02, 1.10)
    assert len(PHASE19_SEEDS) == 3
    assert PHASE19_CPU_REFERENCE_RATIO == 1.0
    cases = phase19_cases()
    assert len(cases) == PHASE19_EXPECTED_CASES == 15
    assert len({case.case_id for case in cases}) == 15
    assert {case.feature_count for case in cases} == {1_943, 2_116, 2_159, 2_202, 2_375}


def test_preparation_requires_passed_frozen_sources(tmp_path: Path) -> None:
    config = _config(tmp_path)
    summary = prepare_phase19(config)
    assert summary["gate"]["passed"] is True
    assert summary["design"]["holdout_used"] is False
    phase10 = json.loads(config.phase10_summary.read_text(encoding="utf-8"))
    phase10["config"]["dtype"] = "float32"
    config.phase10_summary.write_text(json.dumps(phase10), encoding="utf-8")
    try:
        config.validate()
    except ValueError as exc:
        assert "frozen source protocol mismatch" in str(exc)
    else:
        raise AssertionError("Phase 19 accepted a changed Phase 10 precision")


def test_zero_capture_predictor_matches_inference_rows(tmp_path: Path) -> None:
    predictor = _ZeroCapturePredictor(tmp_path)
    prediction = predictor.predict(pd.DataFrame({"feature": [1.0, 2.0, 3.0]}))
    np.testing.assert_array_equal(prediction, np.zeros(3))
    assert predictor.window_directory == tmp_path


def test_capture_audit_requires_real_effective_n_and_anchor_alignment(tmp_path: Path) -> None:
    window = tmp_path / "window"
    window.mkdir()
    train_dates = pd.date_range("2025-01-01", periods=2_159, freq="h", tz="UTC")
    inference_dates = pd.date_range("2025-04-01", periods=600, freq="h", tz="UTC")
    np.savez_compressed(
        window / "train.npz",
        features=np.zeros((2_159, 25), dtype=np.float64),
        target_scaled=np.zeros(2_159, dtype=np.float64),
        dates_ns=train_dates.to_numpy(dtype="datetime64[ns]").astype(np.int64),
    )
    target = np.zeros(600, dtype=np.float64)
    np.savez_compressed(
        window / "inference.npz",
        features=np.zeros((600, 25), dtype=np.float64),
        target_raw=target,
        dates_ns=inference_dates.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        do_predict=np.ones(600, dtype=np.int8),
    )
    (window / "metadata.json").write_text(
        json.dumps({"target_inverse_zero": 0.0, "target_inverse_one": 1.0}),
        encoding="utf-8",
    )
    audit = _audit_capture_window(window)
    assert audit["passed"] is True
    assert audit["train_rows"] == 2_159
    assert audit["inference_rows"] == 600


def test_comparison_reports_exact_repeat_and_precision_drift() -> None:
    reference = np.array([-1.0, 0.0, 2.0])
    exact = _comparison(reference, reference.copy())
    assert exact["max_abs"] == 0.0
    assert exact["relative_l2"] == 0.0
    drift = _comparison(reference, reference + 0.1)
    assert drift["max_abs"] > 0
    assert drift["relative_l2"] > 0


def test_cpu_primal_reference_matches_direct_lstsq() -> None:
    rng = np.random.default_rng(42)
    train = rng.normal(size=(32, 4))
    target = rng.normal(size=32)
    inference = rng.normal(size=(7, 4))
    feature_count = 32
    seed = 20260810
    train_prediction, prediction, diagnostics = _cpu_primal_svd_reference(
        train, target, inference, feature_count, seed
    )
    chunks = list(
        iter_nested_rff_parameters(4, feature_count, 4_096, 0.5, seed, np.dtype("float64"))
    )
    weights = np.concatenate([chunk[0] for chunk in chunks])
    phases = np.concatenate([chunk[1] for chunk in chunks])
    design = math.sqrt(2.0 / feature_count) * np.cos(train @ weights.T + phases)
    inference_design = math.sqrt(2.0 / feature_count) * np.cos(inference @ weights.T + phases)
    feature_mean = design.mean(axis=0)
    target_mean = target.mean()
    effective_rcond = math.sqrt(np.finfo(np.float64).eps * len(train))
    coefficient = np.linalg.lstsq(
        design - feature_mean, target - target_mean, rcond=effective_rcond
    )[0]
    expected_train = target_mean + (design - feature_mean) @ coefficient
    expected = target_mean + (inference_design - feature_mean) @ coefficient
    np.testing.assert_allclose(train_prediction, expected_train, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(prediction, expected, rtol=1e-9, atol=1e-9)
    singular_values = np.linalg.svd(design - feature_mean, compute_uv=False)
    assert diagnostics["rank"] == np.linalg.matrix_rank(
        design - feature_mean, tol=singular_values[0] * effective_rcond
    )


def test_assessment_separates_float64_curve_from_float32_stress() -> None:
    aggregates = []
    mse64 = {0.90: 2.0, 0.98: 5.0, 1.00: 8.0, 1.02: 6.0, 1.10: 3.0}
    mse32 = {0.90: 2.0, 0.98: 3.0, 1.00: 4.0, 1.02: 5.0, 1.10: 6.0}
    rows = []
    for ratio in PHASE19_RATIOS:
        aggregates.append(
            {
                "target_pn_ratio": ratio,
                "float64_oos_mse_median": mse64[ratio],
                "float32_oos_mse_median": mse32[ratio],
            }
        )
        for seed in PHASE19_SEEDS:
            rows.append(
                {
                    "target_pn_ratio": ratio,
                    "seed": seed,
                    "float64_repeat_oos_max_abs": 0.0,
                    "float32_vs_float64_oos_relative_l2": 0.2,
                    "cpu_rank": 2_159 if ratio == 1.0 else None,
                    "cpu_oos_relative_l2": 1e-8 if ratio == 1.0 else None,
                    "cpu_oos_max_abs": 1e-9 if ratio == 1.0 else None,
                }
            )
    assessment = assess_numerical_stability(rows, aggregates, zero_return_mse=1.0)
    assert assessment["local_curve"]["float64_local_double_descent"] is True
    assert assessment["local_curve"]["float64_peak_ratio"] == 1.0
    assert assessment["local_curve"]["float32_peak_matches_float64"] is False
    assert assessment["float32_is_materially_different"] is True
    assert assessment["any_precision_beats_zero_return_mse"] is False
