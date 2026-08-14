"""Phase 19: numerical-stability audit of the real interpolation region."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from research.double_descent.freqai.Phase4CudaRFFRegressor import _CudaWorkerClient
from research.double_descent.metrics import prediction_metrics
from research.double_descent.phase4 import (
    Phase4Config,
    build_freqtrade_config,
    feature_count_for_ratio,
)
from research.double_descent.phase10 import PHASE10_GAMMA, PHASE10_SEEDS
from research.double_descent.rff_cuda_worker import iter_nested_rff_parameters


PHASE19_PROTOCOL_VERSION = 1
PHASE19_RATIOS = (0.90, 0.98, 1.00, 1.02, 1.10)
PHASE19_SEEDS = PHASE10_SEEDS
PHASE19_ANCHOR_TIMERANGE = "20250401-20250501"
PHASE19_EFFECTIVE_N = 2_159
PHASE19_INPUT_FEATURES = 25
PHASE19_MINIMUM_INFERENCE_ROWS = 600
PHASE19_GAMMA = PHASE10_GAMMA
PHASE19_RCOND = 1e-12
PHASE19_RIDGE = 0.0
PHASE19_CHUNK_SIZE = 4_096
PHASE19_CPU_REFERENCE_RATIO = 1.0
PHASE19_FLOAT64_REPEAT_MAX_ABS = 5e-12
PHASE19_CPU_RELATIVE_L2_TOLERANCE = 1e-5
PHASE19_CPU_MAX_ABS_TOLERANCE = 1e-5
PHASE19_CPU_RANK_DELTA_TOLERANCE = 1
PHASE19_EXPECTED_CASES = len(PHASE19_RATIOS) * len(PHASE19_SEEDS)
PHASE19_HOLDOUT_START = "20260101"


@dataclass(frozen=True)
class Phase19Config:
    data_directory: Path
    phase10_summary: Path = Path("user_data/research_results/double_descent/phase10/summary.json")
    phase18_summary: Path = Path("user_data/research_results/double_descent/phase18/summary.json")
    output_directory: Path = Path("user_data/research_results/double_descent/phase19")
    project_root: Path = Path()
    python_executable: str = "python"
    cuda_python_executable: str = ""
    subprocess_timeout_seconds: int = 3_600
    resume: bool = True
    strategy_directory: Path = Path("research/double_descent/freqai")
    model_directory: Path = Path("research/double_descent/freqai")
    models_directory: Path = Path("user_data/models")

    def validate(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self.data_directory.is_dir():
            raise FileNotFoundError(f"data directory does not exist: {self.data_directory}")
        if not Path(self.python_executable).is_file():
            raise FileNotFoundError(f"Freqtrade Python does not exist: {self.python_executable}")
        if not self.phase10_summary.is_file() or not self.phase18_summary.is_file():
            raise FileNotFoundError("Phase 19 requires completed Phase 10 and Phase 18 summaries")
        phase10 = json.loads(self.phase10_summary.read_text(encoding="utf-8"))
        phase18 = json.loads(self.phase18_summary.read_text(encoding="utf-8"))
        if phase10.get("phase") != 10 or phase10.get("gate", {}).get("passed") is not True:
            raise ValueError("Phase 10 did not pass its complete gate")
        if phase18.get("phase") != 18 or phase18.get("gate", {}).get("passed") is not True:
            raise ValueError("Phase 18 did not pass its complete gate")
        source = phase10.get("config", {})
        frozen_checks = {
            "effective_n": source.get("effective_n") == PHASE19_EFFECTIVE_N,
            "gamma": source.get("gamma") == PHASE19_GAMMA,
            "ridge": source.get("ridge") == PHASE19_RIDGE,
            "rcond": source.get("rcond") == PHASE19_RCOND,
            "dtype": source.get("dtype") == "float64",
            "seeds": tuple(source.get("seeds", ())) == PHASE19_SEEDS,
            "holdout": source.get("holdout_start") == PHASE19_HOLDOUT_START,
            "phase10_holdout_sealed": phase10.get("design", {}).get("holdout_used") is False,
            "phase18_holdout_sealed": phase18.get("design", {}).get("holdout_used") is False,
        }
        if not all(frozen_checks.values()):
            failed = sorted(key for key, value in frozen_checks.items() if not value)
            raise ValueError(f"Phase 19 frozen source protocol mismatch: {failed}")
        cuda_python = self.resolved_cuda_python(phase10)
        if not cuda_python.is_file():
            raise FileNotFoundError(f"CUDA Python does not exist: {cuda_python}")
        if tuple(PHASE19_RATIOS) != (0.90, 0.98, 1.00, 1.02, 1.10):
            raise ValueError("Phase 19 critical P/N grid is frozen")
        if PHASE19_ANCHOR_TIMERANGE.split("-")[1] > PHASE19_HOLDOUT_START:
            raise ValueError("Phase 19 anchor may not enter the sealed holdout")
        return phase10, phase18

    def resolved_cuda_python(self, phase10: dict[str, Any] | None = None) -> Path:
        if self.cuda_python_executable:
            return Path(self.cuda_python_executable).resolve()
        if phase10 is None:
            phase10 = json.loads(self.phase10_summary.read_text(encoding="utf-8"))
        return Path(phase10["config"]["cuda_python_executable"]).resolve()


@dataclass(frozen=True)
class Phase19Case:
    case_id: str
    seed: int
    target_pn_ratio: float
    feature_count: int


def phase19_cases() -> list[Phase19Case]:
    return [
        Phase19Case(
            case_id=f"seed-{seed}-pn-{ratio:.5f}",
            seed=seed,
            target_pn_ratio=ratio,
            feature_count=feature_count_for_ratio(ratio, PHASE19_EFFECTIVE_N),
        )
        for seed in PHASE19_SEEDS
        for ratio in PHASE19_RATIOS
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_fingerprint(config: Phase19Config, extra_paths: tuple[Path, ...] = ()) -> str:
    paths = (
        config.project_root / "research/double_descent/phase19.py",
        config.model_directory / "Phase19CaptureRegressor.py",
        config.model_directory / "Phase4CudaRFFRegressor.py",
        config.strategy_directory / "Phase3EffectiveNStrategy.py",
        config.strategy_directory / "Phase4RFFStrategy.py",
        config.project_root / "research/double_descent/rff_cuda_worker.py",
        config.phase10_summary,
        config.phase18_summary,
        *extra_paths,
    )
    digest = hashlib.sha256(f"phase19:{PHASE19_PROTOCOL_VERSION}".encode())
    for path in paths:
        resolved = path.resolve()
        digest.update(resolved.as_posix().encode())
        digest.update(resolved.read_bytes())
    return digest.hexdigest()


def prepare_phase19(config: Phase19Config) -> dict[str, Any]:
    phase10, phase18 = config.validate()
    config.output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = config.output_directory / "case_manifest.csv"
    preparation_path = config.output_directory / "preparation.json"
    cases = phase19_cases()
    _write_csv(manifest_path, [asdict(case) for case in cases])
    checks = {
        "phase10_complete_gate_passed": phase10["gate"]["passed"] is True,
        "phase18_complete_gate_passed": phase18["gate"]["passed"] is True,
        "critical_grid_is_predeclared": tuple(PHASE19_RATIOS) == (0.90, 0.98, 1.00, 1.02, 1.10),
        "three_frozen_seeds": len(PHASE19_SEEDS) == 3,
        "complete_case_count": len(cases) == PHASE19_EXPECTED_CASES,
        "anchor_ends_before_holdout": PHASE19_ANCHOR_TIMERANGE.split("-")[1]
        <= PHASE19_HOLDOUT_START,
        "prediction_or_trading_results_did_not_select_protocol": True,
    }
    summary = {
        "phase": 19,
        "stage": "preparation",
        "protocol_version": PHASE19_PROTOCOL_VERSION,
        "objective": "Separate numerical instability from statistical double descent",
        "design": {
            "anchor_timerange": PHASE19_ANCHOR_TIMERANGE,
            "effective_n": PHASE19_EFFECTIVE_N,
            "input_feature_count": PHASE19_INPUT_FEATURES,
            "critical_pn_ratios": list(PHASE19_RATIOS),
            "seeds": list(PHASE19_SEEDS),
            "gamma": PHASE19_GAMMA,
            "ridge": PHASE19_RIDGE,
            "requested_rcond": PHASE19_RCOND,
            "cuda_precisions": ["float64", "float32"],
            "float64_repeats_per_case": 2,
            "independent_cpu_solver": "SciPy gesdd primal SVD with matched spectral floor",
            "cpu_reference_ratio": PHASE19_CPU_REFERENCE_RATIO,
            "selection_used_prediction_or_trading_results": False,
            "trading_used_for_inference": False,
            "holdout_used": False,
        },
        "gate": {"passed": all(checks.values()), "checks": checks},
        "artifacts": {
            "preparation": str(preparation_path),
            "case_manifest": str(manifest_path),
        },
    }
    preparation_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _safe_clean_model_directory(config: Phase19Config, identifier: str) -> None:
    root = (config.project_root / config.models_directory).resolve()
    target = (root / identifier).resolve()
    if target.parent != root or not identifier.startswith("double-descent-phase19-"):
        raise ValueError("refusing to remove a non-Phase19 model directory")
    if target.exists():
        shutil.rmtree(target)


def _capture_config(config: Phase19Config, phase10: dict[str, Any]) -> dict[str, Any]:
    capture_directory = config.output_directory / "capture"
    phase4 = Phase4Config(
        data_directory=config.data_directory,
        output_directory=capture_directory,
        python_executable=config.python_executable,
        cuda_python_executable=str(config.resolved_cuda_python(phase10)),
        timerange=PHASE19_ANCHOR_TIMERANGE,
        gamma=PHASE19_GAMMA,
        ratios=(1.0,),
    )
    identifier = "double-descent-phase19-capture-v1"
    generated = build_freqtrade_config(
        phase4,
        PHASE19_EFFECTIVE_N,
        identifier,
        capture_directory / "capture_unused_metrics.jsonl",
        "phase19-capture",
    )
    generated["backtest_cache"] = "none"
    generated["freqai"]["model_training_parameters"].update(
        {
            "phase19_capture_directory": str(capture_directory.resolve()),
            "phase19_protocol_version": PHASE19_PROTOCOL_VERSION,
        }
    )
    return generated


def _capture_command(config: Phase19Config, config_path: Path, export_directory: Path) -> list[str]:
    return [
        config.python_executable,
        "-m",
        "freqtrade",
        "backtesting",
        "--config",
        str(config_path),
        "--strategy",
        "Phase4RFFStrategy",
        "--strategy-path",
        str(config.strategy_directory),
        "--freqaimodel",
        "Phase19CaptureRegressor",
        "--freqaimodel-path",
        str(config.model_directory),
        "--timerange",
        PHASE19_ANCHOR_TIMERANGE,
        "--timeframe",
        "1h",
        "--pairs",
        "BTC/USDT:USDT",
        "--datadir",
        str(config.data_directory),
        "--cache",
        "none",
        "--export",
        "trades",
        "--export-directory",
        str(export_directory),
        "--no-color",
    ]


def _audit_capture_window(window_directory: Path) -> dict[str, Any]:
    metadata_path = window_directory / "metadata.json"
    train_path = window_directory / "train.npz"
    inference_path = window_directory / "inference.npz"
    if not (metadata_path.is_file() and train_path.is_file() and inference_path.is_file()):
        raise FileNotFoundError(f"incomplete capture window: {window_directory}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(train_path, allow_pickle=False) as train:
        train_features = train["features"]
        train_target = train["target_scaled"]
        train_dates = train["dates_ns"]
    with np.load(inference_path, allow_pickle=False) as inference:
        inference_features = inference["features"]
        inference_target = inference["target_raw"]
        inference_dates = inference["dates_ns"]
        do_predict = inference["do_predict"]
    start_text, end_text = PHASE19_ANCHOR_TIMERANGE.split("-")
    anchor_start = np.datetime64(f"{start_text[:4]}-{start_text[4:6]}-{start_text[6:]}", "ns")
    anchor_end = np.datetime64(f"{end_text[:4]}-{end_text[4:6]}-{end_text[6:]}", "ns")
    inference_datetime = inference_dates.astype("datetime64[ns]")
    checks = {
        "exact_effective_n": train_features.shape[0] == PHASE19_EFFECTIVE_N,
        "exact_market_feature_count": train_features.shape[1] == PHASE19_INPUT_FEATURES,
        "train_arrays_aligned": len(train_features) == len(train_target) == len(train_dates),
        "minimum_inference_rows": len(inference_features) >= PHASE19_MINIMUM_INFERENCE_ROWS,
        "inference_feature_count_matches": inference_features.shape[1] == PHASE19_INPUT_FEATURES,
        "inference_arrays_aligned": len(inference_features)
        == len(inference_target)
        == len(inference_dates)
        == len(do_predict),
        "train_arrays_finite": np.isfinite(train_features).all()
        and np.isfinite(train_target).all(),
        "inference_features_finite": np.isfinite(inference_features).all(),
        "sufficient_finite_oos_targets": int(np.isfinite(inference_target).sum()) >= 500,
        "inference_inside_anchor": bool(
            inference_datetime.min() >= anchor_start and inference_datetime.max() < anchor_end
        ),
    }
    return {
        "window": window_directory.name,
        "passed": all(checks.values()),
        "checks": checks,
        "metadata": metadata,
        "train_rows": int(train_features.shape[0]),
        "inference_rows": int(inference_features.shape[0]),
        "finite_oos_targets": int(np.isfinite(inference_target).sum()),
        "do_predict_rows": int(np.asarray(do_predict, dtype=bool).sum()),
        "train_path": str(train_path),
        "inference_path": str(inference_path),
    }


def capture_anchor_window(config: Phase19Config) -> dict[str, Any]:
    phase10, _ = config.validate()
    prepare_phase19(config)
    capture_directory = config.output_directory / "capture"
    summary_path = capture_directory / "summary.json"
    fingerprint = _source_fingerprint(config)
    if config.resume and summary_path.is_file():
        cached = json.loads(summary_path.read_text(encoding="utf-8"))
        dataset = Path(cached.get("artifacts", {}).get("anchor_dataset", ""))
        if (
            cached.get("fingerprint") == fingerprint
            and cached.get("gate", {}).get("passed") is True
            and dataset.is_file()
        ):
            print("PHASE19 CAPTURE RECOVERED", flush=True)
            return cached
    if capture_directory.exists():
        shutil.rmtree(capture_directory)
    capture_directory.mkdir(parents=True, exist_ok=True)
    generated = _capture_config(config, phase10)
    identifier = generated["freqai"]["identifier"]
    _safe_clean_model_directory(config, identifier)
    config_path = capture_directory / "config.json"
    export_directory = capture_directory / "backtest"
    log_path = capture_directory / "process.log"
    config_path.write_text(json.dumps(generated, indent=2), encoding="utf-8")
    command = _capture_command(config, config_path, export_directory)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=config.project_root,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=config.subprocess_timeout_seconds,
    )
    wall_seconds = time.perf_counter() - started
    process_text = (
        f"COMMAND: {subprocess.list2cmdline(command)}\n\n{completed.stdout}"
        f"\n\nSTDERR:\n{completed.stderr}"
    )
    log_path.write_text(
        process_text,
        encoding="utf-8",
    )
    windows = sorted((capture_directory / "windows").glob("*"))
    audits = [_audit_capture_window(window) for window in windows if window.is_dir()]
    passing = [row for row in audits if row["passed"]]
    canonical_path = capture_directory / "anchor_dataset.npz"
    if completed.returncode == 0 and len(passing) == 1:
        selected = passing[0]
        with (
            np.load(selected["train_path"], allow_pickle=False) as train,
            np.load(selected["inference_path"], allow_pickle=False) as inference,
        ):
            np.savez_compressed(
                canonical_path,
                train_features=train["features"],
                train_target_scaled=train["target_scaled"],
                train_dates_ns=train["dates_ns"],
                inference_features=inference["features"],
                inference_target_raw=inference["target_raw"],
                inference_dates_ns=inference["dates_ns"],
                do_predict=inference["do_predict"],
                target_inverse_zero=np.array(
                    [selected["metadata"]["target_inverse_zero"]], dtype=np.float64
                ),
                target_inverse_one=np.array(
                    [selected["metadata"]["target_inverse_one"]], dtype=np.float64
                ),
            )
    checks = {
        "freqtrade_returncode_zero": completed.returncode == 0,
        "exactly_one_valid_anchor_window": len(passing) == 1 and len(audits) == 1,
        "anchor_dataset_written": canonical_path.is_file(),
        "holdout_was_not_used": True,
    }
    summary = {
        "phase": 19,
        "stage": "capture",
        "protocol_version": PHASE19_PROTOCOL_VERSION,
        "fingerprint": fingerprint,
        "gate": {"passed": all(checks.values()), "checks": checks},
        "window_audits": audits,
        "returncode": completed.returncode,
        "wall_seconds": wall_seconds,
        "stderr_tail": completed.stderr[-2_000:],
        "artifacts": {
            "summary": str(summary_path),
            "config": str(config_path),
            "process_log": str(log_path),
            "anchor_dataset": str(canonical_path),
        },
    }
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    _safe_clean_model_directory(config, identifier)
    if not summary["gate"]["passed"]:
        raise RuntimeError("Phase 19 anchor capture failed its integrity gate")
    return summary


def _load_anchor(capture_summary: dict[str, Any]) -> dict[str, np.ndarray]:
    path = Path(capture_summary["artifacts"]["anchor_dataset"])
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _raw_prediction(scaled: np.ndarray, anchor: dict[str, np.ndarray]) -> np.ndarray:
    zero = float(anchor["target_inverse_zero"][0])
    one = float(anchor["target_inverse_one"][0])
    return zero + np.asarray(scaled, dtype=np.float64) * (one - zero)


def _comparison(reference: np.ndarray, challenger: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float64)
    challenger = np.asarray(challenger, dtype=np.float64)
    difference = challenger - reference
    reference_norm = float(np.linalg.norm(reference))
    correlation = (
        float(np.corrcoef(reference, challenger)[0, 1])
        if reference.size > 1 and np.std(reference) > 0 and np.std(challenger) > 0
        else math.nan
    )
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "relative_l2": float(np.linalg.norm(difference) / max(reference_norm, 1e-30)),
        "correlation": correlation,
        "sign_agreement": float(np.mean(np.sign(reference) == np.sign(challenger))),
    }


def _cpu_primal_svd_reference(
    train_features: np.ndarray,
    train_target: np.ndarray,
    inference_features: np.ndarray,
    feature_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from scipy import linalg

    started = time.perf_counter()
    train = np.asarray(train_features, dtype=np.float64)
    inference = np.asarray(inference_features, dtype=np.float64)
    target = np.asarray(train_target, dtype=np.float64)
    explicit_train = np.empty((len(train), feature_count), dtype=np.float64)
    explicit_inference = np.empty((len(inference), feature_count), dtype=np.float64)
    offset = 0
    for weights, phases in iter_nested_rff_parameters(
        train.shape[1],
        feature_count,
        PHASE19_CHUNK_SIZE,
        PHASE19_GAMMA,
        seed,
        np.dtype("float64"),
    ):
        width = len(phases)
        scale = math.sqrt(2.0 / feature_count)
        explicit_train[:, offset : offset + width] = scale * np.cos(train @ weights.T + phases)
        explicit_inference[:, offset : offset + width] = scale * np.cos(
            inference @ weights.T + phases
        )
        offset += width
    feature_mean = explicit_train.mean(axis=0)
    centered_train = explicit_train - feature_mean
    centered_inference = explicit_inference - feature_mean
    target_mean = float(target.mean())
    centered_target = target - target_mean
    decomposition_started = time.perf_counter()
    left, singular_values, right_transpose = linalg.svd(
        centered_train,
        full_matrices=False,
        check_finite=False,
        lapack_driver="gesdd",
    )
    decomposition_seconds = time.perf_counter() - decomposition_started
    relative_eigenvalue_threshold = max(PHASE19_RCOND**2, np.finfo(np.float64).eps * len(train))
    relative_singular_threshold = math.sqrt(relative_eigenvalue_threshold)
    retained = singular_values > singular_values[0] * relative_singular_threshold
    retained_singular = singular_values[retained]
    coefficients = right_transpose[retained].T @ (
        (left[:, retained].T @ centered_target) / retained_singular
    )
    train_prediction = target_mean + centered_train @ coefficients
    inference_prediction = target_mean + centered_inference @ coefficients
    squared = retained_singular**2
    diagnostics = {
        "backend": "scipy_cpu",
        "solver": "primal_svd_gesdd",
        "dtype": "float64",
        "rank": int(retained_singular.size),
        "effective_rank": float(squared.sum() ** 2 / np.square(squared).sum()),
        "condition_number": float(retained_singular[0] / retained_singular[-1]),
        "relative_eigenvalue_threshold": relative_eigenvalue_threshold,
        "relative_singular_threshold": relative_singular_threshold,
        "scaled_train_mse": float(np.mean((train_prediction - target) ** 2)),
        "scaled_train_max_abs_error": float(np.max(np.abs(train_prediction - target))),
        "coefficient_norm": float(np.linalg.norm(coefficients)),
        "decomposition_seconds": decomposition_seconds,
        "total_seconds": time.perf_counter() - started,
    }
    return train_prediction, inference_prediction, diagnostics


def _prediction_metrics_for_anchor(
    prediction_scaled: np.ndarray, anchor: dict[str, np.ndarray]
) -> dict[str, Any]:
    prediction_raw = _raw_prediction(prediction_scaled, anchor)
    actual = np.asarray(anchor["inference_target_raw"], dtype=np.float64)
    allowed = np.asarray(anchor["do_predict"], dtype=bool) & np.isfinite(actual)
    return prediction_metrics(actual[allowed], prediction_raw[allowed])


def _zero_return_metrics(anchor: dict[str, np.ndarray]) -> dict[str, Any]:
    actual = np.asarray(anchor["inference_target_raw"], dtype=np.float64)
    allowed = np.asarray(anchor["do_predict"], dtype=bool) & np.isfinite(actual)
    return prediction_metrics(actual[allowed], np.zeros(int(allowed.sum()), dtype=np.float64))


def _case_fingerprint(config: Phase19Config, case: Phase19Case, dataset_path: Path) -> str:
    digest = hashlib.sha256(_source_fingerprint(config, (dataset_path,)).encode())
    digest.update(json.dumps(asdict(case), sort_keys=True).encode())
    return digest.hexdigest()


def _run_case(
    config: Phase19Config,
    case: Phase19Case,
    anchor: dict[str, np.ndarray],
    dataset_path: Path,
    client: _CudaWorkerClient,
) -> dict[str, Any]:
    case_directory = config.output_directory / "runs" / case.case_id
    case_directory.mkdir(parents=True, exist_ok=True)
    result_path = case_directory / "result.json"
    prediction_path = case_directory / "predictions.npz"
    fingerprint = _case_fingerprint(config, case, dataset_path)
    if config.resume and result_path.is_file() and prediction_path.is_file():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if cached.get("fingerprint") == fingerprint and cached.get("passed") is True:
            print(f"PHASE19 RECOVERED {case.case_id}", flush=True)
            return cached
    train_features = anchor["train_features"]
    train_target = anchor["train_target_scaled"]
    inference_features = anchor["inference_features"]
    common = {
        "feature_count": case.feature_count,
        "chunk_size": PHASE19_CHUNK_SIZE,
        "gamma": PHASE19_GAMMA,
        "seed": case.seed,
        "ridge": PHASE19_RIDGE,
        "rcond": PHASE19_RCOND,
    }
    print(f"PHASE19 START {case.case_id} P={case.feature_count}", flush=True)
    started = time.perf_counter()
    train64, diagnostics64 = client.fit(
        train_features, train_target, {**common, "dtype": "float64"}
    )
    prediction64 = client.predict(inference_features)
    train64_repeat, diagnostics64_repeat = client.fit(
        train_features, train_target, {**common, "dtype": "float64"}
    )
    prediction64_repeat = client.predict(inference_features)
    train32, diagnostics32 = client.fit(
        train_features, train_target, {**common, "dtype": "float32"}
    )
    prediction32 = client.predict(inference_features)
    repeat_train = _comparison(train64, train64_repeat)
    repeat_oos = _comparison(prediction64, prediction64_repeat)
    precision_train = _comparison(train64, train32)
    precision_oos = _comparison(prediction64, prediction32)
    cpu_train: np.ndarray | None = None
    cpu_prediction: np.ndarray | None = None
    cpu_diagnostics: dict[str, Any] | None = None
    cpu_train_comparison: dict[str, float] | None = None
    cpu_oos_comparison: dict[str, float] | None = None
    cpu_checks: dict[str, bool] | None = None
    if case.target_pn_ratio == PHASE19_CPU_REFERENCE_RATIO:
        cpu_train, cpu_prediction, cpu_diagnostics = _cpu_primal_svd_reference(
            train_features,
            train_target,
            inference_features,
            case.feature_count,
            case.seed,
        )
        cpu_train_comparison = _comparison(train64, cpu_train)
        cpu_oos_comparison = _comparison(prediction64, cpu_prediction)
        cpu_checks = {
            "rank_within_tolerance": abs(int(diagnostics64["rank"]) - int(cpu_diagnostics["rank"]))
            <= PHASE19_CPU_RANK_DELTA_TOLERANCE,
            "oos_relative_l2_within_tolerance": cpu_oos_comparison["relative_l2"]
            <= PHASE19_CPU_RELATIVE_L2_TOLERANCE,
            "oos_max_abs_within_tolerance": cpu_oos_comparison["max_abs"]
            <= PHASE19_CPU_MAX_ABS_TOLERANCE,
        }
    checks = {
        "float64_train_predictions_finite": bool(np.isfinite(train64).all()),
        "float64_oos_predictions_finite": bool(np.isfinite(prediction64).all()),
        "float32_train_predictions_finite": bool(np.isfinite(train32).all()),
        "float32_oos_predictions_finite": bool(np.isfinite(prediction32).all()),
        "float64_repeat_train_stable": repeat_train["max_abs"] <= PHASE19_FLOAT64_REPEAT_MAX_ABS,
        "float64_repeat_oos_stable": repeat_oos["max_abs"] <= PHASE19_FLOAT64_REPEAT_MAX_ABS,
        "float64_repeat_rank_identical": diagnostics64["rank"] == diagnostics64_repeat["rank"],
        "cpu_reference_passed_when_required": cpu_checks is None or all(cpu_checks.values()),
    }
    arrays = {
        "float64_train": train64,
        "float64_train_repeat": train64_repeat,
        "float64_oos": prediction64,
        "float64_oos_repeat": prediction64_repeat,
        "float32_train": train32,
        "float32_oos": prediction32,
    }
    if cpu_train is not None and cpu_prediction is not None:
        arrays["cpu_float64_train"] = cpu_train
        arrays["cpu_float64_oos"] = cpu_prediction
    np.savez_compressed(prediction_path, **arrays)
    result = {
        **asdict(case),
        "actual_pn_ratio": case.feature_count / len(train_features),
        "fingerprint": fingerprint,
        "passed": all(checks.values()),
        "checks": checks,
        "float64": {
            "diagnostics": diagnostics64,
            "repeat_diagnostics": diagnostics64_repeat,
            "oos_metrics": _prediction_metrics_for_anchor(prediction64, anchor),
        },
        "float32": {
            "diagnostics": diagnostics32,
            "oos_metrics": _prediction_metrics_for_anchor(prediction32, anchor),
        },
        "float64_repeat_train_comparison": repeat_train,
        "float64_repeat_oos_comparison": repeat_oos,
        "float32_vs_float64_train": precision_train,
        "float32_vs_float64_oos": precision_oos,
        "cpu_reference": (
            {
                "diagnostics": cpu_diagnostics,
                "checks": cpu_checks,
                "train_comparison_to_cuda_float64": cpu_train_comparison,
                "oos_comparison_to_cuda_float64": cpu_oos_comparison,
                "oos_metrics": _prediction_metrics_for_anchor(cpu_prediction, anchor),
            }
            if cpu_prediction is not None
            else None
        ),
        "wall_seconds": time.perf_counter() - started,
        "artifacts": {"predictions": str(prediction_path), "result": str(result_path)},
    }
    result_path.write_text(json.dumps(_json_safe(result), indent=2), encoding="utf-8")
    print(
        f"PHASE19 DONE {case.case_id} passed={result['passed']} "
        f"rank64={diagnostics64['rank']} rank32={diagnostics32['rank']} "
        f"seconds={result['wall_seconds']:.1f}",
        flush=True,
    )
    return _json_safe(result)


def _flat_case_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        cpu = result.get("cpu_reference")
        rows.append(
            {
                "case_id": result["case_id"],
                "seed": result["seed"],
                "target_pn_ratio": result["target_pn_ratio"],
                "actual_pn_ratio": result["actual_pn_ratio"],
                "feature_count": result["feature_count"],
                "passed": result["passed"],
                "float64_rank": result["float64"]["diagnostics"]["rank"],
                "float32_rank": result["float32"]["diagnostics"]["rank"],
                "float64_effective_rank": result["float64"]["diagnostics"]["effective_rank"],
                "float32_effective_rank": result["float32"]["diagnostics"]["effective_rank"],
                "float64_condition_number": result["float64"]["diagnostics"]["condition_number"],
                "float32_condition_number": result["float32"]["diagnostics"]["condition_number"],
                "float64_oos_mse": result["float64"]["oos_metrics"]["mse"],
                "float32_oos_mse": result["float32"]["oos_metrics"]["mse"],
                "float64_repeat_oos_max_abs": result["float64_repeat_oos_comparison"]["max_abs"],
                "float32_vs_float64_oos_relative_l2": result["float32_vs_float64_oos"][
                    "relative_l2"
                ],
                "float32_vs_float64_oos_sign_agreement": result["float32_vs_float64_oos"][
                    "sign_agreement"
                ],
                "cpu_rank": cpu["diagnostics"]["rank"] if cpu else None,
                "cpu_oos_relative_l2": (
                    cpu["oos_comparison_to_cuda_float64"]["relative_l2"] if cpu else None
                ),
                "cpu_oos_max_abs": (
                    cpu["oos_comparison_to_cuda_float64"]["max_abs"] if cpu else None
                ),
                "wall_seconds": result["wall_seconds"],
            }
        )
    return rows


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates = []
    for ratio in PHASE19_RATIOS:
        group = [row for row in rows if row["target_pn_ratio"] == ratio]
        aggregates.append(
            {
                "target_pn_ratio": ratio,
                "feature_count": group[0]["feature_count"],
                "seed_count": len(group),
                "float64_oos_mse_median": statistics.median(
                    row["float64_oos_mse"] for row in group
                ),
                "float32_oos_mse_median": statistics.median(
                    row["float32_oos_mse"] for row in group
                ),
                "float64_condition_number_median": statistics.median(
                    row["float64_condition_number"] for row in group
                ),
                "float32_condition_number_median": statistics.median(
                    row["float32_condition_number"] for row in group
                ),
                "float64_effective_rank_median": statistics.median(
                    row["float64_effective_rank"] for row in group
                ),
                "float32_effective_rank_median": statistics.median(
                    row["float32_effective_rank"] for row in group
                ),
                "float64_rank_median": statistics.median(row["float64_rank"] for row in group),
                "float32_rank_median": statistics.median(row["float32_rank"] for row in group),
                "precision_relative_l2_median": statistics.median(
                    row["float32_vs_float64_oos_relative_l2"] for row in group
                ),
                "precision_sign_agreement_median": statistics.median(
                    row["float32_vs_float64_oos_sign_agreement"] for row in group
                ),
            }
        )
    return aggregates


def assess_numerical_stability(
    rows: list[dict[str, Any]], aggregates: list[dict[str, Any]], zero_return_mse: float
) -> dict[str, Any]:
    mse64 = {row["target_pn_ratio"]: row["float64_oos_mse_median"] for row in aggregates}
    mse32 = {row["target_pn_ratio"]: row["float32_oos_mse_median"] for row in aggregates}
    peak64 = max(mse64, key=mse64.get)
    peak32 = max(mse32, key=mse32.get)
    allowed_peak = {0.98, 1.0, 1.02}
    local_curve = {
        "float64_peak_ratio": peak64,
        "float32_peak_ratio": peak32,
        "float64_peak_near_interpolation": peak64 in allowed_peak,
        "float64_left_side_below_peak": mse64[0.90] < mse64[peak64],
        "float64_right_side_recovers_by_1_10": mse64[1.10] < mse64[peak64],
        "float64_local_double_descent": peak64 in allowed_peak
        and mse64[0.90] < mse64[peak64]
        and mse64[1.10] < mse64[peak64],
        "float32_peak_matches_float64": peak32 == peak64,
    }
    log64 = np.log([mse64[ratio] for ratio in PHASE19_RATIOS])
    log32 = np.log([mse32[ratio] for ratio in PHASE19_RATIOS])
    precision_curve_correlation = float(np.corrcoef(log64, log32)[0, 1])
    cpu_rows = [row for row in rows if row["cpu_rank"] is not None]
    maximum_repeat_difference = max(row["float64_repeat_oos_max_abs"] for row in rows)
    maximum_cpu_relative_l2 = max(row["cpu_oos_relative_l2"] for row in cpu_rows)
    maximum_cpu_max_abs = max(row["cpu_oos_max_abs"] for row in cpu_rows)
    median_precision_relative_l2 = statistics.median(
        row["float32_vs_float64_oos_relative_l2"] for row in rows
    )
    best_float64_mse = min(mse64.values())
    best_float32_mse = min(mse32.values())
    return {
        "local_curve": local_curve,
        "zero_return_oos_mse": zero_return_mse,
        "best_float64_oos_mse": best_float64_mse,
        "best_float32_oos_mse": best_float32_mse,
        "best_float64_mse_over_zero": best_float64_mse / zero_return_mse,
        "best_float32_mse_over_zero": best_float32_mse / zero_return_mse,
        "any_precision_beats_zero_return_mse": min(best_float64_mse, best_float32_mse)
        < zero_return_mse,
        "float32_vs_float64_log_mse_correlation": precision_curve_correlation,
        "maximum_float64_repeat_oos_max_abs": maximum_repeat_difference,
        "maximum_cpu_oos_relative_l2": maximum_cpu_relative_l2,
        "maximum_cpu_oos_max_abs": maximum_cpu_max_abs,
        "all_cpu_references_pass_relative_l2_tolerance": maximum_cpu_relative_l2
        <= PHASE19_CPU_RELATIVE_L2_TOLERANCE,
        "all_cpu_references_pass_strict_max_abs_tolerance": maximum_cpu_max_abs
        <= PHASE19_CPU_MAX_ABS_TOLERANCE,
        "median_float32_vs_float64_oos_relative_l2": median_precision_relative_l2,
        "float32_is_materially_different": median_precision_relative_l2 > 1e-3 or peak32 != peak64,
        "interpretation": (
            "Float64 repeatability and independent CPU agreement separate deterministic "
            "statistical interpolation behavior from backend nondeterminism. Float32 is a "
            "precision stress control and is not allowed to redefine the frozen float64 result."
        ),
    }


def _write_plot(path: Path, aggregates: list[dict[str, Any]]) -> None:
    from plotly import graph_objects as go
    from plotly.subplots import make_subplots

    ratios = [row["target_pn_ratio"] for row in aggregates]
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("OOS MSE", "Condition number", "Retained rank / N", "Precision drift"),
    )
    for dtype, color in (("float64", "#2563eb"), ("float32", "#dc2626")):
        figure.add_trace(
            go.Scatter(
                x=ratios,
                y=[row[f"{dtype}_oos_mse_median"] for row in aggregates],
                name=f"{dtype} MSE",
                mode="lines+markers",
                line={"color": color},
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=ratios,
                y=[row[f"{dtype}_condition_number_median"] for row in aggregates],
                name=f"{dtype} condition",
                mode="lines+markers",
                line={"color": color, "dash": "dot"},
            ),
            row=1,
            col=2,
        )
        figure.add_trace(
            go.Scatter(
                x=ratios,
                y=[row[f"{dtype}_rank_median"] / PHASE19_EFFECTIVE_N for row in aggregates],
                name=f"{dtype} rank/N",
                mode="lines+markers",
                line={"color": color},
            ),
            row=2,
            col=1,
        )
    figure.add_trace(
        go.Scatter(
            x=ratios,
            y=[row["precision_relative_l2_median"] for row in aggregates],
            name="float32 vs float64 relative L2",
            mode="lines+markers",
            line={"color": "#7c3aed"},
        ),
        row=2,
        col=2,
    )
    figure.update_yaxes(type="log", row=1, col=1)
    figure.update_yaxes(type="log", row=1, col=2)
    figure.update_yaxes(type="log", row=2, col=2)
    figure.update_xaxes(title_text="P/N")
    figure.update_layout(
        title="Phase 19 — numerical stability around interpolation",
        template="plotly_white",
        height=820,
    )
    path.write_text(figure.to_html(include_plotlyjs="cdn"), encoding="utf-8")


def run_phase19(  # noqa: C901
    config: Phase19Config, selected_case_ids: tuple[str, ...] = ()
) -> dict[str, Any]:
    preparation = prepare_phase19(config)
    capture = capture_anchor_window(config)
    if not preparation["gate"]["passed"] or not capture["gate"]["passed"]:
        raise ValueError("Phase 19 preparation or capture did not pass")
    dataset_path = Path(capture["artifacts"]["anchor_dataset"])
    anchor = _load_anchor(capture)
    cases = phase19_cases()
    requested = set(selected_case_ids)
    unknown = requested - {case.case_id for case in cases}
    if unknown:
        raise ValueError(f"unknown Phase 19 case ids: {sorted(unknown)}")
    to_run = [case for case in cases if not requested or case.case_id in requested]
    results_by_id: dict[str, dict[str, Any]] = {}
    client: _CudaWorkerClient | None = None
    worker_log = config.output_directory / "cuda_worker.log"
    checkpoint_path = config.output_directory / "checkpoint.json"
    try:
        for case in to_run:
            result_path = config.output_directory / "runs" / case.case_id / "result.json"
            prediction_path = result_path.with_name("predictions.npz")
            fingerprint = _case_fingerprint(config, case, dataset_path)
            cached: dict[str, Any] | None = None
            if config.resume and result_path.is_file() and prediction_path.is_file():
                candidate = json.loads(result_path.read_text(encoding="utf-8"))
                if candidate.get("fingerprint") == fingerprint and candidate.get("passed") is True:
                    cached = candidate
            if cached is not None:
                print(f"PHASE19 RECOVERED {case.case_id}", flush=True)
                result = cached
            else:
                if client is None:
                    client = _CudaWorkerClient(
                        str(config.resolved_cuda_python()),
                        worker_log,
                        temporary_prefix="phase19-rff-",
                    )
                result = _run_case(config, case, anchor, dataset_path, client)
            results_by_id[case.case_id] = result
            checkpoint_path.write_text(
                json.dumps(
                    {
                        "phase": 19,
                        "completed_case_ids": list(results_by_id),
                        "passed_case_ids": [
                            case_id for case_id, row in results_by_id.items() if row["passed"]
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    finally:
        if client is not None:
            client.close()
    if not requested:
        for case in cases:
            if case.case_id in results_by_id:
                continue
            result_path = config.output_directory / "runs" / case.case_id / "result.json"
            if result_path.is_file():
                results_by_id[case.case_id] = json.loads(result_path.read_text(encoding="utf-8"))
    complete_results = [
        results_by_id[case.case_id] for case in cases if case.case_id in results_by_id
    ]
    rows = _flat_case_rows(complete_results)
    aggregates = _aggregate_rows(rows) if len(rows) == PHASE19_EXPECTED_CASES else []
    zero_metrics = _zero_return_metrics(anchor)
    assessment = (
        assess_numerical_stability(rows, aggregates, zero_metrics["mse"]) if aggregates else None
    )
    cpu_rows = [row for row in complete_results if row.get("cpu_reference") is not None]
    full_invocation = not requested
    checks = {
        "preparation_passed": preparation["gate"]["passed"] is True,
        "capture_passed": capture["gate"]["passed"] is True,
        "complete_critical_case_count": len(complete_results) == PHASE19_EXPECTED_CASES,
        "all_cases_passed": len(complete_results) == PHASE19_EXPECTED_CASES
        and all(row["passed"] for row in complete_results),
        "all_float64_repeats_stable": len(complete_results) == PHASE19_EXPECTED_CASES
        and all(
            row["checks"]["float64_repeat_train_stable"]
            and row["checks"]["float64_repeat_oos_stable"]
            and row["checks"]["float64_repeat_rank_identical"]
            for row in complete_results
        ),
        "three_independent_cpu_references_passed": len(cpu_rows) == len(PHASE19_SEEDS)
        and all(all(row["cpu_reference"]["checks"].values()) for row in cpu_rows),
        "all_precision_controls_finite": len(complete_results) == PHASE19_EXPECTED_CASES
        and all(
            row["checks"]["float32_train_predictions_finite"]
            and row["checks"]["float32_oos_predictions_finite"]
            for row in complete_results
        ),
        "holdout_was_not_used": True,
        "full_unfiltered_protocol_executed": full_invocation,
    }
    cases_path = config.output_directory / "case_results.csv"
    aggregates_path = config.output_directory / "aggregate_metrics.csv"
    plot_path = config.output_directory / "numerical_stability.html"
    summary_path = config.output_directory / "summary.json"
    _write_csv(cases_path, rows)
    _write_csv(aggregates_path, aggregates)
    if aggregates:
        _write_plot(plot_path, aggregates)
    summary = {
        "phase": 19,
        "stage": (
            "complete"
            if all(checks.values())
            else (
                "complete_gate_failed"
                if len(complete_results) == PHASE19_EXPECTED_CASES
                else "partial"
            )
        ),
        "protocol_version": PHASE19_PROTOCOL_VERSION,
        "objective": "Numerical stability around the real interpolation threshold",
        "design": preparation["design"],
        "preparation_gate": preparation["gate"],
        "capture_gate": capture["gate"],
        "gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "expected_case_count": PHASE19_EXPECTED_CASES,
            "observed_case_count": len(complete_results),
        },
        "assessment": assessment,
        "zero_return_baseline": zero_metrics,
        "case_results": complete_results,
        "aggregates": aggregates,
        "limitations": {
            "single_anchor_window": (
                "Solver comparisons use one predeclared genuine FreqAI window; Phase 10 supplies "
                "the full-year rolling float64 curve."
            ),
            "cpu_scope": (
                "The independent full-N CPU primal SVD is restricted to P/N=1 for all three "
                "seeds because that is the most numerically critical and expensive point."
            ),
            "float32_role": (
                "Float32 is a stress control. Its machine-epsilon spectral floor differs from "
                "float64 and it is not a candidate allowed to replace the frozen primary result."
            ),
            "economic_scope": "No trading decision or economic metric selects or gates Phase 19.",
        },
        "artifacts": {
            "summary": str(summary_path),
            "preparation": preparation["artifacts"]["preparation"],
            "case_manifest": preparation["artifacts"]["case_manifest"],
            "capture_summary": capture["artifacts"]["summary"],
            "anchor_dataset": capture["artifacts"]["anchor_dataset"],
            "case_results": str(cases_path),
            "aggregate_metrics": str(aggregates_path),
            "plot": str(plot_path),
            "checkpoint": str(checkpoint_path),
            "cuda_worker_log": str(worker_log),
        },
    }
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    return _json_safe(summary)
