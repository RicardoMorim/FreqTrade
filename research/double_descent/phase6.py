"""Phase 6: exact RBF-kernel limit and explicit RFF convergence study."""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from research.double_descent.phase3 import Phase3Config, audit_data_coverage
from research.double_descent.phase4 import (
    Phase4Config,
    _aggregate_training_diagnostics,
    _extract_trading_metrics,
    _json_safe,
    _load_evaluation_market_data,
    _parse_timerange,
    _read_predictions,
    _read_training_records,
    build_freqtrade_config,
    evaluate_oos_predictions,
    run_phase4,
)
from research.double_descent.phase5 import PHASE5_SEEDS, summarize_values


REFERENCE_FEATURE_COUNTS = (10_795, 53_975, 107_950)
EXTENSION_FEATURE_COUNTS = (250_000, 500_000, 1_000_000)
PRACTICAL_RELATIVE_PREDICTION_ERROR = 0.10
PRACTICAL_RELATIVE_OOS_MSE_GAP = 0.10

CONVERGENCE_METRICS = (
    "prediction_rmse_to_kernel",
    "prediction_mae_to_kernel",
    "prediction_relative_l2_error",
    "prediction_correlation_with_kernel",
    "prediction_sign_agreement_with_kernel",
    "oos_mse",
    "oos_r2",
    "oos_information_coefficient",
    "oos_directional_accuracy",
    "train_train_mse_mean",
    "train_effective_rank_mean",
    "train_condition_number_maximum",
    "train_training_seconds_total",
    "train_peak_vram_mib",
    "wall_seconds",
    "trading_sharpe",
    "trading_profit_factor",
    "trading_profit_total",
)


@dataclass(frozen=True)
class Phase6Config:
    data_directory: Path
    output_directory: Path = Path("user_data/research_results/double_descent/phase6")
    phase5_summary: Path = Path("user_data/research_results/double_descent/phase5/summary.json")
    python_executable: str = sys.executable
    cuda_python_executable: str = ""
    pair: str = "BTC/USDT:USDT"
    timeframe: str = "1h"
    timerange: str = "20250101-20260101"
    holdout_start: str = "20260101"
    train_period_days: int = 90
    backtest_period_days: int = 30
    effective_n: int = 2_159
    seeds: tuple[int, ...] = PHASE5_SEEDS
    reference_feature_counts: tuple[int, ...] = REFERENCE_FEATURE_COUNTS
    extension_feature_counts: tuple[int, ...] = EXTENSION_FEATURE_COUNTS
    gamma: float = 0.2
    ridge: float = 0.0
    rcond: float = 1e-12
    dtype: str = "float64"
    chunk_size: int = 4_096
    fee: float = 0.001
    minimum_training_windows: int = 10
    subprocess_timeout_seconds: int = 1_800
    interpolation_mse_tolerance: float = 1e-16
    resume: bool = True
    strategy_directory: Path = Path("research/double_descent/freqai")
    model_directory: Path = Path("research/double_descent/freqai")
    models_directory: Path = Path("user_data/models")

    def validate(self) -> None:
        if not self.phase5_summary.is_file():
            raise FileNotFoundError(f"Phase 5 summary does not exist: {self.phase5_summary}")
        if len(self.seeds) != 5 or len(set(self.seeds)) != 5:
            raise ValueError("Phase 6 is frozen to the five predeclared Phase 5 seeds")
        if self.gamma <= 0 or self.ridge < 0 or self.rcond <= 0:
            raise ValueError("gamma/rcond must be positive and ridge must be non-negative")
        if self.dtype != "float64":
            raise ValueError("Phase 6 kernel-limit comparisons require float64")
        feature_counts = (*self.reference_feature_counts, *self.extension_feature_counts)
        if tuple(sorted(set(feature_counts))) != feature_counts:
            raise ValueError("kernel convergence feature counts must be sorted and unique")
        if self.reference_feature_counts != REFERENCE_FEATURE_COUNTS:
            raise ValueError("Phase 6 reference feature counts are frozen to Phase 5 artifacts")
        if any(count <= self.effective_n for count in feature_counts):
            raise ValueError("kernel convergence points must be overparameterized")
        if not 0 <= self.fee < 0.1:
            raise ValueError("fee must be in [0, 0.1)")
        _phase4_config(self, self.seeds[0], self.output_directory / "validation").validate()


def _extension_ratios(config: Phase6Config) -> tuple[float, ...]:
    return tuple(count / config.effective_n for count in config.extension_feature_counts)


def _phase4_config(config: Phase6Config, seed: int, output_directory: Path) -> Phase4Config:
    return Phase4Config(
        data_directory=config.data_directory,
        output_directory=output_directory,
        python_executable=config.python_executable,
        cuda_python_executable=config.cuda_python_executable,
        pair=config.pair,
        timeframe=config.timeframe,
        timerange=config.timerange,
        holdout_start=config.holdout_start,
        train_period_days=config.train_period_days,
        backtest_period_days=config.backtest_period_days,
        effective_n=config.effective_n,
        ratios=_extension_ratios(config),
        seed=seed,
        gamma=config.gamma,
        ridge=config.ridge,
        rcond=config.rcond,
        dtype=config.dtype,
        chunk_size=config.chunk_size,
        fee=config.fee,
        minimum_training_windows=config.minimum_training_windows,
        subprocess_timeout_seconds=config.subprocess_timeout_seconds,
        interpolation_mse_tolerance=config.interpolation_mse_tolerance,
        resume=config.resume,
        strategy_directory=config.strategy_directory,
        model_directory=config.model_directory,
        models_directory=config.models_directory,
    )


def load_compatible_phase5_summary(config: Phase6Config) -> dict[str, Any]:
    summary = json.loads(config.phase5_summary.read_text(encoding="utf-8"))
    observed = summary.get("config", {})
    expected = {
        "pair": config.pair,
        "timeframe": config.timeframe,
        "timerange": config.timerange,
        "holdout_start": config.holdout_start,
        "train_period_days": config.train_period_days,
        "backtest_period_days": config.backtest_period_days,
        "effective_n": config.effective_n,
        "gamma": config.gamma,
        "ridge": config.ridge,
        "rcond": config.rcond,
        "dtype": config.dtype,
        "fee": config.fee,
    }
    compatible = (
        summary.get("phase") == 5
        and summary.get("gate", {}).get("passed")
        and tuple(observed.get("seeds", ())) == config.seeds
        and all(observed.get(key) == value for key, value in expected.items())
    )
    try:
        compatible = compatible and (
            Path(observed["data_directory"]).resolve() == config.data_directory.resolve()
        )
    except (KeyError, TypeError):
        compatible = False
    if not compatible:
        raise ValueError("Phase 5 summary is absent, failed, or incompatible with Phase 6")
    rows_path = config.phase5_summary.parent / "seed_results.csv"
    if not rows_path.is_file():
        raise FileNotFoundError(f"Phase 5 seed results do not exist: {rows_path}")
    rows = pd.read_csv(rows_path)
    selected = rows.loc[rows["feature_count"].isin(config.reference_feature_counts)]
    expected_cases = len(config.seeds) * len(config.reference_feature_counts)
    if (
        len(selected) != expected_cases
        or set(selected["seed"].astype(int)) != set(config.seeds)
        or set(selected["feature_count"].astype(int)) != set(config.reference_feature_counts)
    ):
        raise ValueError("Phase 5 does not contain the complete frozen kernel reference grid")
    return summary


def build_kernel_freqtrade_config(
    config: Phase6Config,
    identifier: str,
    metrics_path: Path,
    run_id: str,
) -> dict[str, Any]:
    base_config = _phase4_config(config, config.seeds[0], config.output_directory)
    generated = build_freqtrade_config(
        base_config,
        feature_count=config.effective_n,
        identifier=identifier,
        metrics_path=metrics_path,
        run_id=run_id,
    )
    generated["freqai"]["model_training_parameters"] = {
        "phase6_metrics_path": str(metrics_path.resolve()),
        "phase6_run_id": run_id,
        "cuda_python_executable": str(Path(config.cuda_python_executable).resolve()),
        "gamma": config.gamma,
        "ridge": config.ridge,
        "rcond": config.rcond,
        "dtype": config.dtype,
    }
    return generated


def _finalize_kernel_case(
    config: Phase6Config,
    identifier: str,
    config_path: Path,
    log_path: Path,
    metrics_path: Path,
    backtest_directory: Path,
    market_data: pd.DataFrame,
    returncode: int | None,
    timed_out: bool,
    wall_seconds: float,
    stderr: str,
    recovered: bool = False,
) -> dict[str, Any]:
    training_records = _read_training_records(metrics_path)
    training = _aggregate_training_diagnostics(training_records, config.interpolation_mse_tolerance)
    prediction_directory = config.models_directory / identifier / "backtesting_predictions"
    predictions = _read_predictions(prediction_directory)
    try:
        oos = evaluate_oos_predictions(predictions, market_data, config.timerange)
    except (KeyError, ValueError) as exc:
        oos = {"error": str(exc)}
    backtest_files = sorted(backtest_directory.glob("*.zip"))
    try:
        trading = _extract_trading_metrics(backtest_files[-1]) if backtest_files else {}
    except (KeyError, TypeError, ValueError) as exc:
        trading = {"error": str(exc)}
    start, end = _parse_timerange(config.timerange)
    expected_predictions = int((end - start).total_seconds() / 3600) - 1
    integrity = {
        "returncode_zero": returncode == 0,
        "not_timed_out": not timed_out,
        "minimum_training_windows": len(training_records) >= config.minimum_training_windows,
        "effective_n_matches_phase3": training.get("effective_n_values") == [config.effective_n],
        "input_dimension_is_25": training.get("input_feature_counts") == [25],
        "all_records_are_exact_kernel": bool(training_records)
        and all(row.get("kernel_limit") is True for row in training_records),
        "all_oos_predictions_present": oos.get("valid_prediction_rows") == expected_predictions,
        "all_predictions_accepted": oos.get("do_predict_fraction") == 1.0,
        "prediction_dates_are_unique": oos.get("duplicate_prediction_dates") == 0,
        "trading_metrics_present": bool(trading) and "error" not in trading,
    }
    return {
        "model": "exact_centered_rbf_kernel",
        "kernel_limit": True,
        "identifier": identifier,
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_seconds": wall_seconds,
        "recovered_from_artifacts": recovered,
        "success": all(integrity.values()),
        "integrity": integrity,
        "training": training,
        "oos": oos,
        "trading": trading,
        "artifacts": {
            "config": str(config_path),
            "log": str(log_path),
            "training_diagnostics": str(metrics_path),
            "prediction_directory": str(prediction_directory),
            "backtest": str(backtest_files[-1]) if backtest_files else None,
        },
        "stderr_tail": stderr[-2000:],
    }


def _recover_kernel_case(config: Phase6Config, market_data: pd.DataFrame) -> dict[str, Any] | None:
    config_path = config.output_directory / "kernel" / "config.json"
    log_path = config.output_directory / "kernel" / "backtest.log"
    metrics_path = config.output_directory / "kernel" / "training_diagnostics.jsonl"
    backtest_directory = config.output_directory / "kernel" / "backtest"
    if not config_path.is_file() or not log_path.is_file() or not metrics_path.is_file():
        return None
    try:
        generated = json.loads(config_path.read_text(encoding="utf-8"))
        freqai = generated["freqai"]
        parameters = freqai["model_training_parameters"]
        matches = all(
            (
                parameters["gamma"] == config.gamma,
                parameters["ridge"] == config.ridge,
                parameters["rcond"] == config.rcond,
                parameters["dtype"] == config.dtype,
                freqai["train_period_days"] == config.train_period_days,
                freqai["backtest_period_days"] == config.backtest_period_days,
                generated["fee"] == config.fee,
            )
        )
        if not matches:
            return None
        result = _finalize_kernel_case(
            config,
            freqai["identifier"],
            config_path,
            log_path,
            metrics_path,
            backtest_directory,
            market_data,
            returncode=0,
            timed_out=False,
            wall_seconds=max(log_path.stat().st_mtime - config_path.stat().st_mtime, 0.0),
            stderr="",
            recovered=True,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not result["success"]:
        return None
    print("PHASE6 RECOVERED exact RBF kernel", flush=True)
    return result


def _run_kernel_case(
    config: Phase6Config,
    run_id: str,
    market_data: pd.DataFrame,
) -> dict[str, Any]:
    kernel_directory = config.output_directory / "kernel"
    backtest_directory = kernel_directory / "backtest"
    backtest_directory.mkdir(parents=True, exist_ok=True)
    metrics_path = kernel_directory / "training_diagnostics.jsonl"
    metrics_path.unlink(missing_ok=True)
    identifier = f"double-descent-phase6-kernel-{run_id}"
    generated = build_kernel_freqtrade_config(config, identifier, metrics_path, run_id)
    config_path = kernel_directory / "config.json"
    config_path.write_text(json.dumps(generated, indent=2), encoding="utf-8")
    log_path = kernel_directory / "backtest.log"
    command = [
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
        "Phase6CudaRBFRegressor",
        "--freqaimodel-path",
        str(config.model_directory),
        "--timerange",
        config.timerange,
        "--timeframe",
        config.timeframe,
        "--pairs",
        config.pair,
        "--datadir",
        str(config.data_directory),
        "--cache",
        "none",
        "--export",
        "trades",
        "--export-directory",
        str(backtest_directory),
        "--no-color",
    ]
    print("PHASE6 START exact RBF kernel", flush=True)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    started = datetime.now(tz=UTC)
    try:
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=config.subprocess_timeout_seconds,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        returncode = None
        stdout = (
            exc.stdout.decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        timed_out = True
    wall_seconds = (datetime.now(tz=UTC) - started).total_seconds()
    log_path.write_text(
        f"COMMAND: {subprocess.list2cmdline(command)}\n\n{stdout}\n\nSTDERR:\n{stderr}",
        encoding="utf-8",
    )
    result = _finalize_kernel_case(
        config,
        identifier,
        config_path,
        log_path,
        metrics_path,
        backtest_directory,
        market_data,
        returncode,
        timed_out,
        wall_seconds,
        stderr,
    )
    print(
        f"PHASE6 DONE exact RBF kernel success={result['success']} seconds={wall_seconds:.1f}",
        flush=True,
    )
    return result


def _load_detailed_cases(path: Path, feature_counts: tuple[int, ...]) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"detailed RFF results do not exist: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    selected = [row for row in rows if int(row["feature_count"]) in feature_counts]
    if len(selected) != len(feature_counts) or not all(row["success"] for row in selected):
        raise ValueError(f"incomplete RFF detailed results in {path}")
    return selected


def _reference_detailed_path(
    config: Phase6Config, phase5_summary: dict[str, Any], seed: int
) -> Path:
    if seed == PHASE5_SEEDS[0]:
        reference = Path(phase5_summary["config"]["phase4_reference_summary"])
        return reference.parent / "results_detailed.json"
    return config.phase5_summary.parent / "seeds" / f"seed-{seed}" / "results_detailed.json"


def _valid_prediction_frame(case: dict[str, Any]) -> pd.DataFrame:
    frame = _read_predictions(Path(case["artifacts"]["prediction_directory"]))
    valid = (frame["do_predict"] == 1) & np.isfinite(frame["&-forward_return"])
    result = frame.loc[valid, ["date", "&-forward_return"]].copy()
    if result.empty or result["date"].duplicated().any():
        raise ValueError("prediction artifacts are empty or contain duplicate dates")
    return result.rename(columns={"&-forward_return": "prediction"})


def compare_predictions_to_kernel(
    rff_case: dict[str, Any], kernel_predictions: pd.DataFrame
) -> dict[str, Any]:
    rff_predictions = _valid_prediction_frame(rff_case).rename(
        columns={"prediction": "rff_prediction"}
    )
    aligned = kernel_predictions.merge(
        rff_predictions, on="date", how="inner", validate="one_to_one"
    )
    kernel = aligned["kernel_prediction"].to_numpy(dtype=np.float64)
    rff = aligned["rff_prediction"].to_numpy(dtype=np.float64)
    difference = rff - kernel
    denominator = float(np.linalg.norm(kernel))
    correlation = float(np.corrcoef(rff, kernel)[0, 1])
    return {
        "seed": int(rff_case["seed"]),
        "feature_count": int(rff_case["feature_count"]),
        "pn_ratio": float(rff_case["actual_pn_ratio"]),
        "aligned_prediction_count": len(aligned),
        "prediction_rmse_to_kernel": float(np.sqrt(np.mean(difference**2))),
        "prediction_mae_to_kernel": float(np.mean(np.abs(difference))),
        "prediction_relative_l2_error": (
            float(np.linalg.norm(difference) / denominator) if denominator else math.inf
        ),
        "prediction_correlation_with_kernel": correlation,
        "prediction_sign_agreement_with_kernel": float(np.mean(np.sign(rff) == np.sign(kernel))),
        "oos_mse": float(rff_case["oos"]["model"]["mse"]),
        "oos_r2": float(rff_case["oos"]["model"]["r2"]),
        "oos_information_coefficient": float(rff_case["oos"]["model"]["information_coefficient"]),
        "oos_directional_accuracy": float(rff_case["oos"]["model"]["directional_accuracy"]),
        "train_train_mse_mean": float(rff_case["training"]["train_mse_mean"]),
        "train_effective_rank_mean": float(rff_case["training"]["effective_rank_mean"]),
        "train_condition_number_maximum": float(rff_case["training"]["condition_number_maximum"]),
        "train_training_seconds_total": float(rff_case["training"]["training_seconds_total"]),
        "train_peak_vram_mib": float(rff_case["training"]["peak_vram_mib"]),
        "wall_seconds": float(rff_case["wall_seconds"]),
        "trading_sharpe": float(rff_case["trading"]["sharpe"]),
        "trading_profit_factor": float(rff_case["trading"]["profit_factor"]),
        "trading_profit_total": float(rff_case["trading"]["profit_total"]),
    }


def aggregate_convergence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates = []
    for feature_count in sorted({int(row["feature_count"]) for row in rows}):
        cases = [row for row in rows if int(row["feature_count"]) == feature_count]
        for metric in CONVERGENCE_METRICS:
            aggregates.append(
                {
                    "feature_count": feature_count,
                    "pn_ratio": float(cases[0]["pn_ratio"]),
                    "metric": metric,
                    **summarize_values([float(row[metric]) for row in cases]),
                }
            )
    return aggregates


def _aggregate_lookup(
    aggregates: list[dict[str, Any]], feature_count: int, metric: str
) -> dict[str, Any]:
    return next(
        row
        for row in aggregates
        if row["feature_count"] == feature_count and row["metric"] == metric
    )


def assess_kernel_limit(
    config: Phase6Config,
    kernel_result: dict[str, Any],
    phase5_summary: dict[str, Any],
    aggregates: list[dict[str, Any]],
) -> dict[str, Any]:
    feature_counts = [
        count
        for count in (*config.reference_feature_counts, *config.extension_feature_counts)
        if any(row["feature_count"] == count for row in aggregates)
    ]
    relative_errors = [
        _aggregate_lookup(aggregates, count, "prediction_relative_l2_error")["mean"]
        for count in feature_counts
    ]
    trend_correlation = float(spearmanr(feature_counts, relative_errors).statistic)
    log_log_slope = float(np.polyfit(np.log(feature_counts), np.log(relative_errors), deg=1)[0])
    final_count = feature_counts[-1]
    final_error = _aggregate_lookup(aggregates, final_count, "prediction_relative_l2_error")["mean"]
    final_mse_summary = _aggregate_lookup(aggregates, final_count, "oos_mse")
    final_mse = final_mse_summary["mean"]
    kernel_mse = float(kernel_result["oos"]["model"]["mse"])
    zero_mse = float(kernel_result["oos"]["zero_baseline"]["mse"])
    best_under_mse = float(
        phase5_summary["multi_seed_assessment"]["best_underparameterized_mean_mse"]
    )
    relative_mse_gap = abs(final_mse - kernel_mse) / kernel_mse
    convergence_trend = trend_correlation <= -0.8 and relative_errors[-1] < relative_errors[0]
    practical_convergence = (
        final_error <= PRACTICAL_RELATIVE_PREDICTION_ERROR
        and relative_mse_gap <= PRACTICAL_RELATIVE_OOS_MSE_GAP
    )
    kernel_beats_zero = kernel_mse < zero_mse
    kernel_beats_under = kernel_mse < best_under_mse
    return {
        "status": "development_kernel_limit",
        "feature_counts": feature_counts,
        "prediction_error_spearman_vs_feature_count": trend_correlation,
        "prediction_error_log_log_slope": log_log_slope,
        "monte_carlo_rate_consistent_with_inverse_sqrt_p": -0.65 <= log_log_slope <= -0.35,
        "convergence_trend_detected": convergence_trend,
        "practical_convergence_at_one_million": practical_convergence,
        "practical_relative_prediction_error_threshold": (PRACTICAL_RELATIVE_PREDICTION_ERROR),
        "practical_relative_oos_mse_gap_threshold": PRACTICAL_RELATIVE_OOS_MSE_GAP,
        "one_million_mean_relative_prediction_error": final_error,
        "one_million_mean_oos_mse": final_mse,
        "kernel_mse_inside_one_million_seed_ci95": (
            final_mse_summary["ci95_low"] <= kernel_mse <= final_mse_summary["ci95_high"]
        ),
        "one_million_relative_oos_mse_gap_to_kernel": relative_mse_gap,
        "one_million_mean_training_seconds": _aggregate_lookup(
            aggregates, final_count, "train_training_seconds_total"
        )["mean"],
        "kernel_training_seconds": float(kernel_result["training"]["training_seconds_total"]),
        "one_million_vs_kernel_training_time_ratio": (
            _aggregate_lookup(aggregates, final_count, "train_training_seconds_total")["mean"]
            / float(kernel_result["training"]["training_seconds_total"])
        ),
        "kernel_oos_mse": kernel_mse,
        "kernel_oos_r2": float(kernel_result["oos"]["model"]["r2"]),
        "kernel_information_coefficient": float(
            kernel_result["oos"]["model"]["information_coefficient"]
        ),
        "kernel_directional_accuracy": float(kernel_result["oos"]["model"]["directional_accuracy"]),
        "kernel_vs_zero_mse_ratio": kernel_mse / zero_mse,
        "kernel_vs_best_underparameterized_mse_ratio": kernel_mse / best_under_mse,
        "kernel_beats_zero_mse": kernel_beats_zero,
        "kernel_beats_best_underparameterized_mse": kernel_beats_under,
        "kernel_trading_sharpe": float(kernel_result["trading"]["sharpe"]),
        "kernel_trading_profit_factor": float(kernel_result["trading"]["profit_factor"]),
        "kernel_trading_profit_total": float(kernel_result["trading"]["profit_total"]),
        "kernel_supports_useful_benign_overfitting": bool(
            practical_convergence and kernel_beats_zero and kernel_beats_under
        ),
        "interpretation": (
            "Convergence of finite RFF predictions to the exact kernel is a numerical property; "
            "the kernel must still beat predictive baselines to support useful benign overfitting."
        ),
    }


def evaluate_phase6_gate(
    config: Phase6Config,
    phase5_summary: dict[str, Any],
    kernel_result: dict[str, Any],
    extension_summaries: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_comparisons = len(config.seeds) * (
        len(config.reference_feature_counts) + len(config.extension_feature_counts)
    )
    expected_aggregates = (
        len(config.reference_feature_counts) + len(config.extension_feature_counts)
    ) * len(CONVERGENCE_METRICS)
    start, end = _parse_timerange(config.timerange)
    holdout = datetime.strptime(config.holdout_start, "%Y%m%d").replace(tzinfo=UTC)
    expected_prediction_vector_length = int((end - start).total_seconds() / 3600)
    checks = {
        "phase5_reference_passed_and_compatible": phase5_summary.get("gate", {}).get("passed")
        is True,
        "exact_kernel_case_succeeded": kernel_result.get("success") is True,
        "all_extension_phase4_gates_passed": len(extension_summaries) == len(config.seeds)
        and all(summary.get("gate", {}).get("passed") for summary in extension_summaries),
        "complete_prediction_comparison_grid": len(comparison_rows) == expected_comparisons,
        "all_prediction_vectors_align": bool(comparison_rows)
        and all(
            row["aligned_prediction_count"] == expected_prediction_vector_length
            for row in comparison_rows
        ),
        "all_comparison_metrics_are_finite": bool(comparison_rows)
        and all(
            math.isfinite(float(row[metric]))
            for row in comparison_rows
            for metric in CONVERGENCE_METRICS
        ),
        "complete_five_seed_aggregate_grid": len(aggregates) == expected_aggregates
        and all(row["seed_count"] == len(config.seeds) for row in aggregates),
        "cuda_float64_kernel_diagnostics_recorded": bool(
            kernel_result.get("training", {}).get("cuda_devices")
        )
        and kernel_result.get("training", {}).get("peak_vram_mib", 0) > 0
        and config.dtype == "float64",
        "holdout_was_not_used": start < end <= holdout,
    }
    return {"passed": all(checks.values()), "checks": checks}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(
    path: Path,
    kernel_result: dict[str, Any],
    aggregates: list[dict[str, Any]],
) -> bool:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return False
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "OOS MSE toward kernel limit",
            "Prediction distance to exact kernel",
            "Information coefficient",
            "Training effective rank",
        ),
    )
    panels = (
        ("oos_mse", 1, 1),
        ("prediction_relative_l2_error", 1, 2),
        ("oos_information_coefficient", 2, 1),
        ("train_effective_rank_mean", 2, 2),
    )
    for metric, panel_row, panel_column in panels:
        rows = sorted(
            (row for row in aggregates if row["metric"] == metric),
            key=lambda row: row["feature_count"],
        )
        x = [row["feature_count"] for row in rows]
        figure.add_trace(
            go.Scatter(
                x=x,
                y=[row["ci95_high"] for row in rows],
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=panel_row,
            col=panel_column,
        )
        figure.add_trace(
            go.Scatter(
                x=x,
                y=[row["ci95_low"] for row in rows],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(31,119,180,0.15)",
                line={"width": 0},
                name="95% seed interval",
                showlegend=metric == "oos_mse",
                hoverinfo="skip",
            ),
            row=panel_row,
            col=panel_column,
        )
        figure.add_trace(
            go.Scatter(
                x=x,
                y=[row["mean"] for row in rows],
                mode="lines+markers",
                line={"width": 3},
                name="RFF five-seed mean",
                showlegend=metric == "oos_mse",
            ),
            row=panel_row,
            col=panel_column,
        )
    exact_lines = (
        (float(kernel_result["oos"]["model"]["mse"]), 1, 1, "Exact kernel MSE"),
        (0.0, 1, 2, "Exact kernel distance"),
        (
            float(kernel_result["oos"]["model"]["information_coefficient"]),
            2,
            1,
            "Exact kernel IC",
        ),
        (
            float(kernel_result["training"]["effective_rank_mean"]),
            2,
            2,
            "Exact kernel effective rank",
        ),
    )
    for value, panel_row, panel_column, name in exact_lines:
        figure.add_hline(
            y=value,
            line_dash="dash",
            line_color="#d62728",
            annotation_text=name,
            row=panel_row,
            col=panel_column,
        )
    figure.update_xaxes(type="log", title_text="RFF feature count P")
    figure.update_yaxes(type="log", title_text="MSE", row=1, col=1)
    figure.update_yaxes(title_text="Relative L2 error", row=1, col=2)
    figure.update_layout(
        title="Phase 6 - RFF convergence to the exact centered RBF kernel",
        template="plotly_white",
        height=800,
        width=1200,
    )
    figure.write_html(path, include_plotlyjs="cdn")
    return True


def run_phase6(config: Phase6Config) -> dict[str, Any]:
    config.validate()
    phase5_summary = load_compatible_phase5_summary(config)
    config.output_directory.mkdir(parents=True, exist_ok=True)
    phase3_config = Phase3Config(
        data_directory=config.data_directory,
        python_executable=config.python_executable,
        timerange=config.timerange,
        train_periods_days=(config.train_period_days,),
        backtest_period_days=config.backtest_period_days,
        minimum_windows_per_period=config.minimum_training_windows,
    )
    data_audit = audit_data_coverage(phase3_config)
    if not data_audit["passed"]:
        raise RuntimeError("Phase 6 data coverage audit failed")
    market_data = _load_evaluation_market_data(
        _phase4_config(config, config.seeds[0], config.output_directory)
    )
    run_id = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    kernel_result = _recover_kernel_case(config, market_data) if config.resume else None
    if kernel_result is None:
        kernel_result = _run_kernel_case(config, run_id, market_data)
    if not kernel_result["success"]:
        raise RuntimeError(f"exact kernel case failed: {kernel_result['integrity']}")

    extension_summaries = []
    checkpoint_path = config.output_directory / "checkpoint.json"
    for seed in config.seeds:
        seed_directory = config.output_directory / "rff_extensions" / f"seed-{seed}"
        print(f"PHASE6 START RFF extensions seed={seed}", flush=True)
        extension = run_phase4(_phase4_config(config, seed, seed_directory))
        extension_summaries.append(extension)
        checkpoint_path.write_text(
            json.dumps(
                {
                    "kernel_completed": True,
                    "completed_extension_seeds": [
                        item["config"]["seed"] for item in extension_summaries
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"PHASE6 DONE RFF extensions seed={seed}", flush=True)

    kernel_predictions = _valid_prediction_frame(kernel_result).rename(
        columns={"prediction": "kernel_prediction"}
    )
    comparison_rows = []
    for seed in config.seeds:
        reference_cases = _load_detailed_cases(
            _reference_detailed_path(config, phase5_summary, seed),
            config.reference_feature_counts,
        )
        extension_cases = _load_detailed_cases(
            config.output_directory / "rff_extensions" / f"seed-{seed}" / "results_detailed.json",
            config.extension_feature_counts,
        )
        for case in (*reference_cases, *extension_cases):
            comparison_rows.append(compare_predictions_to_kernel(case, kernel_predictions))
    aggregates = aggregate_convergence(comparison_rows)
    assessment = assess_kernel_limit(config, kernel_result, phase5_summary, aggregates)
    gate = evaluate_phase6_gate(
        config,
        phase5_summary,
        kernel_result,
        extension_summaries,
        comparison_rows,
        aggregates,
    )

    comparison_path = config.output_directory / "prediction_convergence_by_seed.csv"
    aggregates_path = config.output_directory / "prediction_convergence_aggregate.csv"
    kernel_path = config.output_directory / "kernel_result.json"
    summary_path = config.output_directory / "summary.json"
    plot_path = config.output_directory / "kernel_limit_convergence.html"
    _write_csv(comparison_path, comparison_rows)
    _write_csv(aggregates_path, aggregates)
    kernel_path.write_text(
        json.dumps(_json_safe(kernel_result), indent=2, allow_nan=False), encoding="utf-8"
    )
    plot_written = _write_plot(plot_path, kernel_result, aggregates)
    config_payload = asdict(config)
    for key in (
        "data_directory",
        "output_directory",
        "phase5_summary",
        "strategy_directory",
        "model_directory",
        "models_directory",
    ):
        config_payload[key] = str(config_payload[key])
    summary = {
        "phase": 6,
        "objective": "Measure convergence of financial RFF models to the exact RBF kernel limit",
        "scope": "frozen 2025 development experiment; 2026 holdout remains untouched",
        "run_id": run_id,
        "config": config_payload,
        "data_audit": data_audit,
        "kernel_result": {
            "training": kernel_result["training"],
            "oos": kernel_result["oos"],
            "trading": kernel_result["trading"],
            "artifacts": kernel_result["artifacts"],
        },
        "convergence_aggregate": aggregates,
        "kernel_limit_assessment": assessment,
        "gate": gate,
        "artifacts": {
            "summary": str(summary_path),
            "kernel_result": str(kernel_path),
            "prediction_convergence_by_seed": str(comparison_path),
            "prediction_convergence_aggregate": str(aggregates_path),
            "interactive_plot": str(plot_path) if plot_written else None,
            "checkpoint": str(checkpoint_path),
        },
    }
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    return summary
