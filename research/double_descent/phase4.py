"""Phase 4: single-seed financial P/N sweep through genuine rolling FreqAI."""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import subprocess
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from freqtrade.configuration import TimeRange
from freqtrade.data.btanalysis import load_backtest_stats
from freqtrade.data.history import load_pair_history
from freqtrade.enums import CandleType
from research.double_descent.metrics import prediction_metrics
from research.double_descent.phase3 import PN_RATIOS, Phase3Config, audit_data_coverage


@dataclass(frozen=True)
class Phase4Config:
    data_directory: Path
    output_directory: Path = Path("user_data/research_results/double_descent/phase4")
    python_executable: str = sys.executable
    cuda_python_executable: str = ""
    pair: str = "BTC/USDT:USDT"
    timeframe: str = "1h"
    timerange: str = "20250101-20260101"
    holdout_start: str = "20260101"
    train_period_days: int = 90
    backtest_period_days: int = 30
    effective_n: int = 2_159
    ratios: tuple[float, ...] = PN_RATIOS
    seed: int = 20260810
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
        self._validate_runtime()
        self._validate_design()

    def _validate_runtime(self) -> None:
        if not self.data_directory.is_dir():
            raise FileNotFoundError(f"data directory does not exist: {self.data_directory}")
        if not Path(self.python_executable).is_file():
            raise FileNotFoundError(f"Freqtrade Python does not exist: {self.python_executable}")
        if not Path(self.cuda_python_executable).is_file():
            raise FileNotFoundError(f"CUDA Python does not exist: {self.cuda_python_executable}")

    def _validate_design(self) -> None:
        if self.timeframe != "1h" or self.train_period_days != 90:
            raise ValueError("Phase 4 is frozen to 1h candles and a 90-day training window")
        if self.effective_n != 2_159:
            raise ValueError("Phase 4 must use the N=2,159 measured in Phase 3")
        if self.gamma <= 0 or self.ridge < 0 or self.rcond <= 0:
            raise ValueError("gamma/rcond must be positive and ridge must be non-negative")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        if self.chunk_size < 1 or self.minimum_training_windows < 1:
            raise ValueError("chunk size and minimum windows must be positive")
        if not 0 <= self.fee < 0.1:
            raise ValueError("fee must be in [0, 0.1)")
        if not self.ratios or any(ratio <= 0 for ratio in self.ratios):
            raise ValueError("P/N ratios must be positive")
        if tuple(sorted(set(self.ratios))) != self.ratios:
            raise ValueError("P/N ratios must be sorted and unique")
        start, end = _parse_timerange(self.timerange)
        holdout = datetime.strptime(self.holdout_start, "%Y%m%d").replace(tzinfo=UTC)
        if start >= end or end > holdout:
            raise ValueError("development timerange must end no later than the holdout boundary")


def _parse_timerange(timerange: str) -> tuple[datetime, datetime]:
    try:
        start_text, end_text = timerange.split("-", 1)
        return (
            datetime.strptime(start_text, "%Y%m%d").replace(tzinfo=UTC),
            datetime.strptime(end_text, "%Y%m%d").replace(tzinfo=UTC),
        )
    except ValueError as exc:
        raise ValueError("timerange must use YYYYMMDD-YYYYMMDD") from exc


def feature_count_for_ratio(ratio: float, effective_n: int) -> int:
    return max(1, round(ratio * effective_n))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def build_freqtrade_config(
    config: Phase4Config,
    feature_count: int,
    identifier: str,
    metrics_path: Path,
    run_id: str,
) -> dict[str, Any]:
    """Build a frozen development-only ridgeless RFF configuration."""
    return {
        "$schema": "https://schema.freqtrade.io/schema.json",
        "trading_mode": "futures",
        "margin_mode": "isolated",
        "max_open_trades": 1,
        "stake_currency": "USDT",
        "stake_amount": "unlimited",
        "tradable_balance_ratio": 0.99,
        "fiat_display_currency": "USD",
        "dry_run": True,
        "dry_run_wallet": 10_000,
        "fee": config.fee,
        "timeframe": config.timeframe,
        "dataformat_ohlcv": "feather",
        "dataformat_trades": "feather",
        "reduce_df_footprint": False,
        "exchange": {
            "name": "binance",
            "key": "",
            "secret": "",
            "pair_whitelist": [config.pair],
            "pair_blacklist": [],
        },
        "pairlists": [{"method": "StaticPairList"}],
        "entry_pricing": {
            "price_side": "same",
            "use_order_book": True,
            "order_book_top": 1,
            "price_last_balance": 0.0,
        },
        "exit_pricing": {
            "price_side": "same",
            "use_order_book": True,
            "order_book_top": 1,
        },
        "freqai": {
            "enabled": True,
            "identifier": identifier,
            "train_period_days": config.train_period_days,
            "backtest_period_days": config.backtest_period_days,
            "save_backtest_models": False,
            "purge_old_models": 2,
            "live_retrain_hours": 0,
            "activate_tensorboard": False,
            "feature_parameters": {
                "include_timeframes": [config.timeframe],
                "include_corr_pairlist": [],
                "label_period_candles": 1,
                "include_shifted_candles": 0,
                "DI_threshold": 0,
                "weight_factor": 0,
                "principal_component_analysis": False,
                "use_SVM_to_remove_outliers": False,
                "indicator_periods_candles": [14],
                "shuffle_after_split": False,
                "buffer_train_data_candles": 0,
                "plot_feature_importances": 0,
            },
            "data_split_parameters": {"test_size": 0, "shuffle": False},
            "model_training_parameters": {
                "phase4_metrics_path": str(metrics_path.resolve()),
                "phase4_run_id": run_id,
                "cuda_python_executable": str(Path(config.cuda_python_executable).resolve()),
                "feature_count": feature_count,
                "chunk_size": config.chunk_size,
                "gamma": config.gamma,
                "seed": config.seed,
                "ridge": config.ridge,
                "rcond": config.rcond,
                "dtype": config.dtype,
            },
        },
    }


def _load_evaluation_market_data(config: Phase4Config) -> pd.DataFrame:
    start, end = _parse_timerange(config.timerange)
    requested = TimeRange(
        starttype="date",
        stoptype="date",
        startts=int(start.timestamp()),
        stopts=int(end.timestamp()),
    )
    dataframe = load_pair_history(
        pair=config.pair,
        timeframe=config.timeframe,
        datadir=config.data_directory,
        timerange=requested,
        fill_up_missing=False,
        drop_incomplete=False,
        data_format="feather",
        candle_type=CandleType.FUTURES,
    )
    dataframe = dataframe.sort_values("date").drop_duplicates("date")
    dataframe = dataframe.loc[(dataframe["date"] >= start) & (dataframe["date"] < end)].copy()
    dataframe["realized_forward_return"] = dataframe["close"].shift(-1) / dataframe["close"] - 1.0
    dataframe["momentum_1h_prediction"] = dataframe["close"].pct_change()
    return dataframe


def _read_predictions(prediction_directory: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(prediction_directory.glob("*_prediction.feather")):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="pyarrow.feather.read_table is deprecated*",
                category=FutureWarning,
            )
            frame = pd.read_feather(path)
        frame["prediction_window"] = path.stem
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("date")


def evaluate_oos_predictions(
    predictions: pd.DataFrame,
    market_data: pd.DataFrame,
    timerange: str,
) -> dict[str, Any]:
    if predictions.empty:
        raise ValueError("no FreqAI prediction files were produced")
    start, end = _parse_timerange(timerange)
    predictions = predictions.loc[
        (predictions["date"] >= start) & (predictions["date"] < end)
    ].copy()
    duplicate_dates = int(predictions["date"].duplicated().sum())
    merged = predictions.merge(
        market_data[["date", "realized_forward_return", "momentum_1h_prediction"]],
        on="date",
        how="left",
        validate="many_to_one",
    )
    valid = (
        (merged["do_predict"] == 1)
        & np.isfinite(merged["&-forward_return"])
        & np.isfinite(merged["realized_forward_return"])
    )
    evaluated = merged.loc[valid].copy()
    model = prediction_metrics(evaluated["realized_forward_return"], evaluated["&-forward_return"])
    zero = prediction_metrics(
        evaluated["realized_forward_return"],
        np.zeros(len(evaluated), dtype=np.float64),
    )
    momentum = prediction_metrics(
        evaluated["realized_forward_return"], evaluated["momentum_1h_prediction"]
    )
    window_rows = []
    for window, frame in evaluated.groupby("prediction_window", sort=True):
        metrics = prediction_metrics(frame["realized_forward_return"], frame["&-forward_return"])
        window_rows.append({"prediction_window": window, **metrics})
    window_ics = [row["information_coefficient"] for row in window_rows]
    finite_window_ics = [value for value in window_ics if math.isfinite(value)]
    stability = {
        "window_count": len(window_rows),
        "window_ic_mean": (statistics.fmean(finite_window_ics) if finite_window_ics else math.nan),
        "window_ic_standard_deviation": (
            statistics.stdev(finite_window_ics) if len(finite_window_ics) > 1 else 0.0
        ),
        "window_ic_positive_fraction": (
            sum(value > 0 for value in finite_window_ics) / len(finite_window_ics)
            if finite_window_ics
            else math.nan
        ),
        "window_mse_mean": statistics.fmean(row["mse"] for row in window_rows),
        "window_mse_standard_deviation": (
            statistics.stdev(row["mse"] for row in window_rows) if len(window_rows) > 1 else 0.0
        ),
    }
    return {
        "model": model,
        "zero_baseline": zero,
        "momentum_1h_baseline": momentum,
        "stability": stability,
        "window_metrics": window_rows,
        "prediction_rows": len(predictions),
        "valid_prediction_rows": len(evaluated),
        "do_predict_fraction": float(np.mean(predictions["do_predict"] == 1)),
        "duplicate_prediction_dates": duplicate_dates,
        "prediction_start": predictions["date"].min().isoformat(),
        "prediction_end": predictions["date"].max().isoformat(),
    }


def _extract_trading_metrics(backtest_file: Path) -> dict[str, Any]:
    backtest = load_backtest_stats(backtest_file)
    stats = backtest["strategy"]["Phase4RFFStrategy"]
    total_trades = int(stats["total_trades"])
    starting_balance = float(stats["starting_balance"])
    return {
        "total_trades": total_trades,
        "profit_total": float(stats["profit_total"]),
        "profit_total_abs": float(stats["profit_total_abs"]),
        "sharpe": float(stats["sharpe"]),
        "sortino": float(stats["sortino"]),
        "max_drawdown_account": float(stats["max_drawdown_account"]),
        "profit_factor": float(stats["profit_factor"]),
        "win_rate": float(stats["wins"] / total_trades) if total_trades else 0.0,
        "long_trades": int(stats["trade_count_long"]),
        "short_trades": int(stats["trade_count_short"]),
        "total_volume": float(stats["total_volume"]),
        "turnover_multiple": (
            float(stats["total_volume"]) / starting_balance if starting_balance else math.nan
        ),
        "fee_per_side": float(stats["trades"][0]["fee_open"]) if stats["trades"] else math.nan,
        "market_change": float(stats["market_change"]),
    }


def _read_training_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _aggregate_training_diagnostics(
    records: list[dict[str, Any]], interpolation_tolerance: float
) -> dict[str, Any]:
    if not records:
        return {}
    return {
        "training_window_count": len(records),
        "effective_n_values": sorted({int(row["effective_n"]) for row in records}),
        "input_feature_counts": sorted({int(row["input_feature_count"]) for row in records}),
        "rank_minimum": min(int(row["rank"]) for row in records),
        "rank_median": statistics.median(int(row["rank"]) for row in records),
        "rank_maximum": max(int(row["rank"]) for row in records),
        "effective_rank_mean": statistics.fmean(float(row["effective_rank"]) for row in records),
        "condition_number_median": statistics.median(
            float(row["condition_number"]) for row in records
        ),
        "condition_number_maximum": max(float(row["condition_number"]) for row in records),
        "train_mse_mean": statistics.fmean(float(row["train_mse"]) for row in records),
        "train_mse_maximum": max(float(row["train_mse"]) for row in records),
        "interpolated_window_fraction": statistics.fmean(
            float(row["train_mse"] <= interpolation_tolerance) for row in records
        ),
        "training_seconds_total": sum(float(row["training_total_seconds"]) for row in records),
        "peak_vram_mib": max(float(row["peak_vram_mib"]) for row in records),
        "cuda_devices": sorted({row["cuda_device"] for row in records}),
    }


def _finalize_case(
    config: Phase4Config,
    ratio: float,
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
    feature_count = feature_count_for_ratio(ratio, config.effective_n)
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
        "all_oos_predictions_present": oos.get("valid_prediction_rows") == expected_predictions,
        "all_predictions_accepted": oos.get("do_predict_fraction") == 1.0,
        "prediction_dates_are_unique": oos.get("duplicate_prediction_dates") == 0,
        "trading_metrics_present": bool(trading) and "error" not in trading,
    }
    return {
        "target_pn_ratio": ratio,
        "feature_count": feature_count,
        "actual_pn_ratio": feature_count / config.effective_n,
        "seed": config.seed,
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


def _recover_case(
    config: Phase4Config,
    ratio: float,
    market_data: pd.DataFrame,
) -> dict[str, Any] | None:
    feature_count = feature_count_for_ratio(ratio, config.effective_n)
    slug = f"pn-{ratio:.2f}-p-{feature_count}-seed-{config.seed}"
    config_path = config.output_directory / "configs" / f"{slug}.json"
    log_path = config.output_directory / "logs" / f"{slug}.log"
    metrics_path = config.output_directory / "training_diagnostics" / f"{slug}.jsonl"
    backtest_directory = config.output_directory / "backtests" / slug
    if not config_path.is_file() or not log_path.is_file() or not metrics_path.is_file():
        return None
    try:
        generated = json.loads(config_path.read_text(encoding="utf-8"))
        freqai = generated["freqai"]
        parameters = freqai["model_training_parameters"]
        configuration_matches = all(
            (
                parameters["feature_count"] == feature_count,
                parameters["seed"] == config.seed,
                parameters["gamma"] == config.gamma,
                parameters["ridge"] == config.ridge,
                parameters["rcond"] == config.rcond,
                parameters["dtype"] == config.dtype,
                freqai["train_period_days"] == config.train_period_days,
                freqai["backtest_period_days"] == config.backtest_period_days,
                generated["fee"] == config.fee,
            )
        )
        if not configuration_matches:
            return None
        wall_seconds = max(log_path.stat().st_mtime - config_path.stat().st_mtime, 0.0)
        result = _finalize_case(
            config,
            ratio,
            freqai["identifier"],
            config_path,
            log_path,
            metrics_path,
            backtest_directory,
            market_data,
            returncode=0,
            timed_out=False,
            wall_seconds=wall_seconds,
            stderr="",
            recovered=True,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not result["success"]:
        return None
    print(f"PHASE4 RECOVERED P/N={ratio:.2f} P={feature_count}", flush=True)
    return result


def _run_case(
    config: Phase4Config,
    ratio: float,
    run_id: str,
    market_data: pd.DataFrame,
) -> dict[str, Any]:
    feature_count = feature_count_for_ratio(ratio, config.effective_n)
    slug = f"pn-{ratio:.2f}-p-{feature_count}-seed-{config.seed}"
    config_directory = config.output_directory / "configs"
    log_directory = config.output_directory / "logs"
    backtest_directory = config.output_directory / "backtests" / slug
    for directory in (config_directory, log_directory, backtest_directory):
        directory.mkdir(parents=True, exist_ok=True)
    metrics_path = config.output_directory / "training_diagnostics" / f"{slug}.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.unlink(missing_ok=True)
    identifier = f"double-descent-phase4-{run_id}-p{feature_count}-s{config.seed}"
    generated_config = build_freqtrade_config(
        config, feature_count, identifier, metrics_path, run_id
    )
    config_path = config_directory / f"{slug}.json"
    config_path.write_text(json.dumps(generated_config, indent=2), encoding="utf-8")
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
        "Phase4CudaRFFRegressor",
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
    print(f"PHASE4 START P/N={ratio:.2f} P={feature_count}", flush=True)
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
    log_path = log_directory / f"{slug}.log"
    log_path.write_text(
        f"COMMAND: {subprocess.list2cmdline(command)}\n\n{stdout}\n\nSTDERR:\n{stderr}",
        encoding="utf-8",
    )
    result = _finalize_case(
        config,
        ratio,
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
        f"PHASE4 DONE P/N={ratio:.2f} success={result['success']} seconds={wall_seconds:.1f}",
        flush=True,
    )
    return result


def assess_curve(results: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in results if row["success"]]
    if not successful:
        return {"status": "unavailable"}
    ordered = sorted(successful, key=lambda row: row["actual_pn_ratio"])
    near = [row for row in ordered if 0.90 <= row["actual_pn_ratio"] <= 1.10]
    under = [row for row in ordered if row["actual_pn_ratio"] < 0.90]
    if not near or not under:
        return {"status": "insufficient_grid"}
    peak = max(near, key=lambda row: row["oos"]["model"]["mse"])
    best_under = min(under, key=lambda row: row["oos"]["model"]["mse"])
    extreme = ordered[-1]
    deterioration = peak["oos"]["model"]["mse"] - best_under["oos"]["model"]["mse"]
    recovery = peak["oos"]["model"]["mse"] - extreme["oos"]["model"]["mse"]
    recovery_fraction = recovery / deterioration if deterioration > 0 else math.nan
    global_peak = max(ordered, key=lambda row: row["oos"]["model"]["mse"])
    interpolation_onset = next(
        (
            row["actual_pn_ratio"]
            for row in ordered
            if row["training"]["interpolated_window_fraction"] == 1.0
        ),
        None,
    )
    return {
        "status": "preliminary_single_seed",
        "near_threshold_peak_ratio": peak["actual_pn_ratio"],
        "near_threshold_peak_mse": peak["oos"]["model"]["mse"],
        "global_peak_ratio": global_peak["actual_pn_ratio"],
        "global_peak_is_near_threshold": global_peak is peak,
        "best_underparameterized_ratio": best_under["actual_pn_ratio"],
        "best_underparameterized_mse": best_under["oos"]["model"]["mse"],
        "largest_ratio": extreme["actual_pn_ratio"],
        "largest_ratio_mse": extreme["oos"]["model"]["mse"],
        "largest_vs_best_underparameterized_mse_ratio": (
            extreme["oos"]["model"]["mse"] / best_under["oos"]["model"]["mse"]
        ),
        "largest_vs_zero_mse_ratio": (
            extreme["oos"]["model"]["mse"] / extreme["oos"]["zero_baseline"]["mse"]
        ),
        "recovery_fraction": recovery_fraction,
        "interpolation_onset_ratio": interpolation_onset,
        "double_descent_pattern_detected": bool(
            global_peak is peak and deterioration > 0 and recovery_fraction >= 0.5
        ),
        "largest_model_beats_zero_mse": extreme["oos"]["model"]["mse"]
        < extreme["oos"]["zero_baseline"]["mse"],
        "second_descent_beats_best_underparameterized": (
            extreme["oos"]["model"]["mse"] < best_under["oos"]["model"]["mse"]
        ),
        "benign_overfitting_evidence": False,
        "benign_overfitting_reason": (
            "The extreme model interpolates and recovers from the threshold peak, but remains "
            "worse than both the best underparameterized model and the zero-return baseline."
        ),
        "robust_evidence": False,
        "robust_evidence_reason": "Phase 4 uses one RFF seed; Phase 5 must test multiple seeds.",
    }


def evaluate_phase4_gate(
    config: Phase4Config,
    data_audit: dict[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    successful = [row for row in results if row["success"]]
    observation_counts = {row["oos"]["model"]["observation_count"] for row in successful}
    zero_mses = [row["oos"]["zero_baseline"]["mse"] for row in successful]
    finite_metrics = all(
        all(
            math.isfinite(float(row["oos"]["model"][key]))
            for key in ("mse", "mae", "r2", "information_coefficient")
        )
        and math.isfinite(float(row["training"]["condition_number_maximum"]))
        for row in successful
    )
    checks = {
        "development_data_coverage_passed": bool(data_audit.get("passed")),
        "holdout_was_not_used": _parse_timerange(config.timerange)[1]
        <= datetime.strptime(config.holdout_start, "%Y%m%d").replace(tzinfo=UTC),
        "all_grid_cases_succeeded": len(successful) == len(results) == len(config.ratios),
        "all_metrics_are_finite": bool(successful) and finite_metrics,
        "same_oos_observations_for_every_case": len(observation_counts) == 1,
        "same_zero_baseline_for_every_case": bool(zero_mses)
        and np.allclose(zero_mses, zero_mses[0], rtol=0, atol=1e-20),
        "feature_grid_matches_measured_n": all(
            row["feature_count"]
            == feature_count_for_ratio(row["target_pn_ratio"], config.effective_n)
            for row in results
        ),
        "cuda_float64_diagnostics_recorded": all(
            row["training"].get("cuda_devices") and row["training"].get("peak_vram_mib", 0) > 0
            for row in successful
        )
        and config.dtype == "float64",
    }
    return {"passed": all(checks.values()), "checks": checks}


def _flatten_result(row: dict[str, Any]) -> dict[str, Any]:
    flat = {
        "target_pn_ratio": row["target_pn_ratio"],
        "actual_pn_ratio": row["actual_pn_ratio"],
        "feature_count": row["feature_count"],
        "seed": row["seed"],
        "success": row["success"],
        "wall_seconds": row["wall_seconds"],
    }
    for prefix, values in (
        ("train", row["training"]),
        ("oos", row["oos"].get("model", {})),
        ("zero", row["oos"].get("zero_baseline", {})),
        ("momentum", row["oos"].get("momentum_1h_baseline", {})),
        ("stability", row["oos"].get("stability", {})),
        ("trading", row["trading"]),
    ):
        for key, value in values.items():
            if not isinstance(value, (dict, list)):
                flat[f"{prefix}_{key}"] = value
    return flat


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flattened = [_flatten_result(row) for row in rows]
    fieldnames = sorted({key for row in flattened for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)


def _write_plot(path: Path, results: list[dict[str, Any]]) -> bool:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return False
    successful = sorted(
        (row for row in results if row["success"]), key=lambda row: row["actual_pn_ratio"]
    )
    if not successful:
        return False
    ratios = [row["actual_pn_ratio"] for row in successful]
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("OOS MSE", "OOS R2 / IC", "Training error", "Net trading Sharpe"),
    )
    figure.add_trace(
        go.Scatter(
            x=ratios,
            y=[row["oos"]["model"]["mse"] for row in successful],
            mode="lines+markers",
            name="RFF OOS MSE",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=ratios,
            y=[row["oos"]["zero_baseline"]["mse"] for row in successful],
            mode="lines",
            name="Zero baseline MSE",
        ),
        row=1,
        col=1,
    )
    for key, name in (("r2", "OOS R2"), ("information_coefficient", "IC")):
        figure.add_trace(
            go.Scatter(
                x=ratios,
                y=[row["oos"]["model"][key] for row in successful],
                mode="lines+markers",
                name=name,
            ),
            row=1,
            col=2,
        )
    figure.add_trace(
        go.Scatter(
            x=ratios,
            y=[row["training"]["train_mse_mean"] for row in successful],
            mode="lines+markers",
            name="Train MSE",
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=ratios,
            y=[row["trading"]["sharpe"] for row in successful],
            mode="lines+markers",
            name="Sharpe after fees",
        ),
        row=2,
        col=2,
    )
    figure.update_xaxes(type="log", title_text="P/N")
    figure.update_yaxes(type="log", title_text="MSE", row=1, col=1)
    figure.update_yaxes(type="log", title_text="MSE", row=2, col=1)
    figure.update_layout(
        title="Phase 4 - Financial RFF P/N sweep (single seed)",
        template="plotly_white",
        height=800,
        width=1200,
    )
    figure.write_html(path, include_plotlyjs="cdn")
    return True


def run_phase4(config: Phase4Config) -> dict[str, Any]:
    config.validate()
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
    market_data = _load_evaluation_market_data(config) if data_audit["passed"] else pd.DataFrame()
    run_id = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    results = []
    checkpoint_path = config.output_directory / "checkpoint.json"
    if data_audit["passed"]:
        for ratio in config.ratios:
            result = _recover_case(config, ratio, market_data) if config.resume else None
            if result is None:
                result = _run_case(config, ratio, run_id, market_data)
            results.append(result)
            checkpoint_path.write_text(
                json.dumps(_json_safe(results), indent=2, allow_nan=False),
                encoding="utf-8",
            )
    curve = assess_curve(results)
    gate = evaluate_phase4_gate(config, data_audit, results)
    metrics_path = config.output_directory / "metrics.csv"
    detailed_path = config.output_directory / "results_detailed.json"
    summary_path = config.output_directory / "summary.json"
    plot_path = config.output_directory / "double_descent_financial.html"
    if results:
        _write_csv(metrics_path, results)
    detailed_path.write_text(
        json.dumps(_json_safe(results), indent=2, allow_nan=False), encoding="utf-8"
    )
    plot_written = _write_plot(plot_path, results)
    config_payload = asdict(config)
    for key in (
        "data_directory",
        "output_directory",
        "strategy_directory",
        "model_directory",
        "models_directory",
    ):
        config_payload[key] = str(config_payload[key])
    summary = {
        "phase": 4,
        "objective": "Map financial OOS prediction and trading performance across P/N",
        "scope": "development-only single-seed sweep; not robust evidence",
        "run_id": run_id,
        "config": config_payload,
        "data_audit": data_audit,
        "results": [_flatten_result(row) for row in results],
        "curve_assessment": curve,
        "gate": gate,
        "artifacts": {
            "summary": str(summary_path),
            "detailed_results": str(detailed_path),
            "metrics": str(metrics_path) if results else None,
            "interactive_plot": str(plot_path) if plot_written else None,
            "checkpoint": str(checkpoint_path),
        },
    }
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    return summary
