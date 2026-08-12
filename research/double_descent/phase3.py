"""Orchestration and validation for Phase 3 effective-sample-size measurement."""

from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from freqtrade.configuration import TimeRange
from freqtrade.data.history import load_pair_history
from freqtrade.enums import CandleType
from freqtrade.exchange import timeframe_to_seconds


PN_RATIOS = (
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.98,
    1.00,
    1.02,
    1.05,
    1.10,
    1.25,
    1.50,
    2.00,
    3.00,
    5.00,
    10.00,
    25.00,
    50.00,
)


@dataclass(frozen=True)
class Phase3Config:
    data_directory: Path
    output_directory: Path = Path("user_data/research_results/double_descent/phase3")
    python_executable: str = sys.executable
    pair: str = "BTC/USDT:USDT"
    timeframe: str = "1h"
    timerange: str = "20250101-20260101"
    train_periods_days: tuple[int, ...] = (30, 60, 90, 180, 365)
    backtest_period_days: int = 30
    startup_candles: int = 200
    label_period_candles: int = 1
    indicator_periods_candles: tuple[int, ...] = (14,)
    minimum_windows_per_period: int = 10
    maximum_gap_fraction: float = 0.005
    subprocess_timeout_seconds: int = 900
    strategy_name: str = "Phase3EffectiveNStrategy"
    identifier_prefix: str = "double-descent-phase3"
    allow_timeframe_variation: bool = False
    strategy_directory: Path = Path("research/double_descent/freqai")
    model_directory: Path = Path("research/double_descent/freqai")

    def validate(self) -> None:
        if not self.data_directory.is_dir():
            raise FileNotFoundError(f"data directory does not exist: {self.data_directory}")
        if not Path(self.python_executable).is_file():
            raise FileNotFoundError(f"Python executable does not exist: {self.python_executable}")
        if not self.allow_timeframe_variation and self.timeframe != "1h":
            raise ValueError("Phase 3 is frozen to the initial 1h experiment")
        if timeframe_to_seconds(self.timeframe) < 1:
            raise ValueError("timeframe must resolve to a positive duration")
        if self.startup_candles < 1 or self.label_period_candles < 1:
            raise ValueError("startup and label periods must be positive")
        if not self.indicator_periods_candles or any(
            period < 1 for period in self.indicator_periods_candles
        ):
            raise ValueError("indicator periods must be positive")
        if any(days < 1 for days in self.train_periods_days):
            raise ValueError("training periods must be positive")
        if self.backtest_period_days < 1 or self.minimum_windows_per_period < 1:
            raise ValueError("backtest and minimum-window values must be positive")
        _parse_timerange(self.timerange)


def _parse_timerange(timerange: str) -> tuple[datetime, datetime]:
    try:
        start_text, end_text = timerange.split("-", 1)
        start = datetime.strptime(start_text, "%Y%m%d").replace(tzinfo=UTC)
        end = datetime.strptime(end_text, "%Y%m%d").replace(tzinfo=UTC)
    except ValueError as exc:
        raise ValueError("timerange must use YYYYMMDD-YYYYMMDD") from exc
    if end <= start:
        raise ValueError("timerange end must follow its start")
    return start, end


def audit_data_coverage(config: Phase3Config) -> dict[str, Any]:
    """Check real, unfilled futures candles before launching FreqAI."""
    backtest_start, backtest_end = _parse_timerange(config.timerange)
    timeframe_seconds = timeframe_to_seconds(config.timeframe)
    required_start = backtest_start - timedelta(
        days=max(config.train_periods_days),
        seconds=config.startup_candles * timeframe_seconds,
    )
    requested = TimeRange(
        starttype="date",
        stoptype="date",
        startts=int(required_start.timestamp()),
        stopts=int(backtest_end.timestamp()),
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
    if dataframe.empty:
        return {
            "passed": False,
            "reason": "no futures candles were loaded",
            "required_start": required_start.isoformat(),
            "required_end": backtest_end.isoformat(),
        }
    dates = dataframe["date"]
    available_start = dates.min().to_pydatetime()
    available_end = dates.max().to_pydatetime()
    expected_rows = int((available_end - available_start).total_seconds() / timeframe_seconds) + 1
    unique_rows = int(dates.nunique())
    duplicate_rows = len(dates) - unique_rows
    missing_rows = max(expected_rows - unique_rows, 0)
    gap_fraction = missing_rows / expected_rows
    checks = {
        "covers_required_start": available_start <= required_start,
        "covers_backtest_end": available_end >= backtest_end - timedelta(seconds=timeframe_seconds),
        "no_duplicate_timestamps": duplicate_rows == 0,
        "gap_fraction_within_limit": gap_fraction <= config.maximum_gap_fraction,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "required_start": required_start.isoformat(),
        "required_end": backtest_end.isoformat(),
        "available_start": available_start.isoformat(),
        "available_end": available_end.isoformat(),
        "loaded_rows": len(dataframe),
        "expected_timeframe_rows": expected_rows,
        "expected_hourly_rows": expected_rows if config.timeframe == "1h" else None,
        "duplicate_rows": duplicate_rows,
        "missing_rows": missing_rows,
        "gap_fraction": gap_fraction,
    }


def build_freqtrade_config(
    config: Phase3Config,
    train_period_days: int,
    identifier: str,
    metrics_path: Path,
    run_id: str,
) -> dict[str, Any]:
    """Build the frozen, no-tuning configuration used only to observe FreqAI N."""
    return {
        "$schema": "https://schema.freqtrade.io/schema.json",
        "trading_mode": "futures",
        "margin_mode": "isolated",
        "max_open_trades": 1,
        "stake_currency": "USDT",
        "stake_amount": 100,
        "tradable_balance_ratio": 1.0,
        "fiat_display_currency": "USD",
        "dry_run": True,
        "dry_run_wallet": 10_000,
        "timeframe": config.timeframe,
        "dataformat_ohlcv": "feather",
        "dataformat_trades": "feather",
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
            "train_period_days": train_period_days,
            "backtest_period_days": config.backtest_period_days,
            "save_backtest_models": False,
            "purge_old_models": 2,
            "live_retrain_hours": 0,
            "activate_tensorboard": False,
            "feature_parameters": {
                "include_timeframes": [config.timeframe],
                "include_corr_pairlist": [],
                "label_period_candles": config.label_period_candles,
                "include_shifted_candles": 0,
                "DI_threshold": 0,
                "weight_factor": 0,
                "principal_component_analysis": False,
                "use_SVM_to_remove_outliers": False,
                "indicator_periods_candles": list(config.indicator_periods_candles),
                "shuffle_after_split": False,
                "buffer_train_data_candles": 0,
                "plot_feature_importances": 0,
            },
            "data_split_parameters": {
                "test_size": 0,
                "shuffle": False,
            },
            "model_training_parameters": {
                "phase3_metrics_path": str(metrics_path.resolve()),
                "phase3_run_id": run_id,
            },
        },
    }


def _run_period(
    config: Phase3Config,
    train_period_days: int,
    run_id: str,
) -> dict[str, Any]:
    config_directory = config.output_directory / "configs"
    log_directory = config.output_directory / "logs"
    config_directory.mkdir(parents=True, exist_ok=True)
    log_directory.mkdir(parents=True, exist_ok=True)
    metrics_path = config.output_directory / f"measurements_{train_period_days}d.jsonl"
    metrics_path.unlink(missing_ok=True)
    identifier = f"{config.identifier_prefix}-{run_id}-{train_period_days}d"
    generated_config = build_freqtrade_config(
        config, train_period_days, identifier, metrics_path, run_id
    )
    config_path = config_directory / f"phase3_{train_period_days}d.json"
    config_path.write_text(json.dumps(generated_config, indent=2), encoding="utf-8")
    command = [
        config.python_executable,
        "-m",
        "freqtrade",
        "backtesting",
        "--config",
        str(config_path),
        "--strategy",
        config.strategy_name,
        "--strategy-path",
        str(config.strategy_directory),
        "--freqaimodel",
        "Phase3SampleCountRegressor",
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
        "none",
        "--no-color",
    ]
    started = datetime.now(tz=UTC)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
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
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        timed_out = True
    elapsed_seconds = (datetime.now(tz=UTC) - started).total_seconds()
    log_path = log_directory / f"phase3_{train_period_days}d.log"
    log_path.write_text(
        f"COMMAND: {subprocess.list2cmdline(command)}\n\n{stdout}\n\nSTDERR:\n{stderr}",
        encoding="utf-8",
    )
    measurements = []
    if metrics_path.is_file():
        measurements = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return {
        "train_period_days": train_period_days,
        "success": returncode == 0 and bool(measurements),
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_seconds": elapsed_seconds,
        "measurement_count": len(measurements),
        "measurements": measurements,
        "config_path": str(config_path),
        "metrics_path": str(metrics_path),
        "log_path": str(log_path),
        "stderr_tail": stderr[-2000:],
    }


def aggregate_measurements(
    measurements: list[dict[str, Any]], ratios: tuple[float, ...] = PN_RATIOS
) -> list[dict[str, Any]]:
    rows = []
    periods = sorted({int(row["train_period_days"]) for row in measurements})
    for period in periods:
        period_rows = [row for row in measurements if row["train_period_days"] == period]
        effective_values = [int(row["final_train_rows"]) for row in period_rows]
        raw_values = [int(row["raw_rows"]) for row in period_rows]
        median_n = round(statistics.median(effective_values))
        feature_counts = sorted({int(row["model_feature_count"]) for row in period_rows})
        grid = []
        for ratio in ratios:
            feature_count = max(1, round(ratio * median_n))
            grid.append(
                {
                    "target_pn_ratio": ratio,
                    "feature_count": feature_count,
                    "actual_pn_ratio": feature_count / median_n,
                }
            )
        rows.append(
            {
                "train_period_days": period,
                "window_count": len(period_rows),
                "effective_n_min": min(effective_values),
                "effective_n_median": median_n,
                "effective_n_max": max(effective_values),
                "effective_n_mean": statistics.fmean(effective_values),
                "raw_n_median": statistics.median(raw_values),
                "median_retention_fraction": statistics.median(
                    effective / raw
                    for effective, raw in zip(effective_values, raw_values, strict=True)
                ),
                "model_feature_counts": feature_counts,
                "suggested_p_grid": grid,
            }
        )
    return rows


def evaluate_phase3_gate(
    config: Phase3Config,
    data_audit: dict[str, Any],
    period_results: list[dict[str, Any]],
    measurements: list[dict[str, Any]],
) -> dict[str, Any]:
    requested_periods = set(config.train_periods_days)
    observed_periods = {int(row["train_period_days"]) for row in measurements}
    window_counts = {
        period: sum(row["train_period_days"] == period for row in measurements)
        for period in requested_periods
    }
    invariant_rows = all(
        0 < row["final_train_rows"] <= row["filtered_rows_before_split"] <= row["raw_rows"]
        and row["final_test_rows"] == 0
        and row["pipeline_removed_rows"] == 0
        for row in measurements
    )
    finite_rows = all(
        row["nonfinite_train_feature_values"] == 0 and row["nonfinite_train_label_values"] == 0
        for row in measurements
    )
    feature_counts = {row["model_feature_count"] for row in measurements}
    window_keys = [
        (row["train_period_days"], row["pair"], row["raw_start"], row["raw_end"])
        for row in measurements
    ]
    checks = {
        "data_coverage_passed": bool(data_audit.get("passed")),
        "all_subprocesses_succeeded": len(period_results) == len(requested_periods)
        and all(row["success"] for row in period_results),
        "all_training_periods_observed": observed_periods == requested_periods,
        "minimum_windows_observed": all(
            count >= config.minimum_windows_per_period for count in window_counts.values()
        ),
        "sample_count_invariants_hold": bool(measurements) and invariant_rows,
        "processed_data_are_finite": bool(measurements) and finite_rows,
        "feature_dimension_is_stable": len(feature_counts) == 1,
        "training_windows_are_unique": len(window_keys) == len(set(window_keys)),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "window_counts": window_counts,
        "observed_feature_counts": sorted(feature_counts),
    }


def _write_measurements_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(path: Path, aggregate: list[dict[str, Any]]) -> bool:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return False
    periods = [row["train_period_days"] for row in aggregate]
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Effective training N", "Retention after causal preprocessing"),
    )
    figure.add_trace(
        go.Scatter(
            x=periods,
            y=[row["effective_n_median"] for row in aggregate],
            error_y={
                "type": "data",
                "symmetric": False,
                "array": [row["effective_n_max"] - row["effective_n_median"] for row in aggregate],
                "arrayminus": [
                    row["effective_n_median"] - row["effective_n_min"] for row in aggregate
                ],
            },
            mode="lines+markers",
            name="Effective N",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=periods,
            y=[row["median_retention_fraction"] for row in aggregate],
            name="Retention",
        ),
        row=1,
        col=2,
    )
    figure.update_xaxes(title_text="Training window (days)")
    figure.update_yaxes(title_text="Samples", row=1, col=1)
    figure.update_yaxes(title_text="Fraction", range=[0, 1.05], row=1, col=2)
    figure.update_layout(
        title="Phase 3 - Effective FreqAI training sample size",
        template="plotly_white",
        height=500,
        width=1200,
        showlegend=False,
    )
    figure.write_html(path, include_plotlyjs="cdn")
    return True


def run_phase3(config: Phase3Config) -> dict[str, Any]:
    config.validate()
    config.output_directory.mkdir(parents=True, exist_ok=True)
    data_audit = audit_data_coverage(config)
    run_id = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    period_results = []
    if data_audit["passed"]:
        period_results = [
            _run_period(config, train_period_days, run_id)
            for train_period_days in config.train_periods_days
        ]
    measurements = [
        measurement
        for period_result in period_results
        for measurement in period_result["measurements"]
    ]
    aggregate = aggregate_measurements(measurements) if measurements else []
    gate = evaluate_phase3_gate(config, data_audit, period_results, measurements)
    measurements_path = config.output_directory / "measurements.csv"
    aggregate_path = config.output_directory / "aggregate.json"
    summary_path = config.output_directory / "summary.json"
    plot_path = config.output_directory / "effective_n.html"
    if measurements:
        _write_measurements_csv(measurements_path, measurements)
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    plot_written = _write_plot(plot_path, aggregate) if aggregate else False
    summary = {
        "phase": 3,
        "objective": "Measure effective N delivered by real rolling FreqAI training windows",
        "run_id": run_id,
        "config": {
            **asdict(config),
            "data_directory": str(config.data_directory),
            "output_directory": str(config.output_directory),
            "strategy_directory": str(config.strategy_directory),
            "model_directory": str(config.model_directory),
        },
        "data_audit": data_audit,
        "period_results": [
            {key: value for key, value in row.items() if key != "measurements"}
            for row in period_results
        ],
        "aggregate": aggregate,
        "gate": gate,
        "artifacts": {
            "summary": str(summary_path),
            "measurements": str(measurements_path) if measurements else None,
            "aggregate": str(aggregate_path),
            "interactive_plot": str(plot_path) if plot_written else None,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary
