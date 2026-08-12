"""Phase 12: frozen simple baselines for the financial RFF experiment."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.double_descent.freqai.Phase12BaselineRegressor import BASELINES
from research.double_descent.phase3 import Phase3Config, audit_data_coverage
from research.double_descent.phase4 import (
    Phase4Config,
    _finalize_case,
    _flatten_result,
    _load_evaluation_market_data,
    _parse_timerange,
    _read_training_records,
    build_freqtrade_config,
)
from research.double_descent.phase10 import _json_safe, _write_csv


PHASE12_BASELINES = BASELINES
PHASE12_PREDICTION_BASELINES = tuple(
    baseline for baseline in PHASE12_BASELINES if baseline != "buy_and_hold"
)
PHASE12_RIDGE_ALPHA = 1.0
PHASE12_EXPECTED_CASE_COUNT = len(PHASE12_BASELINES)
RAW_MARKET_FEATURE_COUNT = 25


@dataclass(frozen=True)
class Phase12Config:
    data_directory: Path
    output_directory: Path = Path("user_data/research_results/double_descent/phase12")
    phase10_summary: Path = Path("user_data/research_results/double_descent/phase10/summary.json")
    phase10_map: Path = Path(
        "user_data/research_results/double_descent/phase10/representation_map.csv"
    )
    phase11_summary: Path = Path("user_data/research_results/double_descent/phase11/summary.json")
    python_executable: str = sys.executable
    pair: str = "BTC/USDT:USDT"
    timeframe: str = "1h"
    timerange: str = "20250101-20260101"
    holdout_start: str = "20260101"
    train_period_days: int = 90
    backtest_period_days: int = 30
    effective_n: int = 2_159
    baselines: tuple[str, ...] = PHASE12_BASELINES
    ridge_alpha: float = PHASE12_RIDGE_ALPHA
    fee: float = 0.001
    minimum_training_windows: int = 10
    subprocess_timeout_seconds: int = 1_800
    resume: bool = True
    smoke_test: bool = False
    strategy_directory: Path = Path("research/double_descent/freqai")
    model_directory: Path = Path("research/double_descent/freqai")
    models_directory: Path = Path("user_data/models")

    def validate(self) -> None:
        if not self.data_directory.is_dir():
            raise FileNotFoundError(f"data directory does not exist: {self.data_directory}")
        if not Path(self.python_executable).is_file():
            raise FileNotFoundError(f"Freqtrade Python does not exist: {self.python_executable}")
        for path in (self.strategy_directory, self.model_directory):
            if not path.is_dir():
                raise FileNotFoundError(f"research runtime directory does not exist: {path}")
        self._validate_references()
        self._validate_design()

    def _validate_references(self) -> None:
        for path, phase in ((self.phase10_summary, 10), (self.phase11_summary, 11)):
            if not path.is_file():
                raise FileNotFoundError(f"Phase 12 requires completed Phase {phase}: {path}")
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary.get("phase") != phase or summary.get("gate", {}).get("passed") is not True:
                raise ValueError(f"Phase {phase} did not pass its integrity gate")
            if summary.get("design", {}).get("holdout_used") is not False:
                raise ValueError(f"Phase {phase} does not certify a sealed holdout")
        if not self.phase10_map.is_file():
            raise FileNotFoundError("Phase 12 requires the Phase 10 representation map")

    def _validate_design(self) -> None:
        if not self.baselines or len(set(self.baselines)) != len(self.baselines):
            raise ValueError("baselines must be non-empty and unique")
        if not set(self.baselines).issubset(PHASE12_BASELINES):
            raise ValueError("unknown Phase 12 baseline")
        if not self.smoke_test and self.baselines != PHASE12_BASELINES:
            raise ValueError("the full Phase 12 run requires every frozen baseline")
        if not self.smoke_test and self.ridge_alpha != PHASE12_RIDGE_ALPHA:
            raise ValueError("the full Phase 12 run freezes Ridge alpha=1")
        if self.ridge_alpha <= 0:
            raise ValueError("ridge alpha must be positive")
        if self.timeframe != "1h" or self.train_period_days != 90 or self.effective_n != 2_159:
            raise ValueError("Phase 12 is frozen to 1h, 90 days, and measured N=2,159")
        if self.backtest_period_days != 30 or self.minimum_training_windows < 1:
            raise ValueError("invalid rolling evaluation design")
        if not 0 <= self.fee < 0.1:
            raise ValueError("fee must be in [0, 0.1)")
        start, end = _parse_timerange(self.timerange)
        holdout = datetime.strptime(self.holdout_start, "%Y%m%d").replace(tzinfo=UTC)
        if start >= end or end > holdout:
            raise ValueError("Phase 12 may not enter the sealed holdout")


def _phase4_config(config: Phase12Config) -> Phase4Config:
    return Phase4Config(
        data_directory=config.data_directory,
        output_directory=config.output_directory,
        python_executable=config.python_executable,
        cuda_python_executable=config.python_executable,
        pair=config.pair,
        timeframe=config.timeframe,
        timerange=config.timerange,
        holdout_start=config.holdout_start,
        train_period_days=config.train_period_days,
        backtest_period_days=config.backtest_period_days,
        effective_n=config.effective_n,
        ratios=(RAW_MARKET_FEATURE_COUNT / config.effective_n,),
        seed=20260810,
        gamma=0.5,
        ridge=0.0,
        rcond=1e-12,
        dtype="float64",
        chunk_size=4_096,
        fee=config.fee,
        minimum_training_windows=config.minimum_training_windows,
        subprocess_timeout_seconds=config.subprocess_timeout_seconds,
        resume=config.resume,
        strategy_directory=config.strategy_directory,
        model_directory=config.model_directory,
        models_directory=config.models_directory,
    )


def _phase3_config(config: Phase12Config) -> Phase3Config:
    return Phase3Config(
        data_directory=config.data_directory,
        python_executable=config.python_executable,
        pair=config.pair,
        timeframe=config.timeframe,
        timerange=config.timerange,
        train_periods_days=(config.train_period_days,),
        backtest_period_days=config.backtest_period_days,
        minimum_windows_per_period=config.minimum_training_windows,
    )


def _case_paths(config: Phase12Config, baseline: str) -> tuple[Path, Path, Path, Path]:
    root = config.output_directory / "runs" / baseline
    return (
        root / "config.json",
        root / "freqtrade.log",
        root / "training_diagnostics.jsonl",
        root / "backtest",
    )


def _enrich_case(
    result: dict[str, Any],
    baseline: str,
    metrics_path: Path,
    ridge_alpha: float,
) -> dict[str, Any]:
    records = _read_training_records(metrics_path)
    diagnostics_match = bool(records) and all(
        row.get("phase") == 12
        and row.get("baseline") == baseline
        and row.get("dtype") == "float64"
        and row.get("cuda_device") == "cpu_baseline"
        and int(row.get("input_feature_count", -1)) == RAW_MARKET_FEATURE_COUNT
        and math.isclose(
            float(row.get("ridge_alpha", math.nan)),
            ridge_alpha if baseline == "market_ridge" else 0.0,
            rel_tol=0,
            abs_tol=1e-15,
        )
        for row in records
    )
    result["baseline"] = baseline
    result["classification"] = "economic_only" if baseline == "buy_and_hold" else "prediction"
    result["integrity"]["baseline_diagnostics_match"] = diagnostics_match
    result["success"] = bool(result["success"] and diagnostics_match)
    return result


def _recover_case(
    config: Phase12Config,
    baseline: str,
    phase4: Phase4Config,
    market_data: pd.DataFrame,
) -> dict[str, Any] | None:
    config_path, log_path, metrics_path, backtest_directory = _case_paths(config, baseline)
    if not config_path.is_file() or not log_path.is_file() or not metrics_path.is_file():
        return None
    try:
        generated = json.loads(config_path.read_text(encoding="utf-8"))
        freqai = generated["freqai"]
        parameters = freqai["model_training_parameters"]
        if not all(
            (
                parameters["baseline"] == baseline,
                parameters["ridge_alpha"] == config.ridge_alpha,
                freqai["train_period_days"] == config.train_period_days,
                freqai["backtest_period_days"] == config.backtest_period_days,
                generated["fee"] == config.fee,
            )
        ):
            return None
        result = _finalize_case(
            phase4,
            RAW_MARKET_FEATURE_COUNT / config.effective_n,
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
        result = _enrich_case(result, baseline, metrics_path, config.ridge_alpha)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not result["success"]:
        return None
    print(f"PHASE12 RECOVERED baseline={baseline}", flush=True)
    return result


def _run_case(
    config: Phase12Config,
    baseline: str,
    phase4: Phase4Config,
    run_id: str,
    market_data: pd.DataFrame,
) -> dict[str, Any]:
    config_path, log_path, metrics_path, backtest_directory = _case_paths(config, baseline)
    if backtest_directory.exists():
        shutil.rmtree(backtest_directory)
    for directory in (
        config_path.parent,
        log_path.parent,
        metrics_path.parent,
        backtest_directory,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    metrics_path.unlink(missing_ok=True)
    identifier = f"double-descent-phase12-{run_id}-{baseline}"
    generated = build_freqtrade_config(
        phase4,
        RAW_MARKET_FEATURE_COUNT,
        identifier,
        metrics_path,
        run_id,
    )
    parameters = generated["freqai"]["model_training_parameters"]
    parameters.update(
        {
            "phase12_metrics_path": str(metrics_path.resolve()),
            "phase12_run_id": run_id,
            "baseline": baseline,
            "ridge_alpha": config.ridge_alpha,
        }
    )
    config_path.write_text(json.dumps(generated, indent=2), encoding="utf-8")
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
        "Phase12BaselineRegressor",
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
    print(f"PHASE12 START baseline={baseline}", flush=True)
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
    result = _finalize_case(
        phase4,
        RAW_MARKET_FEATURE_COUNT / config.effective_n,
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
    result = _enrich_case(result, baseline, metrics_path, config.ridge_alpha)
    print(
        f"PHASE12 DONE baseline={baseline} success={result['success']} seconds={wall_seconds:.1f}",
        flush=True,
    )
    return result


def flatten_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        row = _flatten_result(result)
        row["baseline"] = result["baseline"]
        row["classification"] = result["classification"]
        rows.append(row)
    return rows


def compare_baselines(
    config: Phase12Config,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    prediction_rows = [row for row in rows if row["baseline"] in PHASE12_PREDICTION_BASELINES]
    ranked = sorted(prediction_rows, key=lambda row: float(row["oos_mse"]))
    zero = next(row for row in rows if row["baseline"] == "zero_return")
    learnable = [
        row for row in prediction_rows if row["baseline"] not in {"zero_return", "historical_mean"}
    ]
    phase10 = pd.read_csv(config.phase10_map)
    market_rff = phase10.loc[phase10["representation"] == "market_rff"].copy()
    best_rff = market_rff.loc[market_rff["oos_mse"].idxmin()]
    overparameterized = market_rff.loc[market_rff["actual_pn_ratio"] >= 5.0]
    best_overparameterized = overparameterized.loc[overparameterized["oos_mse"].idxmin()]
    raw_anchor = phase10.loc[phase10["representation"] == "market_linear"].iloc[0]
    market_ols = next(row for row in rows if row["baseline"] == "market_ols")
    economic_ranked = sorted(rows, key=lambda row: float(row["trading_profit_total"]), reverse=True)
    best_simple = ranked[0]
    best_learnable = min(learnable, key=lambda row: float(row["oos_mse"]))
    return {
        "primary_metric": "chronological OOS MSE; trading is excluded from selection",
        "prediction_ranking": [
            {
                "rank": index,
                "baseline": row["baseline"],
                "mse": row["oos_mse"],
                "mse_over_zero": float(row["oos_mse"]) / float(zero["oos_mse"]),
                "r2": row["oos_r2"],
                "information_coefficient": row["oos_information_coefficient"],
            }
            for index, row in enumerate(ranked, start=1)
        ],
        "simple_baselines_beating_zero_count": sum(
            float(row["oos_mse"]) < float(zero["oos_mse"]) for row in prediction_rows
        ),
        "best_simple_prediction": {
            "baseline": best_simple["baseline"],
            "mse": best_simple["oos_mse"],
            "r2": best_simple["oos_r2"],
        },
        "best_learnable_prediction": {
            "baseline": best_learnable["baseline"],
            "mse": best_learnable["oos_mse"],
            "mse_over_zero": float(best_learnable["oos_mse"]) / float(zero["oos_mse"]),
        },
        "phase10_raw_linear_reproduction": {
            "phase12_mse": market_ols["oos_mse"],
            "phase10_mse": float(raw_anchor["oos_mse"]),
            "mse_ratio": float(market_ols["oos_mse"]) / float(raw_anchor["oos_mse"]),
            "phase12_prediction_standard_deviation": market_ols[
                "oos_prediction_standard_deviation"
            ],
            "phase10_prediction_standard_deviation": float(
                raw_anchor["oos_prediction_standard_deviation"]
            ),
        },
        "best_phase10_market_rff": {
            "seed": int(best_rff["seed"]),
            "actual_pn_ratio": float(best_rff["actual_pn_ratio"]),
            "feature_count": int(best_rff["feature_count"]),
            "mse": float(best_rff["oos_mse"]),
            "mse_over_best_simple": float(best_rff["oos_mse"]) / float(best_simple["oos_mse"]),
        },
        "best_overparameterized_phase10_market_rff": {
            "seed": int(best_overparameterized["seed"]),
            "actual_pn_ratio": float(best_overparameterized["actual_pn_ratio"]),
            "feature_count": int(best_overparameterized["feature_count"]),
            "mse": float(best_overparameterized["oos_mse"]),
            "mse_over_best_simple": float(best_overparameterized["oos_mse"])
            / float(best_simple["oos_mse"]),
        },
        "rff_complexity_justified_by_oos_prediction": float(best_rff["oos_mse"])
        < float(best_simple["oos_mse"]),
        "economic_ranking_is_descriptive_only": [
            {
                "rank": index,
                "baseline": row["baseline"],
                "net_return": row["trading_profit_total"],
                "sharpe": row["trading_sharpe"],
                "profit_factor": row["trading_profit_factor"],
                "trades": row["trading_total_trades"],
            }
            for index, row in enumerate(economic_ranked, start=1)
        ],
    }


def evaluate_phase12_gate(
    config: Phase12Config,
    data_audit: dict[str, Any],
    results: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_name = {row["baseline"]: row for row in rows}
    zero_mses = [float(row["zero_mse"]) for row in rows]
    prediction_rows = [row for row in rows if row["baseline"] in PHASE12_PREDICTION_BASELINES]
    phase10 = pd.read_csv(config.phase10_map)
    raw_anchor = phase10.loc[phase10["representation"] == "market_linear"].iloc[0]
    market_ols = by_name.get("market_ols", {})
    checks = {
        "data_coverage_passed": data_audit.get("passed") is True,
        "every_rolling_case_passed": bool(results) and all(row["success"] for row in results),
        "complete_predeclared_case_count": len(rows) == len(config.baselines),
        "complete_predeclared_baseline_set": set(by_name) == set(config.baselines),
        "all_baseline_diagnostics_match": bool(results)
        and all(row["integrity"].get("baseline_diagnostics_match") for row in results),
        "same_oos_observations_for_every_case": bool(rows)
        and len({int(row["oos_observation_count"]) for row in rows}) == 1,
        "same_zero_baseline_for_every_case": bool(zero_mses)
        and np.allclose(zero_mses, zero_mses[0], rtol=0, atol=1e-20),
        "prediction_metrics_are_finite": bool(prediction_rows)
        and all(
            math.isfinite(float(row[key]))
            for row in prediction_rows
            for key in ("oos_mse", "oos_mae", "oos_r2", "oos_directional_accuracy")
        ),
        "zero_forecast_is_exact": "zero_return" not in config.baselines
        or (
            float(by_name["zero_return"]["oos_prediction_standard_deviation"]) == 0.0
            and float(by_name["zero_return"]["oos_prediction_mean"]) == 0.0
            and int(by_name["zero_return"]["trading_total_trades"]) == 0
        ),
        "buy_and_hold_is_one_costed_long_trade": "buy_and_hold" not in config.baselines
        or (
            int(by_name["buy_and_hold"]["trading_total_trades"]) == 1
            and int(by_name["buy_and_hold"]["trading_long_trades"]) == 1
            and int(by_name["buy_and_hold"]["trading_short_trades"]) == 0
        ),
        "market_ols_reproduces_phase10_anchor": "market_ols" not in config.baselines
        or (
            bool(market_ols)
            and math.isclose(
                float(market_ols["oos_mse"]),
                float(raw_anchor["oos_mse"]),
                rel_tol=1e-6,
                abs_tol=1e-14,
            )
            and math.isclose(
                float(market_ols["oos_prediction_standard_deviation"]),
                float(raw_anchor["oos_prediction_standard_deviation"]),
                rel_tol=1e-6,
                abs_tol=1e-14,
            )
        ),
        "trading_metrics_are_costed": bool(rows)
        and config.fee == 0.001
        and all("trading_total_trades" in row for row in rows),
        "trading_excluded_from_selection": True,
        "holdout_was_not_used": _parse_timerange(config.timerange)[1]
        <= datetime.strptime(config.holdout_start, "%Y%m%d").replace(tzinfo=UTC),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "expected_case_count": len(config.baselines),
        "observed_case_count": len(rows),
    }


def _write_plot(path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return False
    if not rows:
        return False
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("OOS MSE / zero", "OOS R2", "Net return", "Trade count"),
    )
    names = [row["baseline"] for row in rows]
    zero_mse = float(rows[0]["zero_mse"])
    panels = (
        (1, 1, [float(row["oos_mse"]) / zero_mse for row in rows], "MSE / zero"),
        (1, 2, [row["oos_r2"] for row in rows], "R2"),
        (2, 1, [row["trading_profit_total"] for row in rows], "Net return"),
        (2, 2, [row["trading_total_trades"] for row in rows], "Trades"),
    )
    for panel_row, panel_column, values, label in panels:
        figure.add_trace(
            go.Bar(x=names, y=values, name=label, showlegend=False),
            row=panel_row,
            col=panel_column,
        )
    figure.add_hline(y=1.0, line_dash="dash", row=1, col=1)
    figure.add_hline(y=0.0, line_dash="dash", row=1, col=2)
    figure.add_hline(y=0.0, line_dash="dash", row=2, col=1)
    figure.update_xaxes(tickangle=-35)
    figure.update_layout(
        title="Phase 12 - Frozen simple baselines",
        template="plotly_white",
        height=900,
        width=1450,
    )
    figure.write_html(path, include_plotlyjs="cdn")
    return True


def run_phase12(config: Phase12Config) -> dict[str, Any]:
    config.validate()
    config.output_directory.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    phase4 = _phase4_config(config)
    phase4.validate()
    data_audit = audit_data_coverage(_phase3_config(config))
    market_data = _load_evaluation_market_data(phase4) if data_audit["passed"] else pd.DataFrame()
    results = []
    checkpoint_path = config.output_directory / "checkpoint.json"
    if data_audit["passed"]:
        for baseline in config.baselines:
            result = _recover_case(config, baseline, phase4, market_data) if config.resume else None
            if result is None:
                result = _run_case(config, baseline, phase4, run_id, market_data)
            results.append(result)
            checkpoint_path.write_text(
                json.dumps(_json_safe(results), indent=2, allow_nan=False),
                encoding="utf-8",
            )
    rows = flatten_results(results)
    comparisons = (
        compare_baselines(config, rows)
        if rows
        and all(row.get("success") for row in results)
        and set(config.baselines) == set(BASELINES)
        else {}
    )
    gate = evaluate_phase12_gate(config, data_audit, results, rows)
    results_path = config.output_directory / "baseline_results.csv"
    plot_path = config.output_directory / "simple_baselines.html"
    summary_path = config.output_directory / "summary.json"
    _write_csv(results_path, rows)
    plot_written = _write_plot(plot_path, rows)
    config_payload = asdict(config)
    for key in (
        "data_directory",
        "output_directory",
        "phase10_summary",
        "phase10_map",
        "phase11_summary",
        "strategy_directory",
        "model_directory",
        "models_directory",
    ):
        config_payload[key] = str(config_payload[key])
    summary = {
        "phase": 12,
        "objective": "Test whether complex RFF models outperform frozen simple baselines",
        "scope": "development-only simple-baseline comparison; 2026 holdout sealed",
        "run_id": run_id,
        "design": {
            "prediction_baselines": list(PHASE12_PREDICTION_BASELINES),
            "economic_only_baseline": "buy_and_hold",
            "ridge_alpha_frozen_before_run": config.ridge_alpha,
            "parameters_selected_by_trading": False,
            "trading_used_for_inference": False,
            "holdout_used": False,
        },
        "config": config_payload,
        "data_audit": data_audit,
        "results": results,
        "comparisons": comparisons,
        "compute": {
            "case_count": len(results),
            "rolling_model_fit_count": sum(
                int(row.get("train_training_window_count", 0)) for row in rows
            ),
            "summed_freqtrade_wall_seconds": sum(float(row["wall_seconds"]) for row in rows),
            "summed_cpu_training_seconds": sum(
                float(row.get("train_training_seconds_total", 0.0)) for row in rows
            ),
        },
        "gate": gate,
        "artifacts": {
            "summary": str(summary_path),
            "baseline_results": str(results_path),
            "interactive_plot": str(plot_path) if plot_written else None,
            "checkpoint": str(checkpoint_path),
        },
    }
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    return summary
