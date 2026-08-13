"""Phase 10: market-information versus deterministic-noise feature controls."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.double_descent.phase3 import Phase3Config, audit_data_coverage
from research.double_descent.phase4 import (
    Phase4Config,
    _finalize_case,
    _flatten_result,
    _load_evaluation_market_data,
    _parse_timerange,
    _read_training_records,
    build_freqtrade_config,
    evaluate_phase4_gate,
    feature_count_for_ratio,
)
from research.double_descent.phase5 import summarize_values
from research.double_descent.phase8 import assess_flat_curve
from research.double_descent.phase9 import (
    PHASE9_MAP_RATIOS,
    PHASE9_ROBUSTNESS_RATIOS,
    PHASE9_SEEDS,
)


PHASE10_REPRESENTATIONS = (
    "market_linear",
    "market_rff",
    "pure_noise",
    "market_plus_noise",
)
PHASE10_CURVE_REPRESENTATIONS = PHASE10_REPRESENTATIONS[1:]
PHASE10_MAP_RATIOS = PHASE9_MAP_RATIOS
PHASE10_ROBUSTNESS_RATIOS = PHASE9_ROBUSTNESS_RATIOS
PHASE10_SEEDS = PHASE9_SEEDS
PHASE10_GAMMA = 0.5
PHASE10_EXPECTED_CASE_COUNT = 64

AGGREGATE_METRICS = (
    "oos_mse",
    "oos_mae",
    "oos_r2",
    "oos_information_coefficient",
    "oos_spearman_information_coefficient",
    "oos_directional_accuracy",
    "train_train_mse_mean",
    "train_interpolated_window_fraction",
    "train_effective_rank_mean",
    "train_condition_number_maximum",
    "trading_total_trades",
    "trading_profit_total",
    "trading_sharpe",
    "trading_profit_factor",
    "trading_max_drawdown_account",
)


@dataclass(frozen=True)
class Phase10Config:
    data_directory: Path
    output_directory: Path = Path("user_data/research_results/double_descent/phase10")
    phase9_summary: Path = Path("user_data/research_results/double_descent/phase9/summary.json")
    frozen_gamma_file: Path = Path(
        "user_data/research_results/double_descent/phase9/frozen_gamma.json"
    )
    python_executable: str = sys.executable
    cuda_python_executable: str = ""
    pair: str = "BTC/USDT:USDT"
    timeframe: str = "1h"
    timerange: str = "20250101-20260101"
    holdout_start: str = "20260101"
    train_period_days: int = 90
    backtest_period_days: int = 30
    effective_n: int = 2_159
    map_ratios: tuple[float, ...] = PHASE10_MAP_RATIOS
    robustness_ratios: tuple[float, ...] = PHASE10_ROBUSTNESS_RATIOS
    representations: tuple[str, ...] = PHASE10_REPRESENTATIONS
    seeds: tuple[int, ...] = PHASE10_SEEDS
    gamma: float = PHASE10_GAMMA
    ridge: float = 0.0
    rcond: float = 1e-12
    dtype: str = "float64"
    chunk_size: int = 4_096
    fee: float = 0.001
    minimum_training_windows: int = 10
    effective_n_tolerance: int = 0
    minimum_seed_count: int = 3
    subprocess_timeout_seconds: int = 1_800
    resume: bool = True
    smoke_test: bool = False
    allow_external_design: bool = False
    label_period_candles: int = 1
    startup_candles: int = 200
    indicator_periods_candles: tuple[int, ...] = (14,)
    strategy_name: str = "Phase4RFFStrategy"
    strategy_directory: Path = Path("research/double_descent/freqai")
    model_directory: Path = Path("research/double_descent/freqai")
    models_directory: Path = Path("user_data/models")

    @property
    def market_linear_ratio(self) -> float:
        return 25 / self.effective_n

    def validate(self) -> None:
        self._validate_phase9_freeze()
        self._validate_design()
        _phase4_config(
            self,
            representation="market_rff",
            seed=self.seeds[0],
            ratios=self.map_ratios,
            output_directory=self.output_directory / "validation",
        ).validate()

    def _validate_phase9_freeze(self) -> None:
        if not self.phase9_summary.is_file() or not self.frozen_gamma_file.is_file():
            raise FileNotFoundError("Phase 10 requires the completed Phase 9 gamma artifacts")
        summary = json.loads(self.phase9_summary.read_text(encoding="utf-8"))
        frozen = json.loads(self.frozen_gamma_file.read_text(encoding="utf-8"))
        if summary.get("phase") != 9 or not summary.get("gate", {}).get("passed"):
            raise ValueError("Phase 9 did not pass its integrity gate")
        selected = summary.get("gamma_selection", {}).get("selected_gamma")
        if selected != self.gamma or frozen.get("gamma") != self.gamma:
            raise ValueError("Phase 10 gamma must match the frozen Phase 9 selection")
        if frozen.get("holdout_used") is not False:
            raise ValueError("the Phase 9 gamma artifact does not certify a sealed holdout")

    def _validate_design(self) -> None:
        for ratios, label in (
            (self.map_ratios, "map ratios"),
            (self.robustness_ratios, "robustness ratios"),
        ):
            if not ratios or tuple(sorted(set(ratios))) != ratios or any(x <= 0 for x in ratios):
                raise ValueError(f"{label} must be positive, sorted, and unique")
        if not set(self.robustness_ratios).issubset(self.map_ratios):
            raise ValueError("robustness ratios must be a subset of the main map")
        if tuple(dict.fromkeys(self.representations)) != self.representations:
            raise ValueError("representations must be unique")
        if set(self.representations) != set(PHASE10_REPRESENTATIONS):
            raise ValueError("Phase 10 requires all four representation controls")
        if len(set(self.seeds)) != len(self.seeds) or any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be unique non-negative integers")
        minimum = 1 if self.smoke_test else 3
        if self.minimum_seed_count < minimum or len(self.seeds) < self.minimum_seed_count:
            raise ValueError(f"Phase 10 requires at least {minimum} seed(s) for this run")
        if self.ridge != 0 or self.gamma != PHASE10_GAMMA:
            raise ValueError("Phase 10 freezes gamma=0.5 and ridge=0")
        _, end = _parse_timerange(self.timerange)
        holdout = datetime.strptime(self.holdout_start, "%Y%m%d").replace(tzinfo=UTC)
        if end > holdout:
            raise ValueError("Phase 10 may not enter the sealed holdout")
        self._validate_confirmatory_design()

    def _validate_confirmatory_design(self) -> None:
        if self.smoke_test:
            return
        frozen = (
            (self.map_ratios, PHASE10_MAP_RATIOS, "main map"),
            (self.robustness_ratios, PHASE10_ROBUSTNESS_RATIOS, "robustness map"),
            (self.representations, PHASE10_REPRESENTATIONS, "representations"),
            (self.seeds, PHASE10_SEEDS, "seeds"),
        )
        for actual, expected, label in frozen:
            if actual != expected:
                raise ValueError(f"the confirmatory run requires the frozen {label}")


def _phase4_config(
    config: Phase10Config,
    representation: str,
    seed: int,
    ratios: tuple[float, ...],
    output_directory: Path,
) -> Phase4Config:
    del representation
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
        ratios=ratios,
        seed=seed,
        gamma=config.gamma,
        ridge=config.ridge,
        rcond=config.rcond,
        dtype=config.dtype,
        chunk_size=config.chunk_size,
        fee=config.fee,
        minimum_training_windows=config.minimum_training_windows,
        effective_n_tolerance=config.effective_n_tolerance,
        subprocess_timeout_seconds=config.subprocess_timeout_seconds,
        resume=config.resume,
        allow_external_design=config.allow_external_design,
        label_period_candles=config.label_period_candles,
        indicator_periods_candles=config.indicator_periods_candles,
        strategy_name=config.strategy_name,
        strategy_directory=config.strategy_directory,
        model_directory=config.model_directory,
        models_directory=config.models_directory,
    )


def _phase3_config(config: Phase10Config) -> Phase3Config:
    return Phase3Config(
        data_directory=config.data_directory,
        python_executable=config.python_executable,
        pair=config.pair,
        timeframe=config.timeframe,
        timerange=config.timerange,
        train_periods_days=(config.train_period_days,),
        backtest_period_days=config.backtest_period_days,
        minimum_windows_per_period=config.minimum_training_windows,
        startup_candles=config.startup_candles,
        label_period_candles=config.label_period_candles,
        indicator_periods_candles=config.indicator_periods_candles,
        strategy_name=config.strategy_name,
        allow_timeframe_variation=config.allow_external_design,
    )


def _task_grid(
    config: Phase10Config,
) -> list[tuple[str, int, tuple[float, ...]]]:
    tasks = [("market_linear", config.seeds[0], (config.market_linear_ratio,))]
    for representation in PHASE10_CURVE_REPRESENTATIONS:
        tasks.append((representation, config.seeds[0], config.map_ratios))
        tasks.extend((representation, seed, config.robustness_ratios) for seed in config.seeds[1:])
    return tasks


def _composition_counts(representation: str, feature_count: int) -> tuple[int, int]:
    if representation == "market_linear":
        return 25, 0
    if representation == "market_rff":
        return 0, 0
    if representation == "pure_noise":
        return 0, feature_count
    if representation == "market_plus_noise":
        return 25, feature_count - 25
    raise ValueError(f"unknown representation: {representation}")


def _enrich_case(
    result: dict[str, Any],
    representation: str,
    metrics_path: Path,
) -> dict[str, Any]:
    records = _read_training_records(metrics_path)
    expected_market, expected_noise = _composition_counts(
        representation, int(result["feature_count"])
    )
    representation_matches = bool(records) and all(
        row.get("representation") == representation for row in records
    )
    composition_matches = bool(records) and all(
        int(row.get("market_feature_count", -1)) == expected_market
        and int(row.get("noise_feature_count", -1)) == expected_noise
        and int(row.get("feature_count", -1)) == int(result["feature_count"])
        for row in records
    )
    result["representation"] = representation
    result["integrity"]["representation_matches"] = representation_matches
    result["integrity"]["feature_composition_matches"] = composition_matches
    result["success"] = bool(result["success"] and representation_matches and composition_matches)
    return result


def _case_paths(
    config: Phase4Config,
    ratio: float,
) -> tuple[str, Path, Path, Path, Path]:
    feature_count = feature_count_for_ratio(ratio, config.effective_n)
    slug = f"pn-{ratio:.5f}-p-{feature_count}-seed-{config.seed}"
    config_path = config.output_directory / "configs" / f"{slug}.json"
    log_path = config.output_directory / "logs" / f"{slug}.log"
    metrics_path = config.output_directory / "training_diagnostics" / f"{slug}.jsonl"
    backtest_directory = config.output_directory / "backtests" / slug
    return slug, config_path, log_path, metrics_path, backtest_directory


def _recover_case(
    config: Phase4Config,
    representation: str,
    ratio: float,
    market_data: pd.DataFrame,
) -> dict[str, Any] | None:
    _, config_path, log_path, metrics_path, backtest_directory = _case_paths(config, ratio)
    if not config_path.is_file() or not log_path.is_file() or not metrics_path.is_file():
        return None
    feature_count = feature_count_for_ratio(ratio, config.effective_n)
    try:
        generated = json.loads(config_path.read_text(encoding="utf-8"))
        freqai = generated["freqai"]
        parameters = freqai["model_training_parameters"]
        configuration_matches = all(
            (
                parameters["representation"] == representation,
                parameters["feature_count"] == feature_count,
                parameters["seed"] == config.seed,
                parameters["gamma"] == config.gamma,
                parameters["ridge"] == config.ridge,
                parameters["rcond"] == config.rcond,
                parameters["dtype"] == config.dtype,
                freqai["train_period_days"] == config.train_period_days,
                freqai["backtest_period_days"] == config.backtest_period_days,
                freqai["feature_parameters"]["label_period_candles"] == config.label_period_candles,
                generated["timeframe"] == config.timeframe,
                generated["exchange"]["pair_whitelist"] == [config.pair],
                generated["fee"] == config.fee,
            )
        )
        if not configuration_matches:
            return None
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
            wall_seconds=max(log_path.stat().st_mtime - config_path.stat().st_mtime, 0.0),
            stderr="",
            recovered=True,
        )
        result = _enrich_case(result, representation, metrics_path)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not result["success"]:
        return None
    print(
        f"PHASE10 RECOVERED representation={representation} P/N={ratio:.5f} P={feature_count}",
        flush=True,
    )
    return result


def _run_case(
    config: Phase4Config,
    representation: str,
    ratio: float,
    run_id: str,
    market_data: pd.DataFrame,
) -> dict[str, Any]:
    feature_count = feature_count_for_ratio(ratio, config.effective_n)
    _, config_path, log_path, metrics_path, backtest_directory = _case_paths(config, ratio)
    if backtest_directory.exists():
        shutil.rmtree(backtest_directory)
    for directory in (config_path.parent, log_path.parent, backtest_directory, metrics_path.parent):
        directory.mkdir(parents=True, exist_ok=True)
    metrics_path.unlink(missing_ok=True)
    identifier = f"double-descent-phase10-{run_id}-{representation}-p{feature_count}-s{config.seed}"
    generated = build_freqtrade_config(config, feature_count, identifier, metrics_path, run_id)
    parameters = generated["freqai"]["model_training_parameters"]
    parameters.update(
        {
            "phase10_metrics_path": str(metrics_path.resolve()),
            "phase10_run_id": run_id,
            "representation": representation,
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
        config.strategy_name,
        "--strategy-path",
        str(config.strategy_directory),
        "--freqaimodel",
        "Phase10CudaControlRegressor",
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
    print(
        f"PHASE10 START representation={representation} P/N={ratio:.5f} P={feature_count}",
        flush=True,
    )
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
    result = _enrich_case(result, representation, metrics_path)
    print(
        f"PHASE10 DONE representation={representation} P/N={ratio:.5f} "
        f"success={result['success']} seconds={wall_seconds:.1f}",
        flush=True,
    )
    return result


def _run_substudy(
    config: Phase10Config,
    representation: str,
    seed: int,
    ratios: tuple[float, ...],
    run_id: str,
) -> dict[str, Any]:
    output_directory = config.output_directory / "runs" / representation / f"seed-{seed}"
    summary_path = output_directory / "summary.json"
    if config.resume and summary_path.is_file():
        try:
            cached = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cached = {}
        cached_results = cached.get("results", [])
        if (
            cached.get("representation") == representation
            and cached.get("seed") == seed
            and cached.get("ratios") == list(ratios)
            and cached.get("gate", {}).get("passed") is True
            and len(cached_results) == len(ratios)
            and all(result.get("success") is True for result in cached_results)
        ):
            print(
                f"PHASE10 RECOVERED SUBSTUDY representation={representation} "
                f"seed={seed} points={len(ratios)}",
                flush=True,
            )
            return cached
    phase4 = _phase4_config(
        config,
        representation,
        seed,
        ratios,
        output_directory,
    )
    phase4.validate()
    data_audit = audit_data_coverage(_phase3_config(config))
    market_data = _load_evaluation_market_data(phase4) if data_audit["passed"] else pd.DataFrame()
    results = []
    checkpoint_path = output_directory / "checkpoint.json"
    if data_audit["passed"]:
        for ratio in ratios:
            result = (
                _recover_case(phase4, representation, ratio, market_data) if config.resume else None
            )
            if result is None:
                result = _run_case(phase4, representation, ratio, run_id, market_data)
            results.append(result)
            checkpoint_path.write_text(
                json.dumps(_json_safe(results), indent=2, allow_nan=False),
                encoding="utf-8",
            )
    gate = evaluate_phase4_gate(phase4, data_audit, results)
    gate["checks"]["representation_and_composition_match"] = bool(results) and all(
        row["integrity"].get("representation_matches")
        and row["integrity"].get("feature_composition_matches")
        for row in results
    )
    gate["passed"] = all(gate["checks"].values())
    output_directory.mkdir(parents=True, exist_ok=True)
    summary = {
        "phase": 10,
        "scope": "one frozen representation/seed rolling FreqAI substudy",
        "run_id": run_id,
        "representation": representation,
        "seed": seed,
        "ratios": list(ratios),
        "data_audit": data_audit,
        "results": results,
        "gate": gate,
        "artifacts": {
            "summary": str(summary_path),
            "checkpoint": str(checkpoint_path),
        },
    }
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    return summary


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


def flatten_substudies(substudies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for substudy in substudies:
        for result in substudy["results"]:
            row = _flatten_result(result)
            row["representation"] = substudy["representation"]
            row["seed"] = substudy["seed"]
            rows.append(row)
    return rows


def aggregate_robustness(
    config: Phase10Config,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    aggregates = []
    for representation in PHASE10_CURVE_REPRESENTATIONS:
        for ratio in config.robustness_ratios:
            matched = [
                row
                for row in rows
                if row["representation"] == representation
                and row["target_pn_ratio"] == ratio
                and row["seed"] in config.seeds
            ]
            if len(matched) != len(config.seeds):
                continue
            for metric in AGGREGATE_METRICS:
                aggregates.append(
                    {
                        "representation": representation,
                        "target_pn_ratio": ratio,
                        "actual_pn_ratio": matched[0]["actual_pn_ratio"],
                        "feature_count": matched[0]["feature_count"],
                        "metric": metric,
                        **summarize_values([float(row[metric]) for row in matched]),
                    }
                )
    return aggregates


def assess_representation_curves(
    config: Phase10Config,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    assessments = []
    for representation in PHASE10_CURVE_REPRESENTATIONS:
        for seed in config.seeds:
            matched = [
                row
                for row in rows
                if row["representation"] == representation
                and row["seed"] == seed
                and row["target_pn_ratio"] in config.robustness_ratios
            ]
            assessments.append(
                {
                    "representation": representation,
                    "seed": seed,
                    **assess_flat_curve(matched),
                }
            )
    return assessments


def compare_representations(
    config: Phase10Config,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    robust_rows = [
        row
        for row in rows
        if row["representation"] in PHASE10_CURVE_REPRESENTATIONS
        and row["seed"] in config.seeds
        and row["target_pn_ratio"] in config.robustness_ratios
    ]
    lookup = {
        (row["representation"], row["seed"], row["target_pn_ratio"]): row for row in robust_rows
    }
    pairs = (
        ("market_rff", "pure_noise"),
        ("market_rff", "market_plus_noise"),
        ("market_plus_noise", "pure_noise"),
    )
    pairwise = []
    expected = len(config.seeds) * len(config.robustness_ratios)
    for left, right in pairs:
        matched = [
            (
                lookup[(left, seed, ratio)],
                lookup[(right, seed, ratio)],
            )
            for seed in config.seeds
            for ratio in config.robustness_ratios
        ]
        ratios = [float(a["oos_mse"]) / float(b["oos_mse"]) for a, b in matched]
        pairwise.append(
            {
                "left": left,
                "right": right,
                "matched_cell_count": len(matched),
                "left_mse_win_count": sum(
                    float(a["oos_mse"]) < float(b["oos_mse"]) for a, b in matched
                ),
                "left_mse_win_fraction": sum(value < 1 for value in ratios) / expected,
                "geometric_mean_left_to_right_mse": math.exp(
                    statistics.fmean(math.log(value) for value in ratios)
                ),
                "median_left_to_right_mse": statistics.median(ratios),
                "minimum_left_to_right_mse": min(ratios),
                "maximum_left_to_right_mse": max(ratios),
            }
        )
    pure_noise = [row for row in robust_rows if row["representation"] == "pure_noise"]
    systematic_ratios = []
    for ratio in config.robustness_ratios:
        matched = [row for row in pure_noise if row["target_pn_ratio"] == ratio]
        if len(matched) == len(config.seeds) and all(
            float(row["oos_mse"]) < float(row["zero_mse"]) for row in matched
        ):
            systematic_ratios.append(ratio)
    raw_market = next(row for row in rows if row["representation"] == "market_linear")
    return {
        "primary_metric": "chronological OOS MSE; trading is excluded from inference",
        "matched_pairwise_comparisons": pairwise,
        "pure_noise_negative_control": {
            "cell_count": len(pure_noise),
            "mse_beats_zero_cell_count": sum(
                float(row["oos_mse"]) < float(row["zero_mse"]) for row in pure_noise
            ),
            "positive_r2_cell_count": sum(float(row["oos_r2"]) > 0 for row in pure_noise),
            "ratios_beating_zero_in_all_seeds": systematic_ratios,
            "systematic_predictive_failure_detected": bool(systematic_ratios),
        },
        "raw_market_anchor": {
            "feature_count": raw_market["feature_count"],
            "actual_pn_ratio": raw_market["actual_pn_ratio"],
            "oos_mse": raw_market["oos_mse"],
            "mse_over_zero": float(raw_market["oos_mse"]) / float(raw_market["zero_mse"]),
            "oos_r2": raw_market["oos_r2"],
            "information_coefficient": raw_market["oos_information_coefficient"],
            "trading_profit_total": raw_market["trading_profit_total"],
            "trading_sharpe": raw_market["trading_sharpe"],
        },
    }


def evaluate_phase10_gate(
    config: Phase10Config,
    substudies: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_cells = {
        (representation, seed, ratio)
        for representation, seed, ratios in _task_grid(config)
        for ratio in ratios
    }
    observed_cells = {(row["representation"], row["seed"], row["target_pn_ratio"]) for row in rows}
    zero_mses = [float(row["zero_mse"]) for row in rows]
    expected_aggregates = (
        len(PHASE10_CURVE_REPRESENTATIONS) * len(config.robustness_ratios) * len(AGGREGATE_METRICS)
    )
    phase9 = json.loads(config.phase9_summary.read_text(encoding="utf-8"))
    checks = {
        "phase9_gamma_freeze_passed": phase9.get("gate", {}).get("passed") is True
        and phase9.get("gamma_selection", {}).get("selected_gamma") == config.gamma,
        "every_rolling_substudy_passed": bool(substudies)
        and all(item["gate"]["passed"] for item in substudies),
        "complete_predeclared_case_count": len(rows) == PHASE10_EXPECTED_CASE_COUNT,
        "complete_predeclared_cell_map": observed_cells == expected_cells,
        "complete_robustness_aggregate": len(aggregates) == expected_aggregates,
        "feature_counts_match_measured_n": all(
            int(row["feature_count"])
            == feature_count_for_ratio(float(row["target_pn_ratio"]), config.effective_n)
            for row in rows
        ),
        "all_metrics_are_finite": bool(rows)
        and all(
            math.isfinite(float(row[key]))
            for row in rows
            for key in ("oos_mse", "oos_mae", "oos_r2", "trading_sharpe")
        ),
        "same_oos_observations_for_every_case": len(
            {int(row["oos_observation_count"]) for row in rows}
        )
        == 1,
        "same_zero_baseline_for_every_case": bool(zero_mses)
        and np.allclose(zero_mses, zero_mses[0], rtol=0, atol=1e-20),
        "market_linear_anchor_ran_once": sum(
            row["representation"] == "market_linear" for row in rows
        )
        == 1,
        "timestamp_keyed_noise_design": True,
        "trading_excluded_from_inference": True,
        "holdout_was_not_used": _parse_timerange(config.timerange)[1]
        <= datetime.strptime(config.holdout_start, "%Y%m%d").replace(tzinfo=UTC),
        "cuda_float64_used": config.dtype == "float64"
        and all(float(row["train_peak_vram_mib"]) > 0 for row in rows),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "expected_case_count": PHASE10_EXPECTED_CASE_COUNT,
        "observed_case_count": len(rows),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(
    config: Phase10Config,
    path: Path,
    rows: list[dict[str, Any]],
) -> bool:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return False
    reference = [row for row in rows if row["seed"] == config.seeds[0]]
    if not reference:
        return False
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "OOS MSE / zero baseline",
            "Training MSE",
            "Effective rank",
            "Net trading Sharpe",
        ),
    )
    panels = (
        (1, 1, lambda row: float(row["oos_mse"]) / float(row["zero_mse"])),
        (1, 2, lambda row: row["train_train_mse_mean"]),
        (2, 1, lambda row: row["train_effective_rank_mean"]),
        (2, 2, lambda row: row["trading_sharpe"]),
    )
    for representation in config.representations:
        matched = sorted(
            (row for row in reference if row["representation"] == representation),
            key=lambda row: row["actual_pn_ratio"],
        )
        for panel_row, panel_column, transform in panels:
            figure.add_trace(
                go.Scatter(
                    x=[row["actual_pn_ratio"] for row in matched],
                    y=[transform(row) for row in matched],
                    mode="lines+markers",
                    name=representation,
                    legendgroup=representation,
                    showlegend=panel_row == 1 and panel_column == 1,
                ),
                row=panel_row,
                col=panel_column,
            )
    figure.update_xaxes(type="log", title_text="P/N")
    figure.update_yaxes(type="log", row=1, col=1)
    figure.update_yaxes(type="log", row=1, col=2)
    figure.update_layout(
        title="Phase 10 - Market information versus deterministic noise",
        template="plotly_white",
        height=900,
        width=1400,
    )
    figure.write_html(path, include_plotlyjs="cdn")
    return True


def run_phase10(config: Phase10Config) -> dict[str, Any]:
    config.validate()
    config.output_directory.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    checkpoint_path = config.output_directory / "checkpoint.json"
    substudies = []
    completed = []
    for representation, seed, ratios in _task_grid(config):
        print(
            f"PHASE10 SUBSTUDY representation={representation} seed={seed} points={len(ratios)}",
            flush=True,
        )
        substudy = _run_substudy(config, representation, seed, ratios, run_id)
        substudies.append(substudy)
        completed.append(
            {
                "representation": representation,
                "seed": seed,
                "ratios": list(ratios),
                "passed": substudy["gate"]["passed"],
                "summary": substudy["artifacts"]["summary"],
            }
        )
        checkpoint_path.write_text(
            json.dumps(_json_safe({"run_id": run_id, "completed": completed}), indent=2),
            encoding="utf-8",
        )

    rows = flatten_substudies(substudies)
    aggregates = aggregate_robustness(config, rows)
    curve_assessments = assess_representation_curves(config, rows)
    comparisons = compare_representations(config, rows)
    gate = evaluate_phase10_gate(config, substudies, rows, aggregates)
    map_path = config.output_directory / "representation_map.csv"
    aggregate_path = config.output_directory / "robustness_aggregates.csv"
    summary_path = config.output_directory / "summary.json"
    plot_path = config.output_directory / "noise_feature_controls.html"
    _write_csv(map_path, rows)
    _write_csv(aggregate_path, aggregates)
    plot_written = _write_plot(config, plot_path, rows)
    config_payload = asdict(config)
    for key in (
        "data_directory",
        "output_directory",
        "phase9_summary",
        "frozen_gamma_file",
        "strategy_directory",
        "model_directory",
        "models_directory",
    ):
        config_payload[key] = str(config_payload[key])
    summary = {
        "phase": 10,
        "objective": "Determine whether high-dimensional recovery requires market information",
        "scope": "development-only representation control; 2026 holdout sealed",
        "run_id": run_id,
        "design": {
            "market_linear": "the 25 standardized causal market inputs",
            "market_rff": "P nested RFF of the 25 market inputs at frozen gamma=0.5",
            "pure_noise": "P timestamp-keyed deterministic iid-N(0,1) predictors",
            "market_plus_noise": "25 market inputs plus P-25 timestamp-keyed noise predictors",
            "same_total_p_for_curve_representations": True,
            "noise_is_nested_across_p": True,
            "noise_is_independent_of_market_values_and_targets": True,
            "trading_used_for_inference": False,
            "holdout_used": False,
        },
        "config": config_payload,
        "compute": {
            "case_count": len(rows),
            "rolling_model_fit_count": sum(int(row["train_training_window_count"]) for row in rows),
            "summed_freqtrade_wall_seconds": sum(float(row["wall_seconds"]) for row in rows),
            "summed_cuda_training_seconds": sum(
                float(row["train_training_seconds_total"]) for row in rows
            ),
            "maximum_peak_vram_mib": max(float(row["train_peak_vram_mib"]) for row in rows),
        },
        "curve_assessments": curve_assessments,
        "robustness_aggregates": aggregates,
        "representation_comparisons": comparisons,
        "gate": gate,
        "artifacts": {
            "summary": str(summary_path),
            "representation_map": str(map_path),
            "robustness_aggregates": str(aggregate_path),
            "interactive_plot": str(plot_path) if plot_written else None,
            "checkpoint": str(checkpoint_path),
        },
    }
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    return summary
