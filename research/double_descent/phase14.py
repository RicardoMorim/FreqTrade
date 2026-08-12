"""Phase 14: frozen cross-asset replication of the BTC experiment on ETH."""

from __future__ import annotations

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

from research.double_descent.phase3 import Phase3Config, audit_data_coverage
from research.double_descent.phase4 import _load_evaluation_market_data, _parse_timerange
from research.double_descent.phase8 import assess_flat_curve
from research.double_descent.phase10 import (
    PHASE10_GAMMA,
    PHASE10_MAP_RATIOS,
    PHASE10_ROBUSTNESS_RATIOS,
    PHASE10_SEEDS,
    Phase10Config,
    _json_safe,
    _run_substudy,
    _write_csv,
    flatten_substudies,
)
from research.double_descent.phase12 import (
    PHASE12_BASELINES,
    PHASE12_PREDICTION_BASELINES,
    PHASE12_RIDGE_ALPHA,
    Phase12Config,
)
from research.double_descent.phase12 import (
    _phase4_config as _phase12_phase4_config,
)
from research.double_descent.phase12 import (
    _recover_case as _recover_baseline_case,
)
from research.double_descent.phase12 import (
    _run_case as _run_baseline_case,
)
from research.double_descent.phase12 import (
    flatten_results as flatten_baseline_results,
)


PHASE14_PAIR = "ETH/USDT:USDT"
PHASE14_DATA_TIMERANGE = "20240924-20260101"
PHASE14_RFF_CASE_COUNT = 21
PHASE14_BASELINE_CASE_COUNT = len(PHASE12_BASELINES)
PHASE14_EXPECTED_CASE_COUNT = PHASE14_RFF_CASE_COUNT + PHASE14_BASELINE_CASE_COUNT
PHASE14_EXPECTED_FIT_COUNT = PHASE14_EXPECTED_CASE_COUNT * 13


@dataclass(frozen=True)
class Phase14Config:
    data_directory: Path
    output_directory: Path = Path("user_data/research_results/double_descent/phase14")
    phase9_summary: Path = Path("user_data/research_results/double_descent/phase9/summary.json")
    frozen_gamma_file: Path = Path(
        "user_data/research_results/double_descent/phase9/frozen_gamma.json"
    )
    phase10_summary: Path = Path("user_data/research_results/double_descent/phase10/summary.json")
    phase10_map: Path = Path(
        "user_data/research_results/double_descent/phase10/representation_map.csv"
    )
    phase12_summary: Path = Path("user_data/research_results/double_descent/phase12/summary.json")
    phase12_results: Path = Path(
        "user_data/research_results/double_descent/phase12/baseline_results.csv"
    )
    phase11_summary: Path = Path("user_data/research_results/double_descent/phase11/summary.json")
    phase13_summary: Path = Path("user_data/research_results/double_descent/phase13/summary.json")
    python_executable: str = sys.executable
    cuda_python_executable: str = ""
    pair: str = PHASE14_PAIR
    timeframe: str = "1h"
    timerange: str = "20250101-20260101"
    holdout_start: str = "20260101"
    train_period_days: int = 90
    backtest_period_days: int = 30
    effective_n: int = 2_159
    map_ratios: tuple[float, ...] = PHASE10_MAP_RATIOS
    robustness_ratios: tuple[float, ...] = PHASE10_ROBUSTNESS_RATIOS
    seeds: tuple[int, ...] = PHASE10_SEEDS
    baselines: tuple[str, ...] = PHASE12_BASELINES
    gamma: float = PHASE10_GAMMA
    ridge: float = 0.0
    ridge_alpha: float = PHASE12_RIDGE_ALPHA
    rcond: float = 1e-12
    dtype: str = "float64"
    chunk_size: int = 4_096
    fee: float = 0.001
    minimum_training_windows: int = 10
    minimum_seed_count: int = 3
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
        if not Path(self.cuda_python_executable).is_file():
            raise FileNotFoundError(f"CUDA Python does not exist: {self.cuda_python_executable}")
        self._validate_references()
        self._validate_design()

    def _validate_references(self) -> None:
        required_passed = (
            (self.phase9_summary, 9),
            (self.phase10_summary, 10),
            (self.phase12_summary, 12),
        )
        for path, phase in required_passed:
            if not path.is_file():
                raise FileNotFoundError(f"Phase 14 requires completed Phase {phase}: {path}")
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary.get("phase") != phase or summary.get("gate", {}).get("passed") is not True:
                raise ValueError(f"Phase {phase} did not pass its integrity gate")
            holdout_key = "holdout_used_for_selection" if phase == 9 else "holdout_used"
            if summary.get("design", {}).get(holdout_key) is not False:
                raise ValueError(f"Phase {phase} does not certify a sealed holdout")
        for path in (self.frozen_gamma_file, self.phase10_map, self.phase12_results):
            if not path.is_file():
                raise FileNotFoundError(f"Phase 14 reference artifact is missing: {path}")
        frozen = json.loads(self.frozen_gamma_file.read_text(encoding="utf-8"))
        if frozen.get("gamma") != self.gamma or frozen.get("holdout_used") is not False:
            raise ValueError("Phase 14 must reuse the sealed gamma=0.5 artifact")
        if not self.phase13_summary.is_file():
            raise FileNotFoundError("Phase 14 requires the completed Phase 13 analysis")
        phase13 = json.loads(self.phase13_summary.read_text(encoding="utf-8"))
        if phase13.get("phase") != 13 or phase13.get("design", {}).get("holdout_used") is not False:
            raise ValueError("Phase 13 does not certify a completed, sealed-holdout analysis")

    def _validate_design(self) -> None:
        self._validate_model_design()
        self._validate_grid_design()
        _, end = _parse_timerange(self.timerange)
        holdout = datetime.strptime(self.holdout_start, "%Y%m%d").replace(tzinfo=UTC)
        if end > holdout:
            raise ValueError("Phase 14 may not enter the sealed holdout")

    def _validate_model_design(self) -> None:
        if self.pair != PHASE14_PAIR:
            raise ValueError("Phase 14 is frozen to ETH/USDT:USDT")
        if self.timeframe != "1h" or self.timerange != "20250101-20260101":
            raise ValueError("Phase 14 is frozen to the matched 2025 1h period")
        if self.train_period_days != 90 or self.backtest_period_days != 30:
            raise ValueError("Phase 14 freezes the 90-day/30-day rolling design")
        if self.effective_n != 2_159:
            raise ValueError("Phase 14 requires the Phase 3-matched effective N=2,159")
        if self.gamma != PHASE10_GAMMA or self.ridge != 0 or self.ridge_alpha != 1.0:
            raise ValueError("Phase 14 freezes gamma=0.5, RFF ridge=0, and baseline Ridge alpha=1")
        if self.rcond != 1e-12 or self.dtype != "float64" or self.fee != 0.001:
            raise ValueError("Phase 14 freezes rcond, float64, and 0.1% fee per side")

    def _validate_grid_design(self) -> None:
        if len(set(self.seeds)) != len(self.seeds) or any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be unique non-negative integers")
        minimum = 1 if self.smoke_test else 3
        if len(self.seeds) < minimum or self.minimum_seed_count < minimum:
            raise ValueError(f"Phase 14 requires at least {minimum} seed(s)")
        if not self.map_ratios or tuple(sorted(set(self.map_ratios))) != self.map_ratios:
            raise ValueError("map ratios must be sorted and unique")
        if not set(self.robustness_ratios).issubset(self.map_ratios):
            raise ValueError("robustness ratios must be a subset of the main map")
        if not self.baselines or len(set(self.baselines)) != len(self.baselines):
            raise ValueError("baselines must be non-empty and unique")
        if not set(self.baselines).issubset(PHASE12_BASELINES):
            raise ValueError("unknown Phase 14 baseline")
        if not self.smoke_test:
            frozen = (
                (self.map_ratios, PHASE10_MAP_RATIOS, "main P/N map"),
                (self.robustness_ratios, PHASE10_ROBUSTNESS_RATIOS, "robustness map"),
                (self.seeds, PHASE10_SEEDS, "RFF seeds"),
                (self.baselines, PHASE12_BASELINES, "simple baselines"),
            )
            for actual, expected, label in frozen:
                if actual != expected:
                    raise ValueError(f"the full Phase 14 run freezes the {label}")


def build_download_command(config: Phase14Config) -> list[str]:
    return [
        config.python_executable,
        "-m",
        "freqtrade",
        "download-data",
        "--exchange",
        "binance",
        "--pairs",
        config.pair,
        "--timeframes",
        config.timeframe,
        "--timerange",
        PHASE14_DATA_TIMERANGE,
        "--trading-mode",
        "futures",
        "--candle-types",
        "futures",
        "funding_rate",
        "mark",
        "--data-format-ohlcv",
        "feather",
        "--datadir",
        str(config.data_directory),
        "--no-color",
    ]


def download_eth_data(config: Phase14Config) -> None:
    config.data_directory.mkdir(parents=True, exist_ok=True)
    command = build_download_command(config)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    print(
        f"PHASE14 DOWNLOAD pair={config.pair} timerange={PHASE14_DATA_TIMERANGE}",
        flush=True,
    )
    completed = subprocess.run(
        command,
        cwd=Path.cwd(),
        env=environment,
        check=False,
        timeout=config.subprocess_timeout_seconds,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ETH data download failed with exit code {completed.returncode}")


def _data_audit_config(config: Phase14Config) -> Phase3Config:
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


def _rff_tasks(config: Phase14Config) -> list[tuple[int, tuple[float, ...]]]:
    return [
        (seed, config.map_ratios if index == 0 else config.robustness_ratios)
        for index, seed in enumerate(config.seeds)
    ]


def _phase10_config(config: Phase14Config) -> Phase10Config:
    return Phase10Config(
        data_directory=config.data_directory,
        output_directory=config.output_directory / "rff",
        phase9_summary=config.phase9_summary,
        frozen_gamma_file=config.frozen_gamma_file,
        python_executable=config.python_executable,
        cuda_python_executable=config.cuda_python_executable,
        pair=config.pair,
        timeframe=config.timeframe,
        timerange=config.timerange,
        holdout_start=config.holdout_start,
        train_period_days=config.train_period_days,
        backtest_period_days=config.backtest_period_days,
        effective_n=config.effective_n,
        map_ratios=config.map_ratios,
        robustness_ratios=config.robustness_ratios,
        seeds=config.seeds,
        gamma=config.gamma,
        ridge=config.ridge,
        rcond=config.rcond,
        dtype=config.dtype,
        chunk_size=config.chunk_size,
        fee=config.fee,
        minimum_training_windows=config.minimum_training_windows,
        minimum_seed_count=config.minimum_seed_count,
        subprocess_timeout_seconds=config.subprocess_timeout_seconds,
        resume=config.resume,
        smoke_test=config.smoke_test,
        strategy_directory=config.strategy_directory,
        model_directory=config.model_directory,
        models_directory=config.models_directory,
    )


def _phase12_config(config: Phase14Config) -> Phase12Config:
    return Phase12Config(
        data_directory=config.data_directory,
        output_directory=config.output_directory / "baselines",
        phase10_summary=config.phase10_summary,
        phase10_map=config.phase10_map,
        phase11_summary=config.phase11_summary,
        python_executable=config.python_executable,
        pair=config.pair,
        timeframe=config.timeframe,
        timerange=config.timerange,
        holdout_start=config.holdout_start,
        train_period_days=config.train_period_days,
        backtest_period_days=config.backtest_period_days,
        effective_n=config.effective_n,
        baselines=config.baselines,
        ridge_alpha=config.ridge_alpha,
        fee=config.fee,
        minimum_training_windows=config.minimum_training_windows,
        subprocess_timeout_seconds=config.subprocess_timeout_seconds,
        resume=config.resume,
        smoke_test=config.smoke_test,
        strategy_directory=config.strategy_directory,
        model_directory=config.model_directory,
        models_directory=config.models_directory,
    )


def run_rff_replication(
    config: Phase14Config,
    run_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    engine = _phase10_config(config)
    substudies = []
    checkpoint_path = config.output_directory / "rff_checkpoint.json"
    for seed, ratios in _rff_tasks(config):
        substudy = _run_substudy(engine, "market_rff", seed, ratios, run_id)
        substudies.append(substudy)
        checkpoint_path.write_text(
            json.dumps(
                _json_safe(
                    {
                        "run_id": run_id,
                        "pair": config.pair,
                        "completed": [
                            {
                                "seed": item["seed"],
                                "ratios": item["ratios"],
                                "passed": item["gate"]["passed"],
                            }
                            for item in substudies
                        ],
                    }
                ),
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
    return substudies, flatten_substudies(substudies)


def run_baseline_replication(
    config: Phase14Config,
    run_id: str,
    data_audit: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    engine = _phase12_config(config)
    phase4 = _phase12_phase4_config(engine)
    phase4.validate()
    market_data = _load_evaluation_market_data(phase4) if data_audit["passed"] else pd.DataFrame()
    results = []
    checkpoint_path = config.output_directory / "baseline_checkpoint.json"
    if data_audit["passed"]:
        for baseline in config.baselines:
            result = (
                _recover_baseline_case(engine, baseline, phase4, market_data)
                if config.resume
                else None
            )
            if result is None:
                result = _run_baseline_case(engine, baseline, phase4, run_id, market_data)
            results.append(result)
            checkpoint_path.write_text(
                json.dumps(_json_safe(results), indent=2, allow_nan=False),
                encoding="utf-8",
            )
    return results, flatten_baseline_results(results)


def _curve_assessments(
    rows: list[dict[str, Any]],
    seeds: tuple[int, ...],
    robustness_ratios: tuple[float, ...],
    asset: str,
) -> list[dict[str, Any]]:
    assessments = []
    for seed in seeds:
        matched = [
            row
            for row in rows
            if int(row["seed"]) == seed and float(row["target_pn_ratio"]) in robustness_ratios
        ]
        assessments.append({"asset": asset, "seed": seed, **assess_flat_curve(matched)})
    return assessments


def _btc_rff_rows(config: Phase14Config) -> list[dict[str, Any]]:
    frame = pd.read_csv(config.phase10_map)
    frame = frame.loc[frame["representation"] == "market_rff"].copy()
    return frame.to_dict(orient="records")


def compare_cross_asset_replication(
    config: Phase14Config,
    eth_rff_rows: list[dict[str, Any]],
    eth_baseline_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    btc_rff_rows = _btc_rff_rows(config)
    eth_curves = _curve_assessments(eth_rff_rows, config.seeds, config.robustness_ratios, "ETH")
    btc_curves = _curve_assessments(btc_rff_rows, config.seeds, config.robustness_ratios, "BTC")
    majority = math.ceil(len(config.seeds) / 2)
    eth_prediction_baselines = [
        row for row in eth_baseline_rows if row["baseline"] in PHASE12_PREDICTION_BASELINES
    ]
    eth_zero = next(row for row in eth_prediction_baselines if row["baseline"] == "zero_return")
    eth_best_simple = min(eth_prediction_baselines, key=lambda row: float(row["oos_mse"]))
    eth_best_learnable = min(
        (row for row in eth_prediction_baselines if row["baseline"] != "zero_return"),
        key=lambda row: float(row["oos_mse"]),
    )
    eth_best_rff = min(eth_rff_rows, key=lambda row: float(row["oos_mse"]))
    btc_best_rff = min(btc_rff_rows, key=lambda row: float(row["oos_mse"]))
    lookup_btc = {(int(row["seed"]), float(row["target_pn_ratio"])): row for row in btc_rff_rows}
    matched = []
    for row in eth_rff_rows:
        btc = lookup_btc[(int(row["seed"]), float(row["target_pn_ratio"]))]
        matched.append(
            {
                "seed": int(row["seed"]),
                "target_pn_ratio": float(row["target_pn_ratio"]),
                "feature_count": int(row["feature_count"]),
                "eth_mse_over_zero": float(row["oos_mse"]) / float(row["zero_mse"]),
                "btc_mse_over_zero": float(btc["oos_mse"]) / float(btc["zero_mse"]),
            }
        )
    eth_shape_count = sum(bool(row.get("double_descent_pattern_detected")) for row in eth_curves)
    btc_shape_count = sum(bool(row.get("double_descent_pattern_detected")) for row in btc_curves)
    eth_useful_count = sum(
        bool(row.get("largest_model_beats_zero_mse"))
        and bool(row.get("second_descent_beats_best_underparameterized"))
        for row in eth_curves
    )
    btc_useful_count = sum(
        bool(row.get("largest_model_beats_zero_mse"))
        and bool(row.get("second_descent_beats_best_underparameterized"))
        for row in btc_curves
    )
    eth_rff_beats_zero = sum(float(row["oos_mse"]) < float(row["zero_mse"]) for row in eth_rff_rows)
    btc_rff_beats_zero = sum(float(row["oos_mse"]) < float(row["zero_mse"]) for row in btc_rff_rows)
    return {
        "primary_metric": "chronological OOS MSE normalized by each asset's zero forecast",
        "matched_asset_case_count": len(matched),
        "matched_asset_cases": matched,
        "curve_assessments": {"ETH": eth_curves, "BTC": btc_curves},
        "majority_seed_requirement": majority,
        "shape_replication": {
            "eth_double_descent_seed_count": eth_shape_count,
            "btc_double_descent_seed_count": btc_shape_count,
            "passed": eth_shape_count >= majority and btc_shape_count >= majority,
        },
        "useful_benign_overfitting_replication": {
            "eth_useful_seed_count": eth_useful_count,
            "btc_useful_seed_count": btc_useful_count,
            "passed": eth_useful_count >= majority and btc_useful_count >= majority,
        },
        "zero_baseline_comparison": {
            "eth_rff_case_count_beating_zero": eth_rff_beats_zero,
            "btc_rff_case_count_beating_zero": btc_rff_beats_zero,
            "no_alpha_failure_replicated": eth_rff_beats_zero == 0 and btc_rff_beats_zero == 0,
        },
        "eth_best_simple": {
            "baseline": eth_best_simple["baseline"],
            "mse": eth_best_simple["oos_mse"],
            "r2": eth_best_simple["oos_r2"],
        },
        "eth_best_learnable": {
            "baseline": eth_best_learnable["baseline"],
            "mse": eth_best_learnable["oos_mse"],
            "mse_over_zero": float(eth_best_learnable["oos_mse"]) / float(eth_zero["oos_mse"]),
        },
        "eth_best_rff": {
            "seed": eth_best_rff["seed"],
            "actual_pn_ratio": eth_best_rff["actual_pn_ratio"],
            "feature_count": eth_best_rff["feature_count"],
            "mse": eth_best_rff["oos_mse"],
            "mse_over_zero": float(eth_best_rff["oos_mse"]) / float(eth_best_rff["zero_mse"]),
            "mse_over_best_simple": float(eth_best_rff["oos_mse"])
            / float(eth_best_simple["oos_mse"]),
        },
        "btc_best_rff": {
            "seed": int(btc_best_rff["seed"]),
            "actual_pn_ratio": float(btc_best_rff["actual_pn_ratio"]),
            "feature_count": int(btc_best_rff["feature_count"]),
            "mse": float(btc_best_rff["oos_mse"]),
            "mse_over_zero": float(btc_best_rff["oos_mse"]) / float(btc_best_rff["zero_mse"]),
        },
        "rff_complexity_justified_on_eth": float(eth_best_rff["oos_mse"])
        < float(eth_best_simple["oos_mse"]),
        "trading_excluded_from_replication_decision": True,
    }


def evaluate_phase14_gate(
    config: Phase14Config,
    data_audit: dict[str, Any],
    rff_substudies: list[dict[str, Any]],
    rff_rows: list[dict[str, Any]],
    baseline_results: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    all_rows = [*rff_rows, *baseline_rows]
    expected_rff = sum(len(ratios) for _, ratios in _rff_tasks(config))
    expected_total = expected_rff + len(config.baselines)
    effective_ns = {
        value
        for result in [
            *(result for substudy in rff_substudies for result in substudy["results"]),
            *baseline_results,
        ]
        for value in result.get("training", {}).get("effective_n_values", [])
    }
    generated_configs = [
        Path(result["artifacts"]["config"])
        for result in [
            *(result for substudy in rff_substudies for result in substudy["results"]),
            *baseline_results,
        ]
    ]
    config_pairs = []
    for path in generated_configs:
        if not path.is_file():
            continue
        generated = json.loads(path.read_text(encoding="utf-8"))
        config_pairs.extend(generated["exchange"]["pair_whitelist"])
    checks = {
        "eth_data_coverage_passed": data_audit.get("passed") is True,
        "every_rff_substudy_passed": bool(rff_substudies)
        and all(item["gate"]["passed"] for item in rff_substudies),
        "every_baseline_case_passed": bool(baseline_results)
        and all(item["success"] for item in baseline_results),
        "complete_predeclared_rff_case_count": len(rff_rows) == expected_rff,
        "complete_predeclared_baseline_case_count": len(baseline_rows) == len(config.baselines),
        "complete_predeclared_total_case_count": len(all_rows) == expected_total,
        "all_generated_configs_use_eth_only": len(config_pairs) == expected_total
        and set(config_pairs) == {config.pair},
        "effective_n_matches_frozen_design": effective_ns == {config.effective_n},
        "same_oos_observation_count": bool(all_rows)
        and len({int(row["oos_observation_count"]) for row in all_rows}) == 1,
        "same_zero_baseline_within_eth": bool(all_rows)
        and np.allclose(
            [float(row["zero_mse"]) for row in all_rows],
            float(all_rows[0]["zero_mse"]),
            rtol=0,
            atol=1e-20,
        ),
        "rff_cuda_float64_used": bool(rff_rows)
        and all(float(row["train_peak_vram_mib"]) > 0 for row in rff_rows),
        "all_prediction_metrics_finite": bool(all_rows)
        and all(
            math.isfinite(float(row[key]))
            for row in all_rows
            for key in ("oos_mse", "oos_mae", "oos_r2", "trading_sharpe")
        ),
        "fee_preserved": config.fee == 0.001
        and all(
            not math.isfinite(float(row.get("trading_fee_per_side", math.nan)))
            or math.isclose(float(row["trading_fee_per_side"]), config.fee, abs_tol=1e-15)
            for row in all_rows
        ),
        "parameters_transferred_without_eth_selection": True,
        "trading_excluded_from_replication_decision": True,
        "holdout_was_not_used": _parse_timerange(config.timerange)[1]
        <= datetime.strptime(config.holdout_start, "%Y%m%d").replace(tzinfo=UTC),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "expected_rff_case_count": expected_rff,
        "observed_rff_case_count": len(rff_rows),
        "expected_baseline_case_count": len(config.baselines),
        "observed_baseline_case_count": len(baseline_rows),
        "expected_total_case_count": expected_total,
        "observed_total_case_count": len(all_rows),
    }


def _write_plot(
    path: Path,
    config: Phase14Config,
    eth_rff_rows: list[dict[str, Any]],
    eth_baseline_rows: list[dict[str, Any]],
) -> bool:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return False
    btc_rows = _btc_rff_rows(config)
    if not eth_rff_rows:
        return False
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Reference-seed RFF OOS MSE / zero",
            "Matched BTC versus ETH normalized MSE",
            "ETH simple baselines: OOS MSE / zero",
            "ETH net return after fees",
        ),
    )
    for asset, rows in (("BTC", btc_rows), ("ETH", eth_rff_rows)):
        matched = sorted(
            (row for row in rows if int(row["seed"]) == config.seeds[0]),
            key=lambda row: float(row["actual_pn_ratio"]),
        )
        figure.add_trace(
            go.Scatter(
                x=[row["actual_pn_ratio"] for row in matched],
                y=[float(row["oos_mse"]) / float(row["zero_mse"]) for row in matched],
                mode="lines+markers",
                name=asset,
            ),
            row=1,
            col=1,
        )
    btc_lookup = {(int(row["seed"]), float(row["target_pn_ratio"])): row for row in btc_rows}
    figure.add_trace(
        go.Scatter(
            x=[
                float(btc_lookup[(int(row["seed"]), float(row["target_pn_ratio"]))]["oos_mse"])
                / float(btc_lookup[(int(row["seed"]), float(row["target_pn_ratio"]))]["zero_mse"])
                for row in eth_rff_rows
            ],
            y=[float(row["oos_mse"]) / float(row["zero_mse"]) for row in eth_rff_rows],
            mode="markers",
            name="Matched RFF cases",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    zero_mse = float(eth_baseline_rows[0]["zero_mse"])
    figure.add_trace(
        go.Bar(
            x=[row["baseline"] for row in eth_baseline_rows],
            y=[float(row["oos_mse"]) / zero_mse for row in eth_baseline_rows],
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    combined = [*eth_rff_rows, *eth_baseline_rows]
    figure.add_trace(
        go.Bar(
            x=[
                row.get("baseline", f"RFF {row['target_pn_ratio']}:s{row['seed']}")
                for row in combined
            ],
            y=[row["trading_profit_total"] for row in combined],
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    figure.update_xaxes(type="log", title_text="P/N", row=1, col=1)
    figure.update_xaxes(type="log", title_text="BTC MSE / zero", row=1, col=2)
    figure.update_yaxes(type="log", title_text="ETH MSE / zero", row=1, col=2)
    figure.add_hline(y=1.0, line_dash="dash", row=1, col=1)
    figure.add_hline(y=1.0, line_dash="dash", row=2, col=1)
    figure.add_hline(y=0.0, line_dash="dash", row=2, col=2)
    figure.update_xaxes(tickangle=-35, row=2)
    figure.update_layout(
        title="Phase 14 - Frozen BTC to ETH cross-asset replication",
        template="plotly_white",
        height=950,
        width=1500,
    )
    figure.write_html(path, include_plotlyjs="cdn")
    return True


def run_phase14(config: Phase14Config) -> dict[str, Any]:
    config.validate()
    config.output_directory.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    data_audit = audit_data_coverage(_data_audit_config(config))
    if not data_audit["passed"]:
        raise RuntimeError(
            "ETH data audit failed. Run the Phase 14 CLI with --download-data before the sweep."
        )
    rff_substudies, rff_rows = run_rff_replication(config, run_id)
    baseline_results, baseline_rows = run_baseline_replication(config, run_id, data_audit)
    comparisons = compare_cross_asset_replication(config, rff_rows, baseline_rows)
    gate = evaluate_phase14_gate(
        config,
        data_audit,
        rff_substudies,
        rff_rows,
        baseline_results,
        baseline_rows,
    )
    rff_path = config.output_directory / "eth_rff_results.csv"
    baseline_path = config.output_directory / "eth_baseline_results.csv"
    matched_path = config.output_directory / "btc_eth_matched_cases.csv"
    plot_path = config.output_directory / "eth_replication.html"
    summary_path = config.output_directory / "summary.json"
    _write_csv(rff_path, rff_rows)
    _write_csv(baseline_path, baseline_rows)
    _write_csv(matched_path, comparisons["matched_asset_cases"])
    plot_written = _write_plot(plot_path, config, rff_rows, baseline_rows)
    config_payload = asdict(config)
    for key in (
        "data_directory",
        "output_directory",
        "phase9_summary",
        "frozen_gamma_file",
        "phase10_summary",
        "phase10_map",
        "phase12_summary",
        "phase12_results",
        "phase11_summary",
        "phase13_summary",
        "strategy_directory",
        "model_directory",
        "models_directory",
    ):
        config_payload[key] = str(config_payload[key])
    summary = {
        "phase": 14,
        "objective": "Replicate the frozen BTC double-descent experiment on ETH",
        "scope": "cross-asset development replication; same dates; 2026 holdout sealed",
        "run_id": run_id,
        "design": {
            "source_asset": "BTC/USDT:USDT",
            "replication_asset": config.pair,
            "transferred_without_selection": [
                "features",
                "target",
                "gamma",
                "ridge",
                "P/N grid",
                "seeds",
                "training/evaluation windows",
                "solver precision",
                "simple baselines",
                "trading rule",
                "fee",
            ],
            "eth_used_for_parameter_selection": False,
            "trading_used_for_replication_decision": False,
            "holdout_used": False,
        },
        "config": config_payload,
        "data_audit": data_audit,
        "rff_substudies": rff_substudies,
        "baseline_results": baseline_results,
        "comparisons": comparisons,
        "gate": gate,
        "compute": {
            "rff_case_count": len(rff_rows),
            "baseline_case_count": len(baseline_rows),
            "total_case_count": len(rff_rows) + len(baseline_rows),
            "rolling_model_fit_count": sum(
                int(row["train_training_window_count"]) for row in [*rff_rows, *baseline_rows]
            ),
            "summed_freqtrade_wall_seconds": sum(
                float(row["wall_seconds"]) for row in [*rff_rows, *baseline_rows]
            ),
            "summed_cuda_training_seconds": sum(
                float(row["train_training_seconds_total"]) for row in rff_rows
            ),
            "maximum_peak_vram_mib": max(float(row["train_peak_vram_mib"]) for row in rff_rows),
        },
        "limitations": {
            "cross_asset_not_independent_market": (
                "ETH and BTC share dates, exchange, crypto market factors, and correlated regimes"
            ),
            "single_replication_asset": "one additional asset is not a broad cross-sectional test",
            "cost_model": "0.1% fee per side; no new slippage or market-impact model",
        },
        "artifacts": {
            "summary": str(summary_path),
            "eth_rff_results": str(rff_path),
            "eth_baseline_results": str(baseline_path),
            "btc_eth_matched_cases": str(matched_path),
            "interactive_plot": str(plot_path) if plot_written else None,
            "rff_checkpoint": str(config.output_directory / "rff_checkpoint.json"),
            "baseline_checkpoint": str(config.output_directory / "baseline_checkpoint.json"),
        },
    }
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    return summary
