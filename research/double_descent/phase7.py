"""Phase 7: explicit Ridge regularization map across P/N and RFF seeds."""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.double_descent.phase3 import PN_RATIOS
from research.double_descent.phase4 import Phase4Config, run_phase4
from research.double_descent.phase5 import PHASE5_SEEDS, summarize_values


RIDGE_LAMBDAS = (0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 100.0)
PHASE7_SEEDS = PHASE5_SEEDS[:3]
ROBUSTNESS_RATIOS = (0.1, 1.0, 1.02, 50.0)
STRONG_RIDGE_LAMBDA = 1.0
REPRODUCTION_RELATIVE_TOLERANCE = 1e-8

ROBUSTNESS_METRICS = (
    "oos_mse",
    "oos_mae",
    "oos_r2",
    "oos_information_coefficient",
    "oos_spearman_information_coefficient",
    "oos_directional_accuracy",
    "oos_prediction_standard_deviation",
    "train_train_mse_mean",
    "train_interpolated_window_fraction",
    "train_normalized_feature_coefficient_norm_mean",
    "train_ridge_effective_degrees_of_freedom_mean",
    "train_ridge_system_condition_number_mean",
    "trading_total_trades",
    "trading_profit_total",
    "trading_sharpe",
    "trading_sortino",
    "trading_profit_factor",
    "trading_max_drawdown_account",
    "trading_turnover_multiple",
)

REPRODUCTION_METRICS = (
    "oos_mse",
    "oos_prediction_mean",
    "oos_prediction_standard_deviation",
    "trading_profit_total",
)


@dataclass(frozen=True)
class Phase7Config:
    data_directory: Path
    output_directory: Path = Path("user_data/research_results/double_descent/phase7")
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
    map_ratios: tuple[float, ...] = PN_RATIOS
    robustness_ratios: tuple[float, ...] = ROBUSTNESS_RATIOS
    ridge_lambdas: tuple[float, ...] = RIDGE_LAMBDAS
    seeds: tuple[int, ...] = PHASE7_SEEDS
    gamma: float = 0.2
    rcond: float = 1e-12
    dtype: str = "float64"
    chunk_size: int = 4_096
    fee: float = 0.001
    minimum_training_windows: int = 10
    minimum_seed_count: int = 3
    subprocess_timeout_seconds: int = 1_800
    resume: bool = True
    smoke_test: bool = False

    def validate(self) -> None:
        if not self.phase5_summary.is_file():
            raise FileNotFoundError(f"Phase 5 summary does not exist: {self.phase5_summary}")
        minimum = 1 if self.smoke_test else 3
        if self.minimum_seed_count < minimum or len(self.seeds) < self.minimum_seed_count:
            raise ValueError(f"Phase 7 requires at least {minimum} seed(s) for this run")
        if len(set(self.seeds)) != len(self.seeds) or any(seed < 0 for seed in self.seeds):
            raise ValueError("Phase 7 seeds must be unique non-negative integers")
        if not self.map_ratios or tuple(sorted(set(self.map_ratios))) != self.map_ratios:
            raise ValueError("map ratios must be sorted and unique")
        if not set(self.robustness_ratios).issubset(self.map_ratios):
            raise ValueError("robustness ratios must be a subset of the main map")
        if not self.ridge_lambdas or self.ridge_lambdas[0] != 0:
            raise ValueError("the Ridge grid must begin with the ridgeless lambda=0 control")
        if tuple(sorted(set(self.ridge_lambdas))) != self.ridge_lambdas:
            raise ValueError("Ridge lambdas must be sorted and unique")
        if any(value < 0 for value in self.ridge_lambdas):
            raise ValueError("Ridge lambdas must be non-negative")
        _phase4_config(
            self,
            seed=self.seeds[0],
            ridge=self.ridge_lambdas[0],
            ratios=self.map_ratios,
            output_directory=self.output_directory / "validation",
        ).validate()


def _ridge_slug(ridge: float) -> str:
    return "0" if ridge == 0 else f"{ridge:.0e}".replace("+", "")


def _phase4_config(
    config: Phase7Config,
    seed: int,
    ridge: float,
    ratios: tuple[float, ...],
    output_directory: Path,
) -> Phase4Config:
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
        ridge=ridge,
        rcond=config.rcond,
        dtype=config.dtype,
        chunk_size=config.chunk_size,
        fee=config.fee,
        minimum_training_windows=config.minimum_training_windows,
        subprocess_timeout_seconds=config.subprocess_timeout_seconds,
        resume=config.resume,
    )


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


def load_compatible_phase5_summary(config: Phase7Config) -> dict[str, Any]:
    summary = json.loads(config.phase5_summary.read_text(encoding="utf-8"))
    if not summary.get("gate", {}).get("passed"):
        raise ValueError("Phase 5 reference did not pass its integrity gate")
    reference = summary["config"]
    expected = {
        "pair": config.pair,
        "timeframe": config.timeframe,
        "timerange": config.timerange,
        "holdout_start": config.holdout_start,
        "train_period_days": config.train_period_days,
        "backtest_period_days": config.backtest_period_days,
        "effective_n": config.effective_n,
        "gamma": config.gamma,
        "rcond": config.rcond,
        "dtype": config.dtype,
        "fee": config.fee,
        "ridge": 0.0,
    }
    mismatches = {
        key: {"expected": value, "actual": reference.get(key)}
        for key, value in expected.items()
        if reference.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Phase 5 reference is incompatible: {mismatches}")
    if not set(config.seeds).issubset(reference["seeds"]):
        raise ValueError("Phase 7 seeds are not all present in the Phase 5 reference")
    if not set(config.map_ratios).issubset(reference["ratios"]):
        raise ValueError("Phase 7 ratios are not all present in the Phase 5 reference")
    return summary


def _run_substudy(
    config: Phase7Config,
    seed: int,
    ridge: float,
) -> dict[str, Any]:
    ratios = config.map_ratios if seed == config.seeds[0] else config.robustness_ratios
    directory = config.output_directory / "runs" / f"seed-{seed}" / f"lambda-{_ridge_slug(ridge)}"
    print(
        f"PHASE7 START seed={seed} lambda={ridge:g} points={len(ratios)}",
        flush=True,
    )
    summary = run_phase4(_phase4_config(config, seed, ridge, ratios, directory))
    print(
        f"PHASE7 DONE seed={seed} lambda={ridge:g} passed={summary['gate']['passed']}",
        flush=True,
    )
    return summary


def flatten_substudies(
    config: Phase7Config,
    substudies: list[tuple[float, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ridge, summary in substudies:
        for source in summary["results"]:
            row = dict(source)
            row["ridge"] = ridge
            row["ridge_over_n"] = ridge / config.effective_n
            row["map_role"] = (
                "main_and_robustness"
                if source["seed"] == config.seeds[0]
                and source["target_pn_ratio"] in config.robustness_ratios
                else ("main_map" if source["seed"] == config.seeds[0] else "robustness_replication")
            )
            rows.append(row)
    return rows


def aggregate_robustness(config: Phase7Config, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for ridge in config.ridge_lambdas:
        for ratio in config.robustness_ratios:
            matched = [
                row for row in rows if row["ridge"] == ridge and row["target_pn_ratio"] == ratio
            ]
            if len(matched) != len(config.seeds):
                continue
            for metric in ROBUSTNESS_METRICS:
                aggregates.append(
                    {
                        "ridge": ridge,
                        "target_pn_ratio": ratio,
                        "actual_pn_ratio": matched[0]["actual_pn_ratio"],
                        "feature_count": matched[0]["feature_count"],
                        "metric": metric,
                        **summarize_values([float(row[metric]) for row in matched]),
                    }
                )
    return aggregates


def _aggregate_lookup(
    aggregates: list[dict[str, Any]], ridge: float, ratio: float, metric: str
) -> dict[str, Any]:
    return next(
        row
        for row in aggregates
        if row["ridge"] == ridge and row["target_pn_ratio"] == ratio and row["metric"] == metric
    )


def compare_ridgeless_reproduction(
    config: Phase7Config,
    phase5_summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    reference_path = Path(phase5_summary["artifacts"]["seed_results"])
    if not reference_path.is_absolute():
        reference_path = Path.cwd() / reference_path
    reference = pd.read_csv(reference_path)
    current = [row for row in rows if row["ridge"] == 0]
    differences: dict[str, list[float]] = {metric: [] for metric in REPRODUCTION_METRICS}
    missing: list[dict[str, Any]] = []
    for row in current:
        matched = reference.loc[
            (reference["seed"] == row["seed"])
            & np.isclose(reference["target_pn_ratio"], row["target_pn_ratio"], atol=1e-14)
        ]
        if len(matched) != 1:
            missing.append({"seed": row["seed"], "ratio": row["target_pn_ratio"]})
            continue
        previous = matched.iloc[0]
        for metric in REPRODUCTION_METRICS:
            denominator = max(abs(float(previous[metric])), 1e-20)
            differences[metric].append(
                abs(float(row[metric]) - float(previous[metric])) / denominator
            )
    maximums = {
        metric: max(values) if values else math.inf for metric, values in differences.items()
    }
    return {
        "comparison_count": len(current) - len(missing),
        "expected_comparison_count": len(current),
        "missing": missing,
        "maximum_relative_errors": maximums,
        "tolerance": REPRODUCTION_RELATIVE_TOLERANCE,
        "passed": not missing
        and all(value <= REPRODUCTION_RELATIVE_TOLERANCE for value in maximums.values()),
    }


def _monotone_with_tolerance(values: list[float], increasing: bool) -> bool:
    for left, right in pairwise(values):
        tolerance = 1e-8 * max(abs(left), abs(right), 1.0)
        if increasing and right + tolerance < left:
            return False
        if not increasing and right > left + tolerance:
            return False
    return True


def evaluate_regularization_diagnostics(
    config: Phase7Config, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    checks: dict[str, dict[str, bool]] = {}
    for seed in config.seeds:
        ratios = config.map_ratios if seed == config.seeds[0] else config.robustness_ratios
        for ratio in ratios:
            ordered = sorted(
                (row for row in rows if row["seed"] == seed and row["target_pn_ratio"] == ratio),
                key=lambda row: row["ridge"],
            )
            key = f"seed-{seed}-pn-{ratio:g}"
            checks[key] = {
                "training_mse_non_decreasing": _monotone_with_tolerance(
                    [float(row["train_train_mse_mean"]) for row in ordered], increasing=True
                ),
                "degrees_of_freedom_non_increasing": _monotone_with_tolerance(
                    [
                        float(row["train_ridge_effective_degrees_of_freedom_mean"])
                        for row in ordered
                    ],
                    increasing=False,
                ),
                "coefficient_norm_non_increasing": _monotone_with_tolerance(
                    [
                        float(row["train_normalized_feature_coefficient_norm_mean"])
                        for row in ordered
                    ],
                    increasing=False,
                ),
                "ridge_condition_non_increasing": _monotone_with_tolerance(
                    [float(row["train_ridge_system_condition_number_mean"]) for row in ordered],
                    increasing=False,
                ),
            }
    return {
        "cell_count": len(checks),
        "checks": checks,
        "passed": bool(checks) and all(all(item.values()) for item in checks.values()),
    }


def assess_regularization_map(
    config: Phase7Config,
    rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
) -> dict[str, Any]:
    reference_seed = config.seeds[0]
    main = [row for row in rows if row["seed"] == reference_seed]
    zero_mse = float(main[0]["zero_mse"])
    lambda_summaries: list[dict[str, Any]] = []
    for ridge in config.ridge_lambdas:
        curve = sorted(
            (row for row in main if row["ridge"] == ridge),
            key=lambda row: row["actual_pn_ratio"],
        )
        near = [row for row in curve if 0.9 <= row["actual_pn_ratio"] <= 1.1]
        peak = max(near, key=lambda row: row["oos_mse"])
        best = min(curve, key=lambda row: row["oos_mse"])
        global_peak = max(curve, key=lambda row: row["oos_mse"])
        lambda_summaries.append(
            {
                "ridge": ridge,
                "near_threshold_peak_ratio": peak["actual_pn_ratio"],
                "near_threshold_peak_mse": peak["oos_mse"],
                "global_peak_ratio": global_peak["actual_pn_ratio"],
                "global_peak_mse": global_peak["oos_mse"],
                "best_ratio": best["actual_pn_ratio"],
                "best_mse": best["oos_mse"],
                "best_mse_vs_zero": best["oos_mse"] / zero_mse,
                "curve_mse_dynamic_range": global_peak["oos_mse"] / best["oos_mse"],
            }
        )

    ridgeless = next(row for row in lambda_summaries if row["ridge"] == 0)
    strong = next((row for row in lambda_summaries if row["ridge"] == STRONG_RIDGE_LAMBDA), None)
    positive_best = min((row for row in main if row["ridge"] > 0), key=lambda row: row["oos_mse"])
    robust_mse = [row for row in aggregates if row["metric"] == "oos_mse"]
    best_robust = min(robust_mse, key=lambda row: row["mean"])
    best_rows = [
        row
        for row in rows
        if row["ridge"] == best_robust["ridge"]
        and row["target_pn_ratio"] == best_robust["target_pn_ratio"]
    ]
    seed_beats_zero = sum(float(row["oos_mse"]) < float(row["zero_mse"]) for row in best_rows)
    robust_beats_zero = (
        seed_beats_zero == len(config.seeds)
        and best_robust["mean"] < zero_mse
        and best_robust["ci95_high"] < zero_mse
    )
    best_ic = _aggregate_lookup(
        aggregates,
        best_robust["ridge"],
        best_robust["target_pn_ratio"],
        "oos_information_coefficient",
    )
    best_prediction_std = _aggregate_lookup(
        aggregates,
        best_robust["ridge"],
        best_robust["target_pn_ratio"],
        "oos_prediction_standard_deviation",
    )
    best_sharpe = _aggregate_lookup(
        aggregates,
        best_robust["ridge"],
        best_robust["target_pn_ratio"],
        "trading_sharpe",
    )
    best_profit_factor = _aggregate_lookup(
        aggregates,
        best_robust["ridge"],
        best_robust["target_pn_ratio"],
        "trading_profit_factor",
    )
    best_profit = _aggregate_lookup(
        aggregates,
        best_robust["ridge"],
        best_robust["target_pn_ratio"],
        "trading_profit_total",
    )
    best_r2 = _aggregate_lookup(
        aggregates,
        best_robust["ridge"],
        best_robust["target_pn_ratio"],
        "oos_r2",
    )
    best_direction = _aggregate_lookup(
        aggregates,
        best_robust["ridge"],
        best_robust["target_pn_ratio"],
        "oos_directional_accuracy",
    )
    best_degrees_of_freedom = _aggregate_lookup(
        aggregates,
        best_robust["ridge"],
        best_robust["target_pn_ratio"],
        "train_ridge_effective_degrees_of_freedom_mean",
    )
    best_coefficient_norm = _aggregate_lookup(
        aggregates,
        best_robust["ridge"],
        best_robust["target_pn_ratio"],
        "train_normalized_feature_coefficient_norm_mean",
    )
    best_trades = _aggregate_lookup(
        aggregates,
        best_robust["ridge"],
        best_robust["target_pn_ratio"],
        "trading_total_trades",
    )
    best_turnover = _aggregate_lookup(
        aggregates,
        best_robust["ridge"],
        best_robust["target_pn_ratio"],
        "trading_turnover_multiple",
    )
    economic_seed_count = sum(
        row["trading_sharpe"] > 0.5
        and row["trading_profit_factor"] > 1
        and row["trading_profit_total"] > 0
        and row["trading_total_trades"] >= 30
        for row in best_rows
    )

    strong_suppression: list[float] = []
    if STRONG_RIDGE_LAMBDA in config.ridge_lambdas and 1.0 in config.robustness_ratios:
        for seed in config.seeds:
            baseline = next(
                row
                for row in rows
                if row["seed"] == seed and row["ridge"] == 0 and row["target_pn_ratio"] == 1.0
            )
            regularized = next(
                row
                for row in rows
                if row["seed"] == seed
                and row["ridge"] == STRONG_RIDGE_LAMBDA
                and row["target_pn_ratio"] == 1.0
            )
            strong_suppression.append(1 - regularized["oos_mse"] / baseline["oos_mse"])

    unregularized_same_ratio = _aggregate_lookup(
        aggregates, 0.0, best_robust["target_pn_ratio"], "oos_prediction_standard_deviation"
    )
    p1_mechanism = []
    if 1.0 in config.map_ratios:
        for ridge in dict.fromkeys((0.0, STRONG_RIDGE_LAMBDA, max(config.ridge_lambdas))):
            if ridge not in config.ridge_lambdas:
                continue
            row = next(
                item for item in main if item["ridge"] == ridge and item["target_pn_ratio"] == 1.0
            )
            p1_mechanism.append(
                {
                    "ridge": ridge,
                    "oos_mse": row["oos_mse"],
                    "oos_mse_vs_zero": row["oos_mse"] / zero_mse,
                    "prediction_standard_deviation": row["oos_prediction_standard_deviation"],
                    "training_mse": row["train_train_mse_mean"],
                    "effective_degrees_of_freedom": row[
                        "train_ridge_effective_degrees_of_freedom_mean"
                    ],
                    "normalized_coefficient_norm": row[
                        "train_normalized_feature_coefficient_norm_mean"
                    ],
                    "ridge_system_condition_number": row[
                        "train_ridge_system_condition_number_mean"
                    ],
                }
            )
    predictive_alpha = robust_beats_zero and best_ic["mean"] > 0 and best_ic["ci95_low"] > 0
    economic = economic_seed_count == len(config.seeds)
    return {
        "status": "development_regularization_map",
        "lambda_summaries_reference_seed": lambda_summaries,
        "ridgeless_near_threshold_peak_mse": ridgeless["near_threshold_peak_mse"],
        "strong_lambda": STRONG_RIDGE_LAMBDA,
        "strong_lambda_near_threshold_peak_mse": (
            strong["near_threshold_peak_mse"] if strong else None
        ),
        "strong_lambda_reference_peak_suppression_fraction": (
            1 - strong["near_threshold_peak_mse"] / ridgeless["near_threshold_peak_mse"]
            if strong
            else None
        ),
        "strong_lambda_seed_peak_suppression_fractions": strong_suppression,
        "strong_regularization_suppresses_interpolation_catastrophe": bool(
            strong_suppression and all(value >= 0.9 for value in strong_suppression)
        ),
        "best_positive_lambda_reference_cell": {
            "ridge": positive_best["ridge"],
            "target_pn_ratio": positive_best["target_pn_ratio"],
            "mse": positive_best["oos_mse"],
            "mse_vs_zero": positive_best["oos_mse"] / zero_mse,
        },
        "best_three_seed_robustness_cell": {
            "ridge": best_robust["ridge"],
            "target_pn_ratio": best_robust["target_pn_ratio"],
            "mean_mse": best_robust["mean"],
            "mse_ci95": [best_robust["ci95_low"], best_robust["ci95_high"]],
            "mean_mse_vs_zero": best_robust["mean"] / zero_mse,
            "seed_count_beating_zero": seed_beats_zero,
            "mean_r2": best_r2["mean"],
            "mean_ic": best_ic["mean"],
            "ic_ci95": [best_ic["ci95_low"], best_ic["ci95_high"]],
            "mean_directional_accuracy": best_direction["mean"],
            "mean_prediction_standard_deviation": best_prediction_std["mean"],
            "mean_effective_degrees_of_freedom": best_degrees_of_freedom["mean"],
            "mean_normalized_coefficient_norm": best_coefficient_norm["mean"],
            "mean_trades": best_trades["mean"],
            "mean_turnover_multiple": best_turnover["mean"],
            "mean_sharpe": best_sharpe["mean"],
            "mean_profit_factor": best_profit_factor["mean"],
            "mean_profit_total": best_profit["mean"],
        },
        "zero_baseline_mse": zero_mse,
        "best_cell_robustly_beats_zero": robust_beats_zero,
        "reference_map_cell_count_beating_zero": sum(
            float(row["oos_mse"]) < float(row["zero_mse"]) for row in main
        ),
        "p1_regularization_mechanism_reference_seed": p1_mechanism,
        "predictive_alpha_evidence": predictive_alpha,
        "economic_evidence": economic,
        "economic_gate_seed_count": economic_seed_count,
        "shrinkage_only_explanation_supported": bool(
            not robust_beats_zero and best_prediction_std["mean"] < unregularized_same_ratio["mean"]
        ),
        "multiple_testing_cell_count": len(config.map_ratios) * len(config.ridge_lambdas),
        "selection_warning": (
            "The best cell is descriptive development evidence after a predeclared grid search; "
            "it is not an unbiased estimate and was not evaluated on the final holdout."
        ),
        "seed_interval_warning": (
            "Three-seed t intervals quantify RFF projection dispersion only, not independent "
            "market-sample uncertainty."
        ),
    }


def evaluate_phase7_gate(
    config: Phase7Config,
    substudies: list[tuple[float, dict[str, Any]]],
    rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    reproduction: dict[str, Any],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    expected_substudies = len(config.ridge_lambdas) * len(config.seeds)
    expected_rows = len(config.ridge_lambdas) * (
        len(config.map_ratios) + (len(config.seeds) - 1) * len(config.robustness_ratios)
    )
    expected_aggregates = (
        len(config.ridge_lambdas) * len(config.robustness_ratios) * len(ROBUSTNESS_METRICS)
    )
    keys = {(row["seed"], row["ridge"], row["target_pn_ratio"]) for row in rows}
    observation_counts = {row["oos_observation_count"] for row in rows}
    zero_mses = [float(row["zero_mse"]) for row in rows]
    finite = all(
        all(
            math.isfinite(float(row[key]))
            for key in (
                "oos_mse",
                "oos_mae",
                "oos_r2",
                "oos_information_coefficient",
                "train_train_mse_mean",
                "train_ridge_effective_degrees_of_freedom_mean",
                "train_normalized_feature_coefficient_norm_mean",
                "trading_sharpe",
                "trading_profit_total",
            )
        )
        for row in rows
    )
    checks = {
        "all_phase4_substudies_passed": len(substudies) == expected_substudies
        and all(summary["gate"]["passed"] for _, summary in substudies),
        "complete_predeclared_map": len(rows) == expected_rows and len(keys) == expected_rows,
        "complete_three_seed_robustness_aggregate": len(aggregates) == expected_aggregates,
        "same_oos_observation_count": len(observation_counts) == 1,
        "same_zero_return_baseline": bool(zero_mses)
        and np.allclose(zero_mses, zero_mses[0], rtol=0, atol=1e-20),
        "all_metrics_finite": bool(rows) and finite,
        "ridgeless_control_reproduces_phase5": reproduction["passed"],
        "ridge_diagnostics_obey_theoretical_monotonicity": diagnostics["passed"],
        "cuda_float64_diagnostics_recorded": all(
            summary["gate"]["checks"].get("cuda_float64_diagnostics_recorded", False)
            for _, summary in substudies
        )
        and config.dtype == "float64",
        "holdout_was_not_used": config.timerange.split("-", 1)[1] <= config.holdout_start,
    }
    return {"passed": all(checks.values()), "checks": checks}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _ridge_label(value: float) -> str:
    return "0" if value == 0 else f"{value:g}"


def _write_plot(config: Phase7Config, path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return False
    main = [row for row in rows if row["seed"] == config.seeds[0]]
    if not main:
        return False
    ratios = list(config.map_ratios)
    lambdas = list(config.ridge_lambdas)
    lookup = {(row["ridge"], row["target_pn_ratio"]): row for row in main}
    zero_mse = float(main[0]["zero_mse"])

    def matrix(metric: str, transform: Any = None) -> list[list[float]]:
        values = [[float(lookup[(ridge, ratio)][metric]) for ratio in ratios] for ridge in lambdas]
        return [[transform(value) for value in row] for row in values] if transform else values

    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "log10 OOS MSE / zero baseline",
            "log10 training MSE",
            "Ridge effective degrees of freedom",
            "Net trading Sharpe",
        ),
    )
    panels = (
        (
            matrix("oos_mse", lambda value: math.log10(value / zero_mse)),
            1,
            1,
            "RdBu_r",
        ),
        (
            matrix("train_train_mse_mean", lambda value: math.log10(max(value, 1e-30))),
            1,
            2,
            "Viridis",
        ),
        (matrix("train_ridge_effective_degrees_of_freedom_mean"), 2, 1, "Cividis"),
        (matrix("trading_sharpe"), 2, 2, "RdBu"),
    )
    for values, panel_row, panel_column, colorscale in panels:
        figure.add_trace(
            go.Heatmap(
                z=values,
                x=[f"{ratio:g}" for ratio in ratios],
                y=[_ridge_label(value) for value in lambdas],
                colorscale=colorscale,
                hovertemplate="P/N=%{x}<br>lambda=%{y}<br>value=%{z:.5g}<extra></extra>",
                showscale=False,
            ),
            row=panel_row,
            col=panel_column,
        )
    figure.update_xaxes(title_text="P/N")
    figure.update_yaxes(title_text="Ridge lambda")
    figure.update_layout(
        title="Phase 7 - Frozen RFF regularization map (reference seed)",
        template="plotly_white",
        height=900,
        width=1400,
    )
    figure.write_html(path, include_plotlyjs="cdn")
    return True


def run_phase7(config: Phase7Config) -> dict[str, Any]:
    config.validate()
    phase5_summary = load_compatible_phase5_summary(config)
    config.output_directory.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    checkpoint_path = config.output_directory / "checkpoint.json"
    substudies: list[tuple[float, dict[str, Any]]] = []
    completed: list[dict[str, Any]] = []
    for ridge in config.ridge_lambdas:
        for seed in config.seeds:
            summary = _run_substudy(config, seed, ridge)
            substudies.append((ridge, summary))
            completed.append(
                {
                    "seed": seed,
                    "ridge": ridge,
                    "passed": summary["gate"]["passed"],
                    "summary": summary["artifacts"]["summary"],
                }
            )
            checkpoint_path.write_text(
                json.dumps(_json_safe({"run_id": run_id, "completed": completed}), indent=2),
                encoding="utf-8",
            )

    rows = flatten_substudies(config, substudies)
    aggregates = aggregate_robustness(config, rows)
    reproduction = compare_ridgeless_reproduction(config, phase5_summary, rows)
    diagnostics = evaluate_regularization_diagnostics(config, rows)
    assessment = assess_regularization_map(config, rows, aggregates)
    gate = evaluate_phase7_gate(config, substudies, rows, aggregates, reproduction, diagnostics)

    rows_path = config.output_directory / "regularization_map.csv"
    aggregates_path = config.output_directory / "robustness_aggregates.csv"
    summary_path = config.output_directory / "summary.json"
    plot_path = config.output_directory / "regularization_map.html"
    _write_csv(rows_path, rows)
    _write_csv(aggregates_path, aggregates)
    plot_written = _write_plot(config, plot_path, rows)
    config_payload = asdict(config)
    for key in ("data_directory", "output_directory", "phase5_summary"):
        config_payload[key] = str(config_payload[key])
    summary = {
        "phase": 7,
        "objective": "Map explicit Ridge regularization across P/N without touching the holdout",
        "scope": "predeclared development map plus initial three-seed critical-point robustness",
        "run_id": run_id,
        "design": {
            "ridge_objective": "centered kernel system (K + lambda I) alpha = y",
            "intercept_regularized": False,
            "main_map_seed": config.seeds[0],
            "main_map_cells": len(config.map_ratios) * len(config.ridge_lambdas),
            "robustness_seeds": list(config.seeds),
            "robustness_cells": (
                len(config.seeds) * len(config.robustness_ratios) * len(config.ridge_lambdas)
            ),
            "grid_selected_before_results": True,
            "holdout_used_for_selection": False,
        },
        "compute": {
            "case_count": len(rows),
            "rolling_model_fit_count": len(rows) * 13,
            "summed_freqtrade_wall_seconds": sum(float(row["wall_seconds"]) for row in rows),
            "summed_cuda_training_seconds": sum(
                float(row["train_training_seconds_total"]) for row in rows
            ),
            "maximum_peak_vram_mib": max(float(row["train_peak_vram_mib"]) for row in rows),
        },
        "config": config_payload,
        "ridgeless_reproduction": reproduction,
        "regularization_diagnostics": diagnostics,
        "robustness_aggregates": aggregates,
        "regularization_assessment": assessment,
        "gate": gate,
        "artifacts": {
            "summary": str(summary_path),
            "regularization_map": str(rows_path),
            "robustness_aggregates": str(aggregates_path),
            "interactive_plot": str(plot_path) if plot_written else None,
            "checkpoint": str(checkpoint_path),
        },
    }
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    return summary
