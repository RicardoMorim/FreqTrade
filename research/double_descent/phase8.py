"""Phase 8: training-window sensitivity of financial double descent."""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from research.double_descent.phase4 import (
    PHASE3_EFFECTIVE_N_BY_TRAIN_PERIOD,
    Phase4Config,
    feature_count_for_ratio,
    run_phase4,
)
from research.double_descent.phase5 import PHASE5_SEEDS, summarize_values


PHASE8_TRAINING_WINDOWS = (30, 60, 90, 180, 365)
PHASE8_MAIN_RATIOS = (0.1, 0.5, 0.9, 0.98, 1.0, 1.02, 1.1, 2.0, 5.0)
PHASE8_ROBUSTNESS_RATIOS = (0.1, 1.0, 1.02, 5.0)
PHASE8_SEEDS = PHASE5_SEEDS[:3]
REPRODUCTION_RELATIVE_TOLERANCE = 1e-8

ROBUSTNESS_METRICS = (
    "oos_mse",
    "oos_mae",
    "oos_r2",
    "oos_information_coefficient",
    "oos_directional_accuracy",
    "oos_prediction_standard_deviation",
    "train_train_mse_mean",
    "train_interpolated_window_fraction",
    "train_effective_rank_mean",
    "train_condition_number_maximum",
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
class Phase8Config:
    data_directory: Path
    output_directory: Path = Path("user_data/research_results/double_descent/phase8")
    phase3_summary: Path = Path("user_data/research_results/double_descent/phase3/summary.json")
    phase5_summary: Path = Path("user_data/research_results/double_descent/phase5/summary.json")
    phase5_seed_results: Path = Path(
        "user_data/research_results/double_descent/phase5/seed_results.csv"
    )
    python_executable: str = sys.executable
    cuda_python_executable: str = ""
    pair: str = "BTC/USDT:USDT"
    timeframe: str = "1h"
    timerange: str = "20250101-20260101"
    holdout_start: str = "20260101"
    training_windows: tuple[int, ...] = PHASE8_TRAINING_WINDOWS
    backtest_period_days: int = 30
    main_ratios: tuple[float, ...] = PHASE8_MAIN_RATIOS
    robustness_ratios: tuple[float, ...] = PHASE8_ROBUSTNESS_RATIOS
    seeds: tuple[int, ...] = PHASE8_SEEDS
    gamma: float = 0.2
    ridge: float = 0.0
    rcond: float = 1e-12
    dtype: str = "float64"
    chunk_size: int = 4_096
    fee: float = 0.001
    minimum_training_windows: int = 10
    minimum_seed_count: int = 3
    subprocess_timeout_seconds: int = 7_200
    resume: bool = True
    smoke_test: bool = False

    def validate(self) -> None:
        self._validate_references()
        self._validate_grid()
        self._validate_phase4_runtime()

    def _validate_references(self) -> None:
        for path, label in (
            (self.phase3_summary, "Phase 3 summary"),
            (self.phase5_summary, "Phase 5 summary"),
            (self.phase5_seed_results, "Phase 5 seed results"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label} does not exist: {path}")

    def _validate_grid(self) -> None:
        if not self.training_windows:
            raise ValueError("at least one training window is required")
        if tuple(sorted(set(self.training_windows))) != self.training_windows:
            raise ValueError("training windows must be sorted and unique")
        if any(
            window not in PHASE3_EFFECTIVE_N_BY_TRAIN_PERIOD for window in self.training_windows
        ):
            raise ValueError("every training window must have an exact Phase 3 N measurement")
        if not self.main_ratios or tuple(sorted(set(self.main_ratios))) != self.main_ratios:
            raise ValueError("main P/N ratios must be sorted and unique")
        if not set(self.robustness_ratios).issubset(self.main_ratios):
            raise ValueError("robustness ratios must be a subset of the main grid")
        if len(set(self.seeds)) != len(self.seeds) or any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be unique non-negative integers")
        minimum = 1 if self.smoke_test else 3
        if self.minimum_seed_count < minimum or len(self.seeds) < self.minimum_seed_count:
            raise ValueError(f"Phase 8 requires at least {minimum} seed(s) for this run")
        if self.ridge != 0:
            raise ValueError("Phase 8 changes only training-window length; Ridge must remain zero")
        self._validate_confirmatory_grid()

    def _validate_confirmatory_grid(self) -> None:
        if not self.smoke_test:
            if self.training_windows != PHASE8_TRAINING_WINDOWS:
                raise ValueError("the confirmatory Phase 8 run requires all five frozen windows")
            if self.main_ratios != PHASE8_MAIN_RATIOS:
                raise ValueError(
                    "the confirmatory Phase 8 run requires the frozen compact P/N grid"
                )
            if self.robustness_ratios != PHASE8_ROBUSTNESS_RATIOS:
                raise ValueError("the confirmatory Phase 8 run requires the frozen robustness grid")
            if len(self.seeds) < 3:
                raise ValueError("endpoint robustness requires at least three seeds")

    def _validate_phase4_runtime(self) -> None:
        _phase4_config(
            self,
            window=self.training_windows[0],
            seed=self.seeds[0],
            ratios=self.main_ratios,
            output_directory=self.output_directory / "validation",
        ).validate()


def _phase4_config(
    config: Phase8Config,
    window: int,
    seed: int,
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
        train_period_days=window,
        backtest_period_days=config.backtest_period_days,
        effective_n=PHASE3_EFFECTIVE_N_BY_TRAIN_PERIOD[window],
        ratios=ratios,
        seed=seed,
        gamma=config.gamma,
        ridge=config.ridge,
        rcond=config.rcond,
        dtype=config.dtype,
        chunk_size=config.chunk_size,
        fee=config.fee,
        minimum_training_windows=config.minimum_training_windows,
        subprocess_timeout_seconds=config.subprocess_timeout_seconds,
        resume=config.resume,
        allow_phase3_training_window_variation=True,
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


def load_compatible_phase3_summary(config: Phase8Config) -> dict[str, Any]:
    summary = json.loads(config.phase3_summary.read_text(encoding="utf-8"))
    if summary.get("phase") != 3 or not summary.get("gate", {}).get("passed"):
        raise ValueError("Phase 3 reference did not pass its integrity gate")
    reference = summary.get("config", {})
    expected = {
        "pair": config.pair,
        "timeframe": config.timeframe,
        "timerange": config.timerange,
        "backtest_period_days": config.backtest_period_days,
    }
    mismatches = {
        key: {"expected": value, "actual": reference.get(key)}
        for key, value in expected.items()
        if reference.get(key) != value
    }
    observed_n = {
        int(row["train_period_days"]): int(row["effective_n_median"])
        for row in summary.get("aggregate", [])
    }
    expected_n = {
        window: PHASE3_EFFECTIVE_N_BY_TRAIN_PERIOD[window] for window in config.training_windows
    }
    if mismatches or any(observed_n.get(key) != value for key, value in expected_n.items()):
        raise ValueError(
            f"Phase 3 reference is incompatible: controls={mismatches}, N={observed_n}"
        )
    return summary


def load_compatible_phase5_summary(config: Phase8Config) -> dict[str, Any]:
    summary = json.loads(config.phase5_summary.read_text(encoding="utf-8"))
    if summary.get("phase") != 5 or not summary.get("gate", {}).get("passed"):
        raise ValueError("Phase 5 reference did not pass its integrity gate")
    reference = summary.get("config", {})
    expected = {
        "pair": config.pair,
        "timeframe": config.timeframe,
        "timerange": config.timerange,
        "holdout_start": config.holdout_start,
        "train_period_days": 90,
        "backtest_period_days": config.backtest_period_days,
        "effective_n": PHASE3_EFFECTIVE_N_BY_TRAIN_PERIOD[90],
        "gamma": config.gamma,
        "ridge": config.ridge,
        "rcond": config.rcond,
        "dtype": config.dtype,
        "fee": config.fee,
    }
    mismatches = {
        key: {"expected": value, "actual": reference.get(key)}
        for key, value in expected.items()
        if reference.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Phase 5 reference is incompatible: {mismatches}")
    if config.seeds[0] not in reference.get("seeds", []):
        raise ValueError("the Phase 8 reference seed is absent from Phase 5")
    if not set(config.main_ratios).issubset(reference.get("ratios", [])):
        raise ValueError("the Phase 8 P/N grid is not covered by Phase 5")
    return summary


def _task_grid(config: Phase8Config) -> list[tuple[int, int, tuple[float, ...]]]:
    tasks = [(window, config.seeds[0], config.main_ratios) for window in config.training_windows]
    endpoint_windows = {config.training_windows[0], config.training_windows[-1]}
    for window in sorted(endpoint_windows):
        for seed in config.seeds[1:]:
            tasks.append((window, seed, config.robustness_ratios))
    return tasks


def _run_substudy(
    config: Phase8Config,
    window: int,
    seed: int,
    ratios: tuple[float, ...],
) -> dict[str, Any]:
    directory = config.output_directory / "runs" / f"window-{window}" / f"seed-{seed}"
    print(
        f"PHASE8 START window={window}d N={PHASE3_EFFECTIVE_N_BY_TRAIN_PERIOD[window]} "
        f"seed={seed} points={len(ratios)}",
        flush=True,
    )
    summary = run_phase4(_phase4_config(config, window, seed, ratios, directory))
    print(
        f"PHASE8 DONE window={window}d seed={seed} passed={summary['gate']['passed']}",
        flush=True,
    )
    return summary


def flatten_substudies(
    config: Phase8Config,
    substudies: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    endpoint_windows = {config.training_windows[0], config.training_windows[-1]}
    for substudy in substudies:
        window = substudy["window"]
        seed = substudy["seed"]
        for source in substudy["summary"]["results"]:
            row = dict(source)
            row["train_period_days"] = window
            row["effective_n"] = PHASE3_EFFECTIVE_N_BY_TRAIN_PERIOD[window]
            row["ridge"] = config.ridge
            if seed == config.seeds[0]:
                row["map_role"] = (
                    "main_and_endpoint_robustness"
                    if window in endpoint_windows
                    and source["target_pn_ratio"] in config.robustness_ratios
                    else "main_window_map"
                )
            else:
                row["map_role"] = "endpoint_robustness_replication"
            rows.append(row)
    return rows


def aggregate_endpoint_robustness(
    config: Phase8Config, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for window in (config.training_windows[0], config.training_windows[-1]):
        for ratio in config.robustness_ratios:
            matched = [
                row
                for row in rows
                if row["train_period_days"] == window
                and row["target_pn_ratio"] == ratio
                and row["seed"] in config.seeds
            ]
            if len(matched) != len(config.seeds):
                continue
            for metric in ROBUSTNESS_METRICS:
                aggregates.append(
                    {
                        "train_period_days": window,
                        "effective_n": PHASE3_EFFECTIVE_N_BY_TRAIN_PERIOD[window],
                        "target_pn_ratio": ratio,
                        "actual_pn_ratio": matched[0]["actual_pn_ratio"],
                        "feature_count": matched[0]["feature_count"],
                        "metric": metric,
                        **summarize_values([float(row[metric]) for row in matched]),
                    }
                )
    return aggregates


def assess_flat_curve(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row["actual_pn_ratio"])
    near = [row for row in ordered if 0.90 <= row["actual_pn_ratio"] <= 1.10]
    under = [row for row in ordered if row["actual_pn_ratio"] < 0.90]
    if not near or not under or not ordered:
        return {"status": "insufficient_grid"}
    peak = max(near, key=lambda row: row["oos_mse"])
    best_under = min(under, key=lambda row: row["oos_mse"])
    extreme = ordered[-1]
    global_peak = max(ordered, key=lambda row: row["oos_mse"])
    interpolation_onset = next(
        (
            row["actual_pn_ratio"]
            for row in ordered
            if row["train_interpolated_window_fraction"] == 1.0
        ),
        None,
    )
    deterioration = peak["oos_mse"] - best_under["oos_mse"]
    recovery = peak["oos_mse"] - extreme["oos_mse"]
    recovery_fraction = recovery / deterioration if deterioration > 0 else math.nan
    return {
        "status": "complete",
        "global_peak_ratio": global_peak["actual_pn_ratio"],
        "global_peak_is_near_threshold": global_peak is peak,
        "near_threshold_peak_ratio": peak["actual_pn_ratio"],
        "near_threshold_peak_mse": peak["oos_mse"],
        "best_underparameterized_ratio": best_under["actual_pn_ratio"],
        "best_underparameterized_mse": best_under["oos_mse"],
        "largest_ratio": extreme["actual_pn_ratio"],
        "largest_ratio_mse": extreme["oos_mse"],
        "largest_vs_zero_mse_ratio": extreme["oos_mse"] / extreme["zero_mse"],
        "largest_vs_best_underparameterized_mse_ratio": (
            extreme["oos_mse"] / best_under["oos_mse"]
        ),
        "recovery_fraction": recovery_fraction,
        "strict_interpolation_onset_ratio_within_curve_grid": interpolation_onset,
        "double_descent_pattern_detected": bool(
            global_peak is peak and deterioration > 0 and recovery_fraction >= 0.5
        ),
        "largest_model_beats_zero_mse": extreme["oos_mse"] < extreme["zero_mse"],
        "second_descent_beats_best_underparameterized": (
            extreme["oos_mse"] < best_under["oos_mse"]
        ),
    }


def assess_training_window_effect(
    config: Phase8Config, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    reference_seed = config.seeds[0]
    main_assessments = []
    for window in config.training_windows:
        matched = [
            row
            for row in rows
            if row["train_period_days"] == window and row["seed"] == reference_seed
        ]
        main_assessments.append(
            {
                "train_period_days": window,
                "effective_n": PHASE3_EFFECTIVE_N_BY_TRAIN_PERIOD[window],
                **assess_flat_curve(matched),
            }
        )

    endpoint_assessments = []
    for window in (config.training_windows[0], config.training_windows[-1]):
        for seed in config.seeds:
            matched = [
                row
                for row in rows
                if row["train_period_days"] == window
                and row["seed"] == seed
                and row["target_pn_ratio"] in config.robustness_ratios
            ]
            endpoint_assessments.append(
                {"train_period_days": window, "seed": seed, **assess_flat_curve(matched)}
            )

    finite_main = [
        row
        for row in main_assessments
        if math.isfinite(float(row.get("recovery_fraction", math.nan)))
    ]
    distinct_recoveries = {float(row["recovery_fraction"]) for row in finite_main}
    if len(finite_main) >= 2 and len(distinct_recoveries) >= 2:
        correlation = spearmanr(
            [row["train_period_days"] for row in finite_main],
            [row["recovery_fraction"] for row in finite_main],
        )
        recovery_spearman = float(correlation.statistic)
        recovery_spearman_pvalue = float(correlation.pvalue)
    else:
        recovery_spearman = math.nan
        recovery_spearman_pvalue = math.nan

    endpoint_shape_counts = {
        str(window): sum(
            bool(row.get("double_descent_pattern_detected"))
            for row in endpoint_assessments
            if row["train_period_days"] == window
        )
        for window in (config.training_windows[0], config.training_windows[-1])
    }
    main_shape_count = sum(
        bool(row.get("double_descent_pattern_detected")) for row in main_assessments
    )
    useful_main_count = sum(
        bool(row.get("largest_model_beats_zero_mse"))
        and bool(row.get("second_descent_beats_best_underparameterized"))
        for row in main_assessments
    )
    longest = config.training_windows[-1]
    longest_robust_count = endpoint_shape_counts[str(longest)]
    majority = math.ceil(len(config.seeds) / 2)
    return {
        "reference_seed_window_curves": main_assessments,
        "endpoint_seed_curves": endpoint_assessments,
        "reference_seed_shape_window_count": main_shape_count,
        "reference_seed_window_count": len(config.training_windows),
        "endpoint_shape_seed_counts": endpoint_shape_counts,
        "endpoint_seed_count": len(config.seeds),
        "recovery_fraction_vs_window_spearman": recovery_spearman,
        "recovery_fraction_vs_window_spearman_pvalue_unadjusted": recovery_spearman_pvalue,
        "double_descent_persists_at_longest_window": longest_robust_count >= majority,
        "small_sample_only_explanation_supported": (
            endpoint_shape_counts[str(config.training_windows[0])] >= majority
            and longest_robust_count < majority
        ),
        "useful_benign_overfitting_window_count": useful_main_count,
        "predictive_alpha_evidence": useful_main_count == len(config.training_windows),
        "multiple_testing_note": (
            "The five-window trend p-value is descriptive and unadjusted; it is not used as a "
            "discovery claim. The windows and decision rules were frozen before execution."
        ),
        "identification_limit": (
            "Longer rolling histories increase N but also change the mix and recency of market "
            "regimes, so this phase is a window-sensitivity test rather than a pure causal N test."
        ),
    }


def compare_phase5_reproduction(config: Phase8Config, rows: list[dict[str, Any]]) -> dict[str, Any]:
    reference = pd.read_csv(config.phase5_seed_results)
    reference = reference.loc[reference["seed"] == config.seeds[0]].copy()
    current = [
        row for row in rows if row["train_period_days"] == 90 and row["seed"] == config.seeds[0]
    ]
    comparisons = []
    for row in current:
        matched = reference.loc[
            np.isclose(
                reference["target_pn_ratio"].to_numpy(dtype=float),
                float(row["target_pn_ratio"]),
                rtol=0,
                atol=1e-12,
            )
        ]
        if len(matched) != 1:
            comparisons.append(
                {"target_pn_ratio": row["target_pn_ratio"], "passed": False, "error": "missing"}
            )
            continue
        source = matched.iloc[0]
        metric_results = {}
        for metric in REPRODUCTION_METRICS:
            expected = float(source[metric])
            actual = float(row[metric])
            scale = max(abs(expected), abs(actual), 1e-30)
            relative_error = abs(actual - expected) / scale
            metric_results[metric] = {
                "expected": expected,
                "actual": actual,
                "relative_error": relative_error,
                "passed": relative_error <= REPRODUCTION_RELATIVE_TOLERANCE,
            }
        comparisons.append(
            {
                "target_pn_ratio": row["target_pn_ratio"],
                "passed": all(item["passed"] for item in metric_results.values()),
                "metrics": metric_results,
            }
        )
    return {
        "relative_tolerance": REPRODUCTION_RELATIVE_TOLERANCE,
        "comparison_count": len(comparisons),
        "passed": len(comparisons) == len(config.main_ratios)
        and all(item["passed"] for item in comparisons),
        "comparisons": comparisons,
    }


def evaluate_phase8_gate(
    config: Phase8Config,
    phase3_summary: dict[str, Any],
    phase5_summary: dict[str, Any],
    substudies: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    reproduction: dict[str, Any],
) -> dict[str, Any]:
    endpoint_count = len({config.training_windows[0], config.training_windows[-1]})
    expected_cases = len(config.training_windows) * len(config.main_ratios) + endpoint_count * (
        len(config.seeds) - 1
    ) * len(config.robustness_ratios)
    expected_aggregates = endpoint_count * len(config.robustness_ratios) * len(ROBUSTNESS_METRICS)
    main_cells = {
        (row["train_period_days"], row["target_pn_ratio"])
        for row in rows
        if row["seed"] == config.seeds[0]
    }
    expected_main_cells = {
        (window, ratio) for window in config.training_windows for ratio in config.main_ratios
    }
    endpoint_cells = {
        (row["train_period_days"], row["seed"], row["target_pn_ratio"])
        for row in rows
        if row["train_period_days"] in {config.training_windows[0], config.training_windows[-1]}
        and row["target_pn_ratio"] in config.robustness_ratios
    }
    expected_endpoint_cells = {
        (window, seed, ratio)
        for window in {config.training_windows[0], config.training_windows[-1]}
        for seed in config.seeds
        for ratio in config.robustness_ratios
    }
    finite = all(
        math.isfinite(float(row[key]))
        for row in rows
        for key in ("oos_mse", "oos_mae", "oos_r2", "trading_sharpe")
    )
    checks = {
        "phase3_integrity_gate_passed": bool(phase3_summary.get("gate", {}).get("passed")),
        "phase5_reference_gate_passed": bool(phase5_summary.get("gate", {}).get("passed")),
        "every_rolling_freqai_substudy_passed": bool(substudies)
        and all(item["summary"]["gate"]["passed"] for item in substudies),
        "complete_predeclared_case_count": len(rows) == expected_cases,
        "complete_reference_seed_window_map": main_cells == expected_main_cells,
        "complete_three_seed_endpoint_map": endpoint_cells == expected_endpoint_cells,
        "complete_endpoint_aggregate": len(aggregates) == expected_aggregates,
        "feature_counts_match_each_measured_n": all(
            row["feature_count"]
            == feature_count_for_ratio(row["target_pn_ratio"], row["effective_n"])
            for row in rows
        ),
        "all_metrics_are_finite": bool(rows) and finite,
        "same_oos_observation_count_for_every_case": len(
            {int(row["oos_observation_count"]) for row in rows}
        )
        == 1,
        "same_zero_baseline_for_every_case": bool(rows)
        and np.allclose(
            [float(row["zero_mse"]) for row in rows],
            float(rows[0]["zero_mse"]),
            rtol=0,
            atol=1e-20,
        ),
        "ridgeless_90_day_reference_reproduced": reproduction["passed"],
        "cuda_float64_used": config.dtype == "float64"
        and all(float(row["train_peak_vram_mib"]) > 0 for row in rows),
        "holdout_was_not_used": config.timerange.split("-", 1)[1] <= config.holdout_start,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "expected_case_count": expected_cases,
        "observed_case_count": len(rows),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(config: Phase8Config, path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return False
    main = [row for row in rows if row["seed"] == config.seeds[0]]
    if not main:
        return False
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "OOS MSE / zero baseline",
            "Training MSE",
            "Maximum condition number",
            "Net trading Sharpe",
        ),
    )
    panels = (
        ("oos_mse", 1, 1, lambda row: row["oos_mse"] / row["zero_mse"]),
        ("train_train_mse_mean", 1, 2, lambda row: row["train_train_mse_mean"]),
        ("train_condition_number_maximum", 2, 1, lambda row: row["train_condition_number_maximum"]),
        ("trading_sharpe", 2, 2, lambda row: row["trading_sharpe"]),
    )
    for window in config.training_windows:
        matched = sorted(
            (row for row in main if row["train_period_days"] == window),
            key=lambda row: row["actual_pn_ratio"],
        )
        for _, panel_row, panel_column, transform in panels:
            figure.add_trace(
                go.Scatter(
                    x=[row["actual_pn_ratio"] for row in matched],
                    y=[transform(row) for row in matched],
                    mode="lines+markers",
                    name=f"{window}d / N={PHASE3_EFFECTIVE_N_BY_TRAIN_PERIOD[window]}",
                    legendgroup=str(window),
                    showlegend=panel_row == 1 and panel_column == 1,
                ),
                row=panel_row,
                col=panel_column,
            )
    figure.update_xaxes(type="log", title_text="P/N")
    figure.update_yaxes(type="log", row=1, col=1)
    figure.update_yaxes(type="log", row=1, col=2)
    figure.update_yaxes(type="log", row=2, col=1)
    figure.update_layout(
        title="Phase 8 - Training-window sensitivity at fixed P/N",
        template="plotly_white",
        height=900,
        width=1400,
    )
    figure.write_html(path, include_plotlyjs="cdn")
    return True


def run_phase8(config: Phase8Config) -> dict[str, Any]:
    config.validate()
    phase3_summary = load_compatible_phase3_summary(config)
    phase5_summary = load_compatible_phase5_summary(config)
    config.output_directory.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    checkpoint_path = config.output_directory / "checkpoint.json"
    substudies = []
    completed = []
    for window, seed, ratios in _task_grid(config):
        summary = _run_substudy(config, window, seed, ratios)
        substudies.append({"window": window, "seed": seed, "summary": summary})
        completed.append(
            {
                "train_period_days": window,
                "effective_n": PHASE3_EFFECTIVE_N_BY_TRAIN_PERIOD[window],
                "seed": seed,
                "ratios": list(ratios),
                "passed": summary["gate"]["passed"],
                "summary": summary["artifacts"]["summary"],
            }
        )
        checkpoint_path.write_text(
            json.dumps(_json_safe({"run_id": run_id, "completed": completed}), indent=2),
            encoding="utf-8",
        )

    rows = flatten_substudies(config, substudies)
    aggregates = aggregate_endpoint_robustness(config, rows)
    reproduction = compare_phase5_reproduction(config, rows)
    assessment = assess_training_window_effect(config, rows)
    gate = evaluate_phase8_gate(
        config,
        phase3_summary,
        phase5_summary,
        substudies,
        rows,
        aggregates,
        reproduction,
    )
    rows_path = config.output_directory / "window_map.csv"
    aggregates_path = config.output_directory / "endpoint_robustness.csv"
    summary_path = config.output_directory / "summary.json"
    plot_path = config.output_directory / "training_window_sensitivity.html"
    _write_csv(rows_path, rows)
    _write_csv(aggregates_path, aggregates)
    plot_written = _write_plot(config, plot_path, rows)
    config_payload = asdict(config)
    for key in (
        "data_directory",
        "output_directory",
        "phase3_summary",
        "phase5_summary",
        "phase5_seed_results",
    ):
        config_payload[key] = str(config_payload[key])
    summary = {
        "phase": 8,
        "objective": "Test whether financial double descent persists as rolling training N grows",
        "scope": "predeclared development-only window sensitivity; final holdout remains untouched",
        "run_id": run_id,
        "design": {
            "variable_changed": "rolling training-window length and its measured effective N",
            "frozen_controls": [
                "BTC perpetual 1h data",
                "2025 chronological OOS evaluation",
                "25 causal market-state inputs",
                "nested RFF seed/gamma",
                "ridgeless float64 CUDA solver",
                "sign strategy and 0.1% fee per side",
            ],
            "main_map_seed": config.seeds[0],
            "endpoint_robustness_windows": [
                config.training_windows[0],
                config.training_windows[-1],
            ],
            "endpoint_robustness_seeds": list(config.seeds),
            "maximum_ratio_rationale": (
                "P/N=5 is materially overparameterized and bounds the 365-day dual computation; "
                "Phase 6 separately established the much larger-P kernel limit."
            ),
            "grid_selected_before_results": True,
            "holdout_used_for_selection": False,
        },
        "compute": {
            "case_count": len(rows),
            "rolling_model_fit_count": sum(int(row["train_training_window_count"]) for row in rows),
            "summed_freqtrade_wall_seconds": sum(float(row["wall_seconds"]) for row in rows),
            "summed_cuda_training_seconds": sum(
                float(row["train_training_seconds_total"]) for row in rows
            ),
            "maximum_peak_vram_mib": max(float(row["train_peak_vram_mib"]) for row in rows),
        },
        "config": config_payload,
        "phase5_reproduction": reproduction,
        "endpoint_robustness": aggregates,
        "training_window_assessment": assessment,
        "gate": gate,
        "artifacts": {
            "summary": str(summary_path),
            "window_map": str(rows_path),
            "endpoint_robustness": str(aggregates_path),
            "interactive_plot": str(plot_path) if plot_written else None,
            "checkpoint": str(checkpoint_path),
        },
    }
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    return summary
