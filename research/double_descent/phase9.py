"""Phase 9: pre-2025 gamma calibration for the financial RFF experiment."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from research.double_descent.phase4 import Phase4Config, feature_count_for_ratio, run_phase4
from research.double_descent.phase5 import PHASE5_SEEDS, summarize_values
from research.double_descent.phase8 import assess_flat_curve


PHASE9_GAMMAS = (0.05, 0.1, 0.2, 0.5)
PHASE9_CALIBRATION_TIMERANGE = "20240701-20250101"
PHASE9_MAP_RATIOS = (0.1, 0.5, 0.9, 0.98, 1.0, 1.02, 1.1, 2.0, 5.0, 10.0, 50.0)
PHASE9_ROBUSTNESS_RATIOS = (0.1, 1.0, 1.02, 5.0, 50.0)
PHASE9_SELECTION_RATIOS = (0.1, 5.0, 50.0)
PHASE9_SEEDS = PHASE5_SEEDS[:3]
BASELINE_GAMMA = 0.2
MINIMUM_RELATIVE_SCORE_IMPROVEMENT = 0.05
MINIMUM_MATCHED_WIN_FRACTION = 2 / 3

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
    "train_effective_rank_mean",
    "train_condition_number_maximum",
    "trading_total_trades",
    "trading_profit_total",
    "trading_sharpe",
    "trading_profit_factor",
    "trading_max_drawdown_account",
)


@dataclass(frozen=True)
class Phase9Config:
    data_directory: Path
    output_directory: Path = Path("user_data/research_results/double_descent/phase9")
    phase5_summary: Path = Path("user_data/research_results/double_descent/phase5/summary.json")
    python_executable: str = sys.executable
    cuda_python_executable: str = ""
    pair: str = "BTC/USDT:USDT"
    timeframe: str = "1h"
    calibration_timerange: str = PHASE9_CALIBRATION_TIMERANGE
    main_experiment_start: str = "20250101"
    holdout_start: str = "20260101"
    train_period_days: int = 90
    backtest_period_days: int = 30
    effective_n: int = 2_159
    map_ratios: tuple[float, ...] = PHASE9_MAP_RATIOS
    robustness_ratios: tuple[float, ...] = PHASE9_ROBUSTNESS_RATIOS
    selection_ratios: tuple[float, ...] = PHASE9_SELECTION_RATIOS
    gammas: tuple[float, ...] = PHASE9_GAMMAS
    seeds: tuple[int, ...] = PHASE9_SEEDS
    baseline_gamma: float = BASELINE_GAMMA
    minimum_relative_score_improvement: float = MINIMUM_RELATIVE_SCORE_IMPROVEMENT
    minimum_matched_win_fraction: float = MINIMUM_MATCHED_WIN_FRACTION
    ridge: float = 0.0
    rcond: float = 1e-12
    dtype: str = "float64"
    chunk_size: int = 4_096
    fee: float = 0.001
    minimum_training_windows: int = 6
    minimum_seed_count: int = 3
    subprocess_timeout_seconds: int = 1_800
    resume: bool = True
    smoke_test: bool = False

    def validate(self) -> None:
        self._validate_reference()
        self._validate_grid()
        self._validate_dates()
        self._validate_runtime()

    def _validate_reference(self) -> None:
        if not self.phase5_summary.is_file():
            raise FileNotFoundError(f"Phase 5 summary does not exist: {self.phase5_summary}")

    def _validate_grid(self) -> None:
        for values, label in (
            (self.map_ratios, "map ratios"),
            (self.robustness_ratios, "robustness ratios"),
            (self.selection_ratios, "selection ratios"),
            (self.gammas, "gammas"),
        ):
            if (
                not values
                or tuple(sorted(set(values))) != values
                or any(value <= 0 for value in values)
            ):
                raise ValueError(f"{label} must be positive, sorted, and unique")
        if not set(self.robustness_ratios).issubset(self.map_ratios):
            raise ValueError("robustness ratios must be a subset of the main map")
        if not set(self.selection_ratios).issubset(self.robustness_ratios):
            raise ValueError("selection ratios must be a subset of the robustness grid")
        if self.baseline_gamma not in self.gammas:
            raise ValueError("the current baseline gamma must be included")
        if len(set(self.seeds)) != len(self.seeds) or any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be unique non-negative integers")
        minimum = 1 if self.smoke_test else 3
        if self.minimum_seed_count < minimum or len(self.seeds) < self.minimum_seed_count:
            raise ValueError(f"Phase 9 requires at least {minimum} seed(s) for this run")
        if not 0 <= self.minimum_relative_score_improvement < 1:
            raise ValueError("minimum score improvement must be in [0, 1)")
        if not 0 < self.minimum_matched_win_fraction <= 1:
            raise ValueError("minimum matched win fraction must be in (0, 1]")
        if self.ridge != 0:
            raise ValueError("Phase 9 changes only gamma; Ridge must remain zero")
        self._validate_confirmatory_grid()

    def _validate_confirmatory_grid(self) -> None:
        if self.smoke_test:
            return
        expected = (
            (self.map_ratios, PHASE9_MAP_RATIOS, "main map"),
            (self.robustness_ratios, PHASE9_ROBUSTNESS_RATIOS, "robustness grid"),
            (self.selection_ratios, PHASE9_SELECTION_RATIOS, "selection grid"),
            (self.gammas, PHASE9_GAMMAS, "gamma grid"),
        )
        for actual, frozen, label in expected:
            if actual != frozen:
                raise ValueError(f"the confirmatory Phase 9 run requires the frozen {label}")

    def _validate_dates(self) -> None:
        _, calibration_end = _parse_timerange(self.calibration_timerange)
        experiment_start = _parse_date(self.main_experiment_start)
        holdout_start = _parse_date(self.holdout_start)
        if calibration_end > experiment_start or experiment_start > holdout_start:
            raise ValueError("gamma calibration must finish before the main experiment and holdout")

    def _validate_runtime(self) -> None:
        _phase4_config(
            self,
            gamma=self.gammas[0],
            seed=self.seeds[0],
            ratios=self.map_ratios,
            output_directory=self.output_directory / "validation",
        ).validate()


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%d").replace(tzinfo=UTC)


def _parse_timerange(timerange: str) -> tuple[datetime, datetime]:
    start, end = timerange.split("-", 1)
    return _parse_date(start), _parse_date(end)


def _gamma_slug(gamma: float) -> str:
    return f"{gamma:g}".replace(".", "p")


def _phase4_config(
    config: Phase9Config,
    gamma: float,
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
        timerange=config.calibration_timerange,
        holdout_start=config.holdout_start,
        train_period_days=config.train_period_days,
        backtest_period_days=config.backtest_period_days,
        effective_n=config.effective_n,
        ratios=ratios,
        seed=seed,
        gamma=gamma,
        ridge=config.ridge,
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


def load_compatible_phase5_summary(config: Phase9Config) -> dict[str, Any]:
    summary = json.loads(config.phase5_summary.read_text(encoding="utf-8"))
    if summary.get("phase") != 5 or not summary.get("gate", {}).get("passed"):
        raise ValueError("Phase 5 reference did not pass its integrity gate")
    reference = summary.get("config", {})
    expected = {
        "pair": config.pair,
        "timeframe": config.timeframe,
        "holdout_start": config.holdout_start,
        "train_period_days": config.train_period_days,
        "backtest_period_days": config.backtest_period_days,
        "effective_n": config.effective_n,
        "gamma": config.baseline_gamma,
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
    if not set(config.seeds).issubset(reference.get("seeds", [])):
        raise ValueError("Phase 9 seeds are absent from the audited Phase 5 derivation")
    if not set(config.map_ratios).issubset(reference.get("ratios", [])):
        raise ValueError("Phase 9 ratios are not covered by the audited Phase 5 grid")
    return summary


def _task_grid(config: Phase9Config) -> list[tuple[float, int, tuple[float, ...]]]:
    tasks = []
    for gamma in config.gammas:
        tasks.append((gamma, config.seeds[0], config.map_ratios))
        tasks.extend((gamma, seed, config.robustness_ratios) for seed in config.seeds[1:])
    return tasks


def _run_substudy(
    config: Phase9Config,
    gamma: float,
    seed: int,
    ratios: tuple[float, ...],
) -> dict[str, Any]:
    directory = config.output_directory / "runs" / f"gamma-{_gamma_slug(gamma)}" / f"seed-{seed}"
    print(
        f"PHASE9 START gamma={gamma:g} seed={seed} points={len(ratios)}",
        flush=True,
    )
    summary = run_phase4(_phase4_config(config, gamma, seed, ratios, directory))
    print(
        f"PHASE9 DONE gamma={gamma:g} seed={seed} passed={summary['gate']['passed']}",
        flush=True,
    )
    return summary


def flatten_substudies(
    config: Phase9Config, substudies: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for substudy in substudies:
        gamma = substudy["gamma"]
        seed = substudy["seed"]
        for source in substudy["summary"]["results"]:
            row = dict(source)
            row["gamma"] = gamma
            row["ridge"] = config.ridge
            row["map_role"] = (
                "main_and_robustness"
                if seed == config.seeds[0] and source["target_pn_ratio"] in config.robustness_ratios
                else ("main_gamma_map" if seed == config.seeds[0] else "robustness_replication")
            )
            rows.append(row)
    return rows


def aggregate_robustness(config: Phase9Config, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for gamma in config.gammas:
        for ratio in config.robustness_ratios:
            matched = [
                row
                for row in rows
                if row["gamma"] == gamma
                and row["target_pn_ratio"] == ratio
                and row["seed"] in config.seeds
            ]
            if len(matched) != len(config.seeds):
                continue
            for metric in ROBUSTNESS_METRICS:
                aggregates.append(
                    {
                        "gamma": gamma,
                        "target_pn_ratio": ratio,
                        "actual_pn_ratio": matched[0]["actual_pn_ratio"],
                        "feature_count": matched[0]["feature_count"],
                        "metric": metric,
                        **summarize_values([float(row[metric]) for row in matched]),
                    }
                )
    return aggregates


def assess_gamma_curves(config: Phase9Config, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assessments = []
    for gamma in config.gammas:
        for seed in config.seeds:
            matched = [
                row
                for row in rows
                if row["gamma"] == gamma
                and row["seed"] == seed
                and row["target_pn_ratio"] in config.robustness_ratios
            ]
            assessments.append({"gamma": gamma, "seed": seed, **assess_flat_curve(matched)})
    return assessments


def _geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric mean requires positive finite values")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def select_gamma(
    config: Phase9Config,
    rows: list[dict[str, Any]],
    curve_assessments: list[dict[str, Any]],
) -> dict[str, Any]:
    selection_rows = [
        row
        for row in rows
        if row["seed"] in config.seeds and row["target_pn_ratio"] in config.selection_ratios
    ]
    baseline_lookup = {
        (row["seed"], row["target_pn_ratio"]): row
        for row in selection_rows
        if row["gamma"] == config.baseline_gamma
    }
    expected_cells = len(config.seeds) * len(config.selection_ratios)
    minimum_shape_seeds = math.ceil(len(config.seeds) * config.minimum_matched_win_fraction)
    scores = []
    for gamma in config.gammas:
        matched = [row for row in selection_rows if row["gamma"] == gamma]
        normalized_mse = [float(row["oos_mse"]) / float(row["zero_mse"]) for row in matched]
        wins = sum(
            float(row["oos_mse"])
            < float(baseline_lookup[(row["seed"], row["target_pn_ratio"])]["oos_mse"])
            for row in matched
            if (row["seed"], row["target_pn_ratio"]) in baseline_lookup
        )
        shape_seed_count = sum(
            bool(item.get("double_descent_pattern_detected"))
            for item in curve_assessments
            if item["gamma"] == gamma
        )
        scores.append(
            {
                "gamma": gamma,
                "cell_count": len(matched),
                "geometric_mean_mse_over_zero": _geometric_mean(normalized_mse),
                "arithmetic_mean_mse_over_zero": statistics.fmean(normalized_mse),
                "median_mse_over_zero": statistics.median(normalized_mse),
                "matched_wins_vs_baseline": wins,
                "matched_win_fraction_vs_baseline": wins / expected_cells,
                "beats_zero_cell_count": sum(value < 1 for value in normalized_mse),
                "shape_seed_count": shape_seed_count,
                "eligible": len(matched) == expected_cells
                and shape_seed_count >= minimum_shape_seeds,
            }
        )

    eligible = [row for row in scores if row["eligible"]]
    if not eligible:
        raise ValueError("no gamma passed the predeclared numerical/shape eligibility rule")
    raw_best = min(eligible, key=lambda row: row["geometric_mean_mse_over_zero"])
    baseline = next(row for row in scores if row["gamma"] == config.baseline_gamma)
    required_wins = math.ceil(expected_cells * config.minimum_matched_win_fraction)
    relative_improvement = 1 - (
        raw_best["geometric_mean_mse_over_zero"] / baseline["geometric_mean_mse_over_zero"]
    )
    promote_challenger = bool(
        raw_best["gamma"] != config.baseline_gamma
        and relative_improvement >= config.minimum_relative_score_improvement
        and raw_best["matched_wins_vs_baseline"] >= required_wins
    )
    selected_gamma = raw_best["gamma"] if promote_challenger else config.baseline_gamma
    selected = next(row for row in scores if row["gamma"] == selected_gamma)
    return {
        "selection_period": config.calibration_timerange,
        "selection_metric": (
            "geometric mean OOS MSE / zero at P/N 0.1, 5, and 50 across three seeds"
        ),
        "trading_metrics_used_for_selection": False,
        "candidate_scores": sorted(scores, key=lambda row: row["geometric_mean_mse_over_zero"]),
        "raw_best_gamma": raw_best["gamma"],
        "baseline_gamma": config.baseline_gamma,
        "relative_score_improvement_vs_baseline": relative_improvement,
        "required_relative_improvement": config.minimum_relative_score_improvement,
        "required_matched_wins": required_wins,
        "promote_challenger": promote_challenger,
        "selected_gamma": selected_gamma,
        "selected_score": selected["geometric_mean_mse_over_zero"],
        "selected_gamma_beats_zero_cell_count": selected["beats_zero_cell_count"],
        "selected_gamma_selection_cell_count": selected["cell_count"],
        "predictive_alpha_evidence": selected["beats_zero_cell_count"] == selected["cell_count"],
        "decision_reason": (
            "The challenger passed both the 5% score-improvement and two-thirds matched-win rules."
            if promote_challenger
            else "No challenger cleared both promotion hurdles; retain the existing gamma=0.2."
        ),
        "freeze_scope": (
            "Gamma is frozen for subsequent development experiments. The 2026 holdout remains "
            "sealed and is not part of this selection."
        ),
    }


def evaluate_phase9_gate(
    config: Phase9Config,
    phase5_summary: dict[str, Any],
    substudies: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    selection: dict[str, Any],
) -> dict[str, Any]:
    expected_cases = len(config.gammas) * (
        len(config.map_ratios) + (len(config.seeds) - 1) * len(config.robustness_ratios)
    )
    expected_aggregates = (
        len(config.gammas) * len(config.robustness_ratios) * len(ROBUSTNESS_METRICS)
    )
    main_cells = {
        (row["gamma"], row["target_pn_ratio"]) for row in rows if row["seed"] == config.seeds[0]
    }
    expected_main_cells = {(gamma, ratio) for gamma in config.gammas for ratio in config.map_ratios}
    robust_cells = {
        (row["gamma"], row["seed"], row["target_pn_ratio"])
        for row in rows
        if row["target_pn_ratio"] in config.robustness_ratios
    }
    expected_robust_cells = {
        (gamma, seed, ratio)
        for gamma in config.gammas
        for seed in config.seeds
        for ratio in config.robustness_ratios
    }
    _, calibration_end = _parse_timerange(config.calibration_timerange)
    checks = {
        "phase5_reference_gate_passed": bool(phase5_summary.get("gate", {}).get("passed")),
        "every_rolling_freqai_substudy_passed": bool(substudies)
        and all(item["summary"]["gate"]["passed"] for item in substudies),
        "complete_predeclared_case_count": len(rows) == expected_cases,
        "complete_reference_seed_gamma_map": main_cells == expected_main_cells,
        "complete_three_seed_robustness_map": robust_cells == expected_robust_cells,
        "complete_robustness_aggregate": len(aggregates) == expected_aggregates,
        "feature_counts_match_measured_n": all(
            row["feature_count"]
            == feature_count_for_ratio(row["target_pn_ratio"], config.effective_n)
            for row in rows
        ),
        "all_metrics_are_finite": bool(rows)
        and all(
            math.isfinite(float(row[key]))
            for row in rows
            for key in ("oos_mse", "oos_mae", "oos_r2", "trading_sharpe")
        ),
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
        "selection_is_prediction_only": selection["trading_metrics_used_for_selection"] is False,
        "selection_decision_is_complete": selection["selected_gamma"] in config.gammas,
        "calibration_precedes_main_experiment": calibration_end
        <= _parse_date(config.main_experiment_start),
        "holdout_was_not_used": calibration_end <= _parse_date(config.holdout_start),
        "cuda_float64_used": config.dtype == "float64"
        and all(float(row["train_peak_vram_mib"]) > 0 for row in rows),
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


def _write_plot(config: Phase9Config, path: Path, rows: list[dict[str, Any]]) -> bool:
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
            "Effective rank",
            "Net trading Sharpe",
        ),
    )
    panels = (
        (1, 1, lambda row: row["oos_mse"] / row["zero_mse"]),
        (1, 2, lambda row: row["train_train_mse_mean"]),
        (2, 1, lambda row: row["train_effective_rank_mean"]),
        (2, 2, lambda row: row["trading_sharpe"]),
    )
    for gamma in config.gammas:
        matched = sorted(
            (row for row in main if row["gamma"] == gamma),
            key=lambda row: row["actual_pn_ratio"],
        )
        for panel_row, panel_column, transform in panels:
            figure.add_trace(
                go.Scatter(
                    x=[row["actual_pn_ratio"] for row in matched],
                    y=[transform(row) for row in matched],
                    mode="lines+markers",
                    name=f"gamma={gamma:g}",
                    legendgroup=str(gamma),
                    showlegend=panel_row == 1 and panel_column == 1,
                ),
                row=panel_row,
                col=panel_column,
            )
    figure.update_xaxes(type="log", title_text="P/N")
    figure.update_yaxes(type="log", row=1, col=1)
    figure.update_yaxes(type="log", row=1, col=2)
    figure.update_layout(
        title="Phase 9 - Pre-2025 gamma calibration",
        template="plotly_white",
        height=900,
        width=1400,
    )
    figure.write_html(path, include_plotlyjs="cdn")
    return True


def run_phase9(config: Phase9Config) -> dict[str, Any]:
    config.validate()
    phase5_summary = load_compatible_phase5_summary(config)
    config.output_directory.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    checkpoint_path = config.output_directory / "checkpoint.json"
    substudies = []
    completed = []
    for gamma, seed, ratios in _task_grid(config):
        result = _run_substudy(config, gamma, seed, ratios)
        substudies.append({"gamma": gamma, "seed": seed, "summary": result})
        completed.append(
            {
                "gamma": gamma,
                "seed": seed,
                "ratios": list(ratios),
                "passed": result["gate"]["passed"],
                "summary": result["artifacts"]["summary"],
            }
        )
        checkpoint_path.write_text(
            json.dumps(_json_safe({"run_id": run_id, "completed": completed}), indent=2),
            encoding="utf-8",
        )

    rows = flatten_substudies(config, substudies)
    aggregates = aggregate_robustness(config, rows)
    curve_assessments = assess_gamma_curves(config, rows)
    selection = select_gamma(config, rows, curve_assessments)
    gate = evaluate_phase9_gate(
        config,
        phase5_summary,
        substudies,
        rows,
        aggregates,
        selection,
    )
    rows_path = config.output_directory / "gamma_map.csv"
    aggregates_path = config.output_directory / "robustness_aggregates.csv"
    summary_path = config.output_directory / "summary.json"
    plot_path = config.output_directory / "gamma_calibration.html"
    frozen_path = config.output_directory / "frozen_gamma.json"
    _write_csv(rows_path, rows)
    _write_csv(aggregates_path, aggregates)
    plot_written = _write_plot(config, plot_path, rows)
    frozen_path.write_text(
        json.dumps(
            {
                "gamma": selection["selected_gamma"],
                "selected_on": config.calibration_timerange,
                "holdout_used": False,
                "run_id": run_id,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    config_payload = asdict(config)
    for key in ("data_directory", "output_directory", "phase5_summary"):
        config_payload[key] = str(config_payload[key])
    summary = {
        "phase": 9,
        "objective": "Select and freeze a reasonable RFF/RBF gamma before final evaluation",
        "scope": "pre-2025 calibration only; 2025 experiment and 2026 holdout excluded",
        "run_id": run_id,
        "design": {
            "variable_changed": "RFF/RBF gamma only",
            "paired_random_draws": (
                "same seeds and Gaussian draws; changing gamma rescales the projections"
            ),
            "main_map_seed": config.seeds[0],
            "robustness_seeds": list(config.seeds),
            "selection_ratios": list(config.selection_ratios),
            "minimum_relative_improvement": config.minimum_relative_score_improvement,
            "minimum_matched_win_fraction": config.minimum_matched_win_fraction,
            "grid_selected_before_results": True,
            "trading_used_for_selection": False,
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
        "curve_assessments": curve_assessments,
        "robustness_aggregates": aggregates,
        "gamma_selection": selection,
        "gate": gate,
        "artifacts": {
            "summary": str(summary_path),
            "gamma_map": str(rows_path),
            "robustness_aggregates": str(aggregates_path),
            "frozen_gamma": str(frozen_path),
            "interactive_plot": str(plot_path) if plot_written else None,
            "checkpoint": str(checkpoint_path),
        },
    }
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    return summary
