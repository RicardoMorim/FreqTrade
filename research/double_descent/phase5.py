"""Phase 5: multi-seed robustness analysis for the financial P/N sweep."""

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
from scipy.stats import t as student_t

from research.double_descent.phase3 import PN_RATIOS
from research.double_descent.phase4 import Phase4Config, run_phase4


PHASE4_REFERENCE_SEED = 20260810
PHASE5_ADDITIONAL_SEEDS = (1898170439, 3432960257, 2841638297, 2713191886)
PHASE5_SEEDS = (PHASE4_REFERENCE_SEED, *PHASE5_ADDITIONAL_SEEDS)

AGGREGATE_METRICS = (
    "oos_mse",
    "oos_rmse",
    "oos_mae",
    "oos_r2",
    "oos_r2_vs_zero",
    "oos_information_coefficient",
    "oos_spearman_information_coefficient",
    "oos_directional_accuracy",
    "train_train_mse_mean",
    "train_train_mse_maximum",
    "train_condition_number_median",
    "train_condition_number_maximum",
    "train_effective_rank_mean",
    "train_interpolated_window_fraction",
    "train_training_seconds_total",
    "train_peak_vram_mib",
    "stability_window_ic_mean",
    "stability_window_ic_standard_deviation",
    "stability_window_ic_positive_fraction",
    "stability_window_mse_mean",
    "stability_window_mse_standard_deviation",
    "trading_total_trades",
    "trading_sharpe",
    "trading_sortino",
    "trading_profit_factor",
    "trading_profit_total",
    "trading_max_drawdown_account",
    "trading_win_rate",
    "trading_turnover_multiple",
)


@dataclass(frozen=True)
class Phase5Config:
    data_directory: Path
    output_directory: Path = Path("user_data/research_results/double_descent/phase5")
    phase4_reference_summary: Path | None = Path(
        "user_data/research_results/double_descent/phase4/summary.json"
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
    ratios: tuple[float, ...] = PN_RATIOS
    seeds: tuple[int, ...] = PHASE5_SEEDS
    gamma: float = 0.2
    ridge: float = 0.0
    rcond: float = 1e-12
    dtype: str = "float64"
    chunk_size: int = 4_096
    fee: float = 0.001
    minimum_training_windows: int = 10
    minimum_seed_count: int = 5
    subprocess_timeout_seconds: int = 1_800
    resume: bool = True

    def validate(self) -> None:
        if self.minimum_seed_count < 3:
            raise ValueError("Phase 5 requires at least three seeds for an initial robustness test")
        if len(self.seeds) < self.minimum_seed_count:
            raise ValueError(
                f"Phase 5 requires at least {self.minimum_seed_count} independent seeds"
            )
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Phase 5 seeds must be unique")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("Phase 5 seeds must be non-negative")
        phase4_config(self, self.seeds[0], self.output_directory / "validation").validate()


def phase4_config(config: Phase5Config, seed: int, output_directory: Path) -> Phase4Config:
    """Create a Phase 4 replicate while changing only seed and artifact location."""
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
        ratios=config.ratios,
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


def _phase4_summary_matches(summary: dict[str, Any], config: Phase5Config, seed: int) -> bool:
    expected = phase4_config(config, seed, Path("unused"))
    observed = summary.get("config", {})
    fields = (
        "pair",
        "timeframe",
        "timerange",
        "holdout_start",
        "train_period_days",
        "backtest_period_days",
        "effective_n",
        "gamma",
        "ridge",
        "rcond",
        "dtype",
        "chunk_size",
        "fee",
        "minimum_training_windows",
        "interpolation_mse_tolerance",
    )
    if summary.get("phase") != 4 or not summary.get("gate", {}).get("passed"):
        return False
    if observed.get("seed") != seed or tuple(observed.get("ratios", ())) != config.ratios:
        return False
    try:
        same_data_directory = (
            Path(observed["data_directory"]).resolve() == config.data_directory.resolve()
        )
    except (KeyError, TypeError):
        return False
    if not same_data_directory:
        return False
    return all(observed.get(field) == getattr(expected, field) for field in fields)


def _load_compatible_summary(
    path: Path | None, config: Phase5Config, seed: int
) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return summary if _phase4_summary_matches(summary, config, seed) else None


def _run_or_recover_seed(config: Phase5Config, seed: int) -> tuple[dict[str, Any], str]:
    if seed == PHASE4_REFERENCE_SEED:
        reference = _load_compatible_summary(config.phase4_reference_summary, config, seed)
        if reference is not None:
            print(f"PHASE5 REFERENCE seed={seed}", flush=True)
            return reference, "phase4_reference"

    seed_directory = config.output_directory / "seeds" / f"seed-{seed}"
    existing = _load_compatible_summary(seed_directory / "summary.json", config, seed)
    if config.resume and existing is not None:
        print(f"PHASE5 RECOVERED seed={seed}", flush=True)
        return existing, "phase5_checkpoint"

    print(f"PHASE5 START seed={seed}", flush=True)
    summary = run_phase4(phase4_config(config, seed, seed_directory))
    print(f"PHASE5 DONE seed={seed} passed={summary['gate']['passed']}", flush=True)
    return summary, "phase5_run"


def summarize_values(values: list[float], confidence: float = 0.95) -> dict[str, float | int]:
    """Summarize independent seed results with a two-sided Student-t interval."""
    finite = [float(value) for value in values]
    if not finite or any(not math.isfinite(value) for value in finite):
        raise ValueError("cannot aggregate an empty or non-finite sample")
    mean = statistics.fmean(finite)
    median = statistics.median(finite)
    standard_deviation = statistics.stdev(finite) if len(finite) > 1 else 0.0
    if len(finite) > 1:
        critical = float(student_t.ppf((1.0 + confidence) / 2.0, len(finite) - 1))
        half_width = critical * standard_deviation / math.sqrt(len(finite))
    else:
        half_width = 0.0
    return {
        "seed_count": len(finite),
        "mean": mean,
        "median": median,
        "standard_deviation": standard_deviation,
        "minimum": min(finite),
        "maximum": max(finite),
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def aggregate_seed_summaries(seed_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate every predeclared metric at matched P/N values across seeds."""
    by_ratio: dict[float, list[dict[str, Any]]] = {}
    for summary in seed_summaries:
        for row in summary["results"]:
            by_ratio.setdefault(float(row["target_pn_ratio"]), []).append(row)

    aggregates: list[dict[str, Any]] = []
    for ratio in sorted(by_ratio):
        rows = by_ratio[ratio]
        feature_counts = {int(row["feature_count"]) for row in rows}
        if len(feature_counts) != 1:
            raise ValueError(f"feature count differs across seeds at P/N={ratio}")
        feature_count = next(iter(feature_counts))
        for metric in AGGREGATE_METRICS:
            values = [float(row[metric]) for row in rows]
            aggregates.append(
                {
                    "target_pn_ratio": ratio,
                    "actual_pn_ratio": float(rows[0]["actual_pn_ratio"]),
                    "feature_count": feature_count,
                    "metric": metric,
                    **summarize_values(values),
                }
            )
    return aggregates


def _aggregate_lookup(
    aggregates: list[dict[str, Any]], ratio: float, metric: str
) -> dict[str, Any]:
    return next(
        row for row in aggregates if row["target_pn_ratio"] == ratio and row["metric"] == metric
    )


def assess_multi_seed(
    seed_summaries: list[dict[str, Any]], aggregates: list[dict[str, Any]]
) -> dict[str, Any]:
    curves = [summary["curve_assessment"] for summary in seed_summaries]
    usable = [curve for curve in curves if curve.get("status") != "insufficient_grid"]
    if not usable:
        return {"status": "insufficient_grid", "seed_count": len(seed_summaries)}

    seed_count = len(usable)
    required_robust_count = math.ceil(0.8 * seed_count)
    shape_count = sum(bool(curve.get("double_descent_pattern_detected")) for curve in usable)
    peak_count = sum(bool(curve.get("global_peak_is_near_threshold")) for curve in usable)
    beats_zero_count = sum(bool(curve.get("largest_model_beats_zero_mse")) for curve in usable)
    beats_under_count = sum(
        bool(curve.get("second_descent_beats_best_underparameterized")) for curve in usable
    )
    recovery = [float(curve["recovery_fraction"]) for curve in usable]
    largest_ratio = max(float(row["target_pn_ratio"]) for row in seed_summaries[0]["results"])
    extreme_r2 = _aggregate_lookup(aggregates, largest_ratio, "oos_r2")
    extreme_ic = _aggregate_lookup(aggregates, largest_ratio, "oos_information_coefficient")
    extreme_sharpe = _aggregate_lookup(aggregates, largest_ratio, "trading_sharpe")
    extreme_pf = _aggregate_lookup(aggregates, largest_ratio, "trading_profit_factor")
    extreme_return = _aggregate_lookup(aggregates, largest_ratio, "trading_profit_total")
    robust_shape = shape_count >= required_robust_count and peak_count >= required_robust_count
    benign = (
        robust_shape
        and beats_zero_count >= required_robust_count
        and beats_under_count >= required_robust_count
    )
    under_ratios = [
        float(row["target_pn_ratio"])
        for row in seed_summaries[0]["results"]
        if float(row["actual_pn_ratio"]) < 0.9
    ]
    best_under = min(
        (_aggregate_lookup(aggregates, ratio, "oos_mse") for ratio in under_ratios),
        key=lambda row: row["mean"],
    )
    extreme_rows = [
        next(row for row in summary["results"] if float(row["target_pn_ratio"]) == largest_ratio)
        for summary in seed_summaries
    ]
    any_model_beats_zero_count = sum(
        min(float(row["oos_mse"]) for row in summary["results"])
        < float(summary["results"][0]["zero_mse"])
        for summary in seed_summaries
    )
    economic_seed_count = sum(
        float(row["trading_sharpe"]) > 0.5
        and float(row["trading_profit_factor"]) > 1.0
        and float(row["trading_profit_total"]) > 0
        for row in extreme_rows
    )
    prediction_alpha = extreme_r2["mean"] > 0 and extreme_ic["ci95_low"] > 0
    economic = economic_seed_count >= required_robust_count and (
        extreme_sharpe["mean"] > 0.5 and extreme_pf["mean"] > 1.0 and extreme_return["mean"] > 0
    )
    return {
        "status": "initial_five_seed_robustness",
        "seed_count": seed_count,
        "required_robust_count": required_robust_count,
        "double_descent_pattern_seed_count": shape_count,
        "global_peak_near_threshold_seed_count": peak_count,
        "largest_model_beats_zero_seed_count": beats_zero_count,
        "any_model_beats_zero_seed_count": any_model_beats_zero_count,
        "largest_model_beats_best_underparameterized_seed_count": beats_under_count,
        "largest_model_economic_gate_seed_count": economic_seed_count,
        "peak_ratios": [curve.get("global_peak_ratio") for curve in usable],
        "interpolation_onset_ratios": [curve.get("interpolation_onset_ratio") for curve in usable],
        "recovery_fraction": summarize_values(recovery),
        "robust_double_descent_shape": robust_shape,
        "benign_overfitting_evidence": benign,
        "predictive_alpha_evidence_at_largest_ratio": prediction_alpha,
        "economic_evidence_at_largest_ratio": economic,
        "largest_ratio": largest_ratio,
        "largest_ratio_mean_r2": extreme_r2["mean"],
        "largest_ratio_mean_ic": extreme_ic["mean"],
        "largest_ratio_ic_ci95": [extreme_ic["ci95_low"], extreme_ic["ci95_high"]],
        "largest_ratio_mean_mse": _aggregate_lookup(aggregates, largest_ratio, "oos_mse")["mean"],
        "largest_ratio_vs_zero_mean_mse": (
            _aggregate_lookup(aggregates, largest_ratio, "oos_mse")["mean"]
            / float(seed_summaries[0]["results"][0]["zero_mse"])
        ),
        "best_underparameterized_mean_ratio": best_under["target_pn_ratio"],
        "best_underparameterized_mean_mse": best_under["mean"],
        "largest_ratio_vs_best_underparameterized_mean_mse": (
            _aggregate_lookup(aggregates, largest_ratio, "oos_mse")["mean"] / best_under["mean"]
        ),
        "largest_ratio_mean_sharpe": extreme_sharpe["mean"],
        "interpretation": (
            "A repeatable interpolation spike and second descent is not benign overfitting unless "
            "the extreme models also generalize better than simple predictive baselines."
        ),
    }


def evaluate_phase5_gate(
    config: Phase5Config,
    seed_summaries: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
) -> dict[str, Any]:
    results = [row for summary in seed_summaries for row in summary.get("results", [])]
    expected_result_count = len(config.seeds) * len(config.ratios)
    observed_seeds = {int(row["seed"]) for row in results}
    zero_mses = {float(row["zero_mse"]) for row in results}
    observation_counts = {int(row["oos_observation_count"]) for row in results}
    finite_aggregates = all(
        math.isfinite(float(row[key]))
        for row in aggregates
        for key in ("mean", "median", "standard_deviation", "ci95_low", "ci95_high")
    )
    checks = {
        "predeclared_unique_seed_count_completed": len(seed_summaries) == len(config.seeds)
        and len(config.seeds) >= config.minimum_seed_count
        and observed_seeds == set(config.seeds),
        "every_phase4_integrity_gate_passed": all(
            summary.get("gate", {}).get("passed") for summary in seed_summaries
        ),
        "complete_matched_grid_for_every_seed": len(results) == expected_result_count
        and all(row.get("success") for row in results),
        "same_zero_baseline_across_seeds": len(zero_mses) == 1,
        "same_oos_observation_count_across_seeds": len(observation_counts) == 1,
        "all_aggregate_statistics_are_finite": bool(aggregates) and finite_aggregates,
        "all_aggregate_statistics_use_every_seed": all(
            row["seed_count"] == len(config.seeds) for row in aggregates
        ),
        "aggregate_grid_is_complete": len(aggregates)
        == len(config.ratios) * len(AGGREGATE_METRICS),
        "holdout_was_not_used": config.timerange.split("-", 1)[1] <= config.holdout_start,
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
    seed_summaries: list[dict[str, Any]],
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
        subplot_titles=("OOS MSE", "Information coefficient", "Training MSE", "Net Sharpe"),
    )
    panels = (
        ("oos_mse", 1, 1),
        ("oos_information_coefficient", 1, 2),
        ("train_train_mse_mean", 2, 1),
        ("trading_sharpe", 2, 2),
    )
    for summary in seed_summaries:
        ordered = sorted(summary["results"], key=lambda row: row["actual_pn_ratio"])
        for metric, panel_row, panel_column in panels:
            figure.add_trace(
                go.Scatter(
                    x=[row["actual_pn_ratio"] for row in ordered],
                    y=[row[metric] for row in ordered],
                    mode="lines",
                    line={"width": 1},
                    opacity=0.25,
                    name=f"seed {ordered[0]['seed']}",
                    legendgroup=f"seed-{ordered[0]['seed']}",
                    showlegend=metric == "oos_mse",
                ),
                row=panel_row,
                col=panel_column,
            )
    for metric, panel_row, panel_column in panels:
        rows = sorted(
            (row for row in aggregates if row["metric"] == metric),
            key=lambda row: row["actual_pn_ratio"],
        )
        x = [row["actual_pn_ratio"] for row in rows]
        upper = [row["ci95_high"] for row in rows]
        lower = [row["ci95_low"] for row in rows]
        figure.add_trace(
            go.Scatter(
                x=x + list(reversed(x)),
                y=upper + list(reversed(lower)),
                fill="toself",
                fillcolor="rgba(31,119,180,0.15)",
                line={"color": "rgba(255,255,255,0)"},
                hoverinfo="skip",
                name="95% t interval",
                legendgroup="aggregate",
                showlegend=metric == "oos_mse",
            ),
            row=panel_row,
            col=panel_column,
        )
        figure.add_trace(
            go.Scatter(
                x=x,
                y=[row["mean"] for row in rows],
                mode="lines+markers",
                line={"width": 3, "color": "#1f77b4"},
                name="five-seed mean",
                legendgroup="aggregate",
                showlegend=metric == "oos_mse",
            ),
            row=panel_row,
            col=panel_column,
        )
    figure.update_xaxes(type="log", title_text="P/N")
    figure.update_yaxes(type="log", title_text="MSE", row=1, col=1)
    figure.update_yaxes(type="log", title_text="MSE", row=2, col=1)
    figure.update_layout(
        title="Phase 5 - Financial RFF P/N sweep across predeclared seeds",
        template="plotly_white",
        height=800,
        width=1200,
    )
    figure.write_html(path, include_plotlyjs="cdn")
    return True


def run_phase5(config: Phase5Config) -> dict[str, Any]:
    config.validate()
    config.output_directory.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    seed_summaries: list[dict[str, Any]] = []
    sources: dict[str, str] = {}
    checkpoint_path = config.output_directory / "checkpoint.json"
    for seed in config.seeds:
        summary, source = _run_or_recover_seed(config, seed)
        seed_summaries.append(summary)
        sources[str(seed)] = source
        checkpoint_path.write_text(
            json.dumps(
                _json_safe(
                    {
                        "completed_seeds": [item["config"]["seed"] for item in seed_summaries],
                        "sources": sources,
                    }
                ),
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

    aggregates = aggregate_seed_summaries(seed_summaries)
    assessment = assess_multi_seed(seed_summaries, aggregates)
    gate = evaluate_phase5_gate(config, seed_summaries, aggregates)
    seed_results = [row for summary in seed_summaries for row in summary["results"]]
    seed_results_path = config.output_directory / "seed_results.csv"
    aggregates_path = config.output_directory / "aggregate_metrics.csv"
    summary_path = config.output_directory / "summary.json"
    plot_path = config.output_directory / "multi_seed_double_descent.html"
    _write_csv(seed_results_path, seed_results)
    _write_csv(aggregates_path, aggregates)
    plot_written = _write_plot(plot_path, seed_summaries, aggregates)

    config_payload = asdict(config)
    for key in ("data_directory", "output_directory", "phase4_reference_summary"):
        if config_payload[key] is not None:
            config_payload[key] = str(config_payload[key])
    summary = {
        "phase": 5,
        "objective": "Test whether the Phase 4 P/N curve is robust to RFF projection seed",
        "scope": "initial five-seed development robustness; final study requires 10-20 seeds",
        "run_id": run_id,
        "seed_derivation": {
            "reference_seed": PHASE4_REFERENCE_SEED,
            "method": "four deterministic children from numpy SeedSequence(20260810)",
            "seeds": list(config.seeds),
            "selection_used_results": False,
        },
        "config": config_payload,
        "seed_sources": sources,
        "seed_curve_assessments": {
            str(summary["config"]["seed"]): summary["curve_assessment"]
            for summary in seed_summaries
        },
        "aggregate_metrics": aggregates,
        "multi_seed_assessment": assessment,
        "gate": gate,
        "artifacts": {
            "summary": str(summary_path),
            "seed_results": str(seed_results_path),
            "aggregate_metrics": str(aggregates_path),
            "interactive_plot": str(plot_path) if plot_written else None,
            "checkpoint": str(checkpoint_path),
        },
    }
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    return summary
