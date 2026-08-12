"""Phase 13: causal market-regime decomposition of frozen OOS predictions."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from freqtrade.configuration import TimeRange
from freqtrade.data.btanalysis import load_backtest_data
from freqtrade.data.history import load_pair_history
from freqtrade.enums import CandleType
from research.double_descent.metrics import prediction_metrics
from research.double_descent.phase4 import _parse_timerange, _read_predictions
from research.double_descent.phase8 import assess_flat_curve
from research.double_descent.phase10 import (
    PHASE10_ROBUSTNESS_RATIOS,
    PHASE10_SEEDS,
    _json_safe,
    _write_csv,
)
from research.double_descent.phase12 import PHASE12_PREDICTION_BASELINES


TREND_LOOKBACK_HOURS = 30 * 24
VOLATILITY_REFERENCE_HOURS = 60 * 24
TREND_SCORE_THRESHOLD = 0.5
HAC_LAG_HOURS = 24
FDR_ALPHA = 0.05
MINIMUM_MARGINAL_OBSERVATIONS = 500
MINIMUM_JOINT_OBSERVATIONS = 250
MINIMUM_INTERPRETABLE_TRADES = 30
PHASE10_MARKET_RFF_CASE_COUNT = 21
PHASE12_CASE_COUNT = 8
PREDICTION_SOURCE_COUNT = PHASE10_MARKET_RFF_CASE_COUNT + len(PHASE12_PREDICTION_BASELINES)
ECONOMIC_SOURCE_COUNT = PHASE10_MARKET_RFF_CASE_COUNT + PHASE12_CASE_COUNT

TREND_REGIMES = ("bear", "sideways", "bull")
VOLATILITY_REGIMES = ("low_volatility", "high_volatility")
MARGINAL_REGIMES = tuple(("trend", value) for value in TREND_REGIMES) + tuple(
    ("volatility", value) for value in VOLATILITY_REGIMES
)
JOINT_REGIMES = tuple(
    ("joint", f"{trend}_{volatility}")
    for trend in TREND_REGIMES
    for volatility in VOLATILITY_REGIMES
)
REGIME_CELLS = MARGINAL_REGIMES + JOINT_REGIMES


@dataclass(frozen=True)
class Phase13Config:
    data_directory: Path
    output_directory: Path = Path("user_data/research_results/double_descent/phase13")
    phase10_summary: Path = Path("user_data/research_results/double_descent/phase10/summary.json")
    phase10_map: Path = Path(
        "user_data/research_results/double_descent/phase10/representation_map.csv"
    )
    phase12_summary: Path = Path("user_data/research_results/double_descent/phase12/summary.json")
    python_executable: str = sys.executable
    pair: str = "BTC/USDT:USDT"
    timeframe: str = "1h"
    timerange: str = "20250101-20260101"
    holdout_start: str = "20260101"
    trend_lookback_hours: int = TREND_LOOKBACK_HOURS
    volatility_reference_hours: int = VOLATILITY_REFERENCE_HOURS
    trend_score_threshold: float = TREND_SCORE_THRESHOLD
    hac_lag_hours: int = HAC_LAG_HOURS
    fdr_alpha: float = FDR_ALPHA
    minimum_marginal_observations: int = MINIMUM_MARGINAL_OBSERVATIONS
    minimum_joint_observations: int = MINIMUM_JOINT_OBSERVATIONS
    minimum_interpretable_trades: int = MINIMUM_INTERPRETABLE_TRADES
    starting_balance: float = 10_000.0
    fee_per_side: float = 0.001
    robustness_ratios: tuple[float, ...] = PHASE10_ROBUSTNESS_RATIOS
    seeds: tuple[int, ...] = PHASE10_SEEDS
    smoke_test: bool = False

    def validate(self) -> None:
        if not self.data_directory.is_dir():
            raise FileNotFoundError(f"data directory does not exist: {self.data_directory}")
        if not Path(self.python_executable).is_file():
            raise FileNotFoundError(f"Python does not exist: {self.python_executable}")
        self._validate_references()
        self._validate_design()

    def _validate_references(self) -> None:
        references = ((self.phase10_summary, 10), (self.phase12_summary, 12))
        for path, phase in references:
            if not path.is_file():
                raise FileNotFoundError(f"Phase 13 requires completed Phase {phase}: {path}")
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary.get("phase") != phase or summary.get("gate", {}).get("passed") is not True:
                raise ValueError(f"Phase {phase} did not pass its integrity gate")
            if summary.get("design", {}).get("holdout_used") is not False:
                raise ValueError(f"Phase {phase} does not certify a sealed holdout")
        if not self.phase10_map.is_file():
            raise FileNotFoundError("Phase 13 requires the Phase 10 representation map")

    def _validate_design(self) -> None:
        if self.timeframe != "1h":
            raise ValueError("Phase 13 is frozen to 1h observations")
        if not self.smoke_test:
            frozen = (
                (self.trend_lookback_hours, TREND_LOOKBACK_HOURS, "trend lookback"),
                (
                    self.volatility_reference_hours,
                    VOLATILITY_REFERENCE_HOURS,
                    "volatility reference",
                ),
                (self.trend_score_threshold, TREND_SCORE_THRESHOLD, "trend threshold"),
                (self.hac_lag_hours, HAC_LAG_HOURS, "HAC lag"),
                (self.fdr_alpha, FDR_ALPHA, "FDR alpha"),
                (
                    self.minimum_marginal_observations,
                    MINIMUM_MARGINAL_OBSERVATIONS,
                    "marginal sample floor",
                ),
                (
                    self.minimum_joint_observations,
                    MINIMUM_JOINT_OBSERVATIONS,
                    "joint sample floor",
                ),
                (self.robustness_ratios, PHASE10_ROBUSTNESS_RATIOS, "robustness ratios"),
                (self.seeds, PHASE10_SEEDS, "seeds"),
            )
            for actual, expected, label in frozen:
                if actual != expected:
                    raise ValueError(f"the full Phase 13 run freezes the {label}")
        if self.trend_lookback_hours < 2 or self.volatility_reference_hours < 2:
            raise ValueError("regime lookbacks must be at least two hours")
        if self.trend_score_threshold <= 0 or self.hac_lag_hours < 0:
            raise ValueError("trend threshold must be positive and HAC lag non-negative")
        if not 0 < self.fdr_alpha < 1:
            raise ValueError("FDR alpha must be in (0, 1)")
        if self.minimum_marginal_observations < self.minimum_joint_observations:
            raise ValueError("marginal sample floor must not be below the joint floor")
        if self.starting_balance <= 0 or self.fee_per_side != 0.001:
            raise ValueError("Phase 13 freezes a 10,000 wallet and 0.1% fee per side")
        start, end = _parse_timerange(self.timerange)
        holdout = datetime.strptime(self.holdout_start, "%Y%m%d").replace(tzinfo=UTC)
        if start >= end or end > holdout:
            raise ValueError("Phase 13 may not enter the sealed holdout")


def causal_regime_features(
    market: pd.DataFrame,
    trend_lookback_hours: int = TREND_LOOKBACK_HOURS,
    volatility_reference_hours: int = VOLATILITY_REFERENCE_HOURS,
    trend_score_threshold: float = TREND_SCORE_THRESHOLD,
) -> pd.DataFrame:
    """Label regimes from trailing prices only; the forward return is evaluation-only."""
    frame = market.sort_values("date").drop_duplicates("date").copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    log_price = np.log(frame["close"].astype(np.float64))
    hourly_log_return = log_price.diff()
    frame["trailing_log_return"] = log_price - log_price.shift(trend_lookback_hours)
    frame["trailing_realized_volatility"] = hourly_log_return.rolling(trend_lookback_hours).std(
        ddof=1
    ) * math.sqrt(trend_lookback_hours)
    frame["trend_score"] = frame["trailing_log_return"] / frame["trailing_realized_volatility"]
    frame["volatility_reference"] = (
        frame["trailing_realized_volatility"].shift(1).rolling(volatility_reference_hours).median()
    )
    frame["trend_regime"] = "sideways"
    frame.loc[frame["trend_score"] >= trend_score_threshold, "trend_regime"] = "bull"
    frame.loc[frame["trend_score"] <= -trend_score_threshold, "trend_regime"] = "bear"
    frame["volatility_regime"] = np.where(
        frame["trailing_realized_volatility"] > frame["volatility_reference"],
        "high_volatility",
        "low_volatility",
    )
    unavailable = frame[["trend_score", "volatility_reference"]].isna().any(axis=1)
    frame.loc[unavailable, ["trend_regime", "volatility_regime"]] = pd.NA
    frame["joint_regime"] = (
        frame["trend_regime"].astype("string") + "_" + frame["volatility_regime"].astype("string")
    )
    frame["realized_forward_return"] = frame["close"].shift(-1) / frame["close"] - 1.0
    return frame


def load_regime_frame(config: Phase13Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    start, end = _parse_timerange(config.timerange)
    required_history = config.trend_lookback_hours + config.volatility_reference_hours + 1
    history_start = start - timedelta(hours=required_history)
    requested = TimeRange(
        starttype="date",
        stoptype="date",
        startts=int(history_start.timestamp()),
        stopts=int(end.timestamp()),
    )
    market = load_pair_history(
        pair=config.pair,
        timeframe=config.timeframe,
        datadir=config.data_directory,
        timerange=requested,
        fill_up_missing=False,
        drop_incomplete=False,
        data_format="feather",
        candle_type=CandleType.FUTURES,
    )
    market = market.sort_values("date").drop_duplicates("date")
    market = market.loc[(market["date"] >= history_start) & (market["date"] < end)].copy()
    expected = pd.date_range(market["date"].min(), market["date"].max(), freq="1h", tz="UTC")
    regimes = causal_regime_features(
        market,
        config.trend_lookback_hours,
        config.volatility_reference_hours,
        config.trend_score_threshold,
    )
    regimes = regimes.loc[(regimes["date"] >= start) & (regimes["date"] < end)].copy()
    valid_target = np.isfinite(regimes["realized_forward_return"])
    classified = regimes["joint_regime"].notna()
    evaluated = regimes.loc[valid_target & classified].copy()
    counts = {
        f"{axis}:{regime}": int((evaluated[_regime_column(axis)] == regime).sum())
        for axis, regime in REGIME_CELLS
    }
    audit = {
        "loaded_rows_with_prehistory": len(market),
        "available_start": market["date"].min().isoformat(),
        "available_end": market["date"].max().isoformat(),
        "missing_hour_count": len(expected.difference(pd.DatetimeIndex(market["date"]))),
        "duplicate_timestamp_count": int(market["date"].duplicated().sum()),
        "oos_rows": len(regimes),
        "classified_valid_target_rows": len(evaluated),
        "regime_counts": counts,
        "classification_start": evaluated["date"].min().isoformat(),
        "classification_end": evaluated["date"].max().isoformat(),
    }
    return regimes, audit


def _regime_column(axis: str) -> str:
    columns = {
        "trend": "trend_regime",
        "volatility": "volatility_regime",
        "joint": "joint_regime",
    }
    try:
        return columns[axis]
    except KeyError as exc:
        raise ValueError(f"unknown regime axis: {axis}") from exc


def load_source_cases(config: Phase13Config) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    phase10_runs = config.phase10_summary.parent / "runs" / "market_rff"
    for path in sorted(phase10_runs.glob("seed-*/summary.json")):
        substudy = json.loads(path.read_text(encoding="utf-8"))
        if substudy.get("gate", {}).get("passed") is not True:
            raise ValueError(f"Phase 10 substudy failed: {path}")
        for result in substudy["results"]:
            cases.append(
                {
                    "source_id": (f"market_rff-p{result['feature_count']}-seed{result['seed']}"),
                    "phase": 10,
                    "source_type": "market_rff",
                    "model_name": "market_rff",
                    "classification": "prediction",
                    "seed": int(result["seed"]),
                    "target_pn_ratio": float(result["target_pn_ratio"]),
                    "actual_pn_ratio": float(result["actual_pn_ratio"]),
                    "feature_count": int(result["feature_count"]),
                    "prediction_directory": Path(result["artifacts"]["prediction_directory"]),
                    "backtest": Path(result["artifacts"]["backtest"]),
                    "overall_profit_total": float(result["trading"]["profit_total"]),
                }
            )
    phase12 = json.loads(config.phase12_summary.read_text(encoding="utf-8"))
    for result in phase12["results"]:
        cases.append(
            {
                "source_id": f"baseline-{result['baseline']}",
                "phase": 12,
                "source_type": "simple_baseline",
                "model_name": result["baseline"],
                "classification": result["classification"],
                "seed": None,
                "target_pn_ratio": float(result["target_pn_ratio"]),
                "actual_pn_ratio": float(result["actual_pn_ratio"]),
                "feature_count": int(result["feature_count"]),
                "prediction_directory": Path(result["artifacts"]["prediction_directory"]),
                "backtest": Path(result["artifacts"]["backtest"]),
                "overall_profit_total": float(result["trading"]["profit_total"]),
            }
        )
    source_ids = [case["source_id"] for case in cases]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Phase 13 source identifiers must be unique")
    return cases


def hac_mean_test(values: np.ndarray, maximum_lag: int = HAC_LAG_HOURS) -> dict[str, float]:
    """Two-sided asymptotic test of a mean with a Bartlett HAC variance."""
    sample = np.asarray(values, dtype=np.float64)
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return {"mean": math.nan, "standard_error": math.nan, "t_stat": math.nan, "p": math.nan}
    mean = float(np.mean(sample))
    centered = sample - mean
    gamma_zero = float(centered @ centered / sample.size)
    long_run_variance = gamma_zero
    used_lag = min(maximum_lag, sample.size - 1)
    for lag in range(1, used_lag + 1):
        weight = 1.0 - lag / (used_lag + 1.0)
        covariance = float(centered[lag:] @ centered[:-lag] / sample.size)
        long_run_variance += 2.0 * weight * covariance
    standard_error = math.sqrt(max(long_run_variance, 0.0) / sample.size)
    numerical_floor = np.finfo(np.float64).eps * max(abs(mean), 1.0)
    if standard_error <= numerical_floor:
        if abs(mean) <= numerical_floor:
            return {"mean": mean, "standard_error": 0.0, "t_stat": 0.0, "p": 1.0}
        return {
            "mean": mean,
            "standard_error": 0.0,
            "t_stat": math.copysign(math.inf, mean),
            "p": 0.0,
        }
    statistic = mean / standard_error
    return {
        "mean": mean,
        "standard_error": standard_error,
        "t_stat": statistic,
        "p": float(2.0 * norm.sf(abs(statistic))),
    }


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full(values.shape, np.nan)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if finite_indices.size == 0:
        return adjusted.tolist()
    ordered = finite_indices[np.argsort(values[finite_indices])]
    count = ordered.size
    running = 1.0
    for rank in range(count, 0, -1):
        index = ordered[rank - 1]
        running = min(running, float(values[index]) * count / rank)
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def _evaluate_case_predictions(
    case: dict[str, Any],
    regimes: pd.DataFrame,
    config: Phase13Config,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions = _read_predictions(case["prediction_directory"])
    merged = predictions.merge(
        regimes[
            [
                "date",
                "realized_forward_return",
                "trend_regime",
                "volatility_regime",
                "joint_regime",
            ]
        ],
        on="date",
        how="left",
        validate="one_to_one",
    )
    valid = (
        (merged["do_predict"] == 1)
        & np.isfinite(merged["&-forward_return"])
        & np.isfinite(merged["realized_forward_return"])
        & merged["joint_regime"].notna()
    )
    evaluated = merged.loc[valid].copy()
    rows = []
    for axis, regime in REGIME_CELLS:
        matched = evaluated.loc[evaluated[_regime_column(axis)] == regime]
        actual = matched["realized_forward_return"].to_numpy(dtype=np.float64)
        predicted = matched["&-forward_return"].to_numpy(dtype=np.float64)
        model = prediction_metrics(actual, predicted)
        zero = prediction_metrics(actual, np.zeros_like(actual))
        loss_difference = np.square(predicted - actual) - np.square(actual)
        test = hac_mean_test(loss_difference, config.hac_lag_hours)
        rows.append(
            {
                "source_id": case["source_id"],
                "phase": case["phase"],
                "source_type": case["source_type"],
                "model_name": case["model_name"],
                "classification": case["classification"],
                "seed": case["seed"],
                "target_pn_ratio": case["target_pn_ratio"],
                "actual_pn_ratio": case["actual_pn_ratio"],
                "feature_count": case["feature_count"],
                "regime_axis": axis,
                "regime": regime,
                "minimum_observation_requirement": (
                    config.minimum_joint_observations
                    if axis == "joint"
                    else config.minimum_marginal_observations
                ),
                "sample_adequate": len(matched)
                >= (
                    config.minimum_joint_observations
                    if axis == "joint"
                    else config.minimum_marginal_observations
                ),
                **{f"oos_{key}": value for key, value in model.items()},
                **{f"zero_{key}": value for key, value in zero.items()},
                "mse_over_zero": float(model["mse"]) / float(zero["mse"]),
                "loss_difference_vs_zero_mean": test["mean"],
                "loss_difference_hac_standard_error": test["standard_error"],
                "loss_difference_hac_t_stat": test["t_stat"],
                "loss_difference_hac_p_unadjusted": test["p"],
            }
        )
    integrity = {
        "prediction_rows": len(predictions),
        "valid_classified_rows": len(evaluated),
        "duplicate_prediction_dates": int(predictions["date"].duplicated().sum()),
        "all_predictions_accepted": bool((predictions["do_predict"] == 1).all()),
        "prediction_start": predictions["date"].min().isoformat(),
        "prediction_end": predictions["date"].max().isoformat(),
    }
    return rows, integrity


def apply_fdr(prediction_rows: list[dict[str, Any]], alpha: float) -> None:
    family = [
        row
        for row in prediction_rows
        if row["regime_axis"] != "joint" and row["model_name"] != "zero_return"
    ]
    adjusted = benjamini_hochberg(
        [float(row["loss_difference_hac_p_unadjusted"]) for row in family]
    )
    for row in prediction_rows:
        row["loss_difference_fdr_q"] = math.nan
        row["significantly_better_than_zero_fdr"] = False
        row["significantly_worse_than_zero_fdr"] = False
    for row, q_value in zip(family, adjusted, strict=True):
        row["loss_difference_fdr_q"] = q_value
        row["significantly_better_than_zero_fdr"] = bool(
            q_value <= alpha and float(row["loss_difference_vs_zero_mean"]) < 0
        )
        row["significantly_worse_than_zero_fdr"] = bool(
            q_value <= alpha and float(row["loss_difference_vs_zero_mean"]) > 0
        )


def evaluate_predictions(
    cases: list[dict[str, Any]],
    regimes: pd.DataFrame,
    config: Phase13Config,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    integrity = {}
    for case in cases:
        if case["classification"] != "prediction":
            continue
        case_rows, case_integrity = _evaluate_case_predictions(case, regimes, config)
        rows.extend(case_rows)
        integrity[case["source_id"]] = case_integrity
    apply_fdr(rows, config.fdr_alpha)
    return rows, integrity


def _trade_metrics(trades: pd.DataFrame, config: Phase13Config) -> dict[str, Any]:
    if trades.empty:
        return {
            "trade_count": 0,
            "profit_total_contribution": 0.0,
            "profit_abs": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "long_trades": 0,
            "short_trades": 0,
            "turnover_multiple": 0.0,
            "mean_fee_per_side": math.nan,
        }
    wins = trades.loc[trades["profit_abs"] > 0, "profit_abs"].sum()
    losses = trades.loc[trades["profit_abs"] < 0, "profit_abs"].sum()
    return {
        "trade_count": len(trades),
        "profit_total_contribution": float(trades["profit_abs"].sum() / config.starting_balance),
        "profit_abs": float(trades["profit_abs"].sum()),
        "profit_factor": float(wins / abs(losses)) if losses < 0 else math.inf,
        "win_rate": float((trades["profit_abs"] > 0).mean()),
        "long_trades": int((~trades["is_short"]).sum()),
        "short_trades": int(trades["is_short"].sum()),
        "turnover_multiple": float(trades["stake_amount"].sum() / config.starting_balance),
        "mean_fee_per_side": float((trades["fee_open"] + trades["fee_close"]).mean() / 2),
    }


def evaluate_economics(
    cases: list[dict[str, Any]],
    regimes: pd.DataFrame,
    config: Phase13Config,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    regime_lookup = regimes.set_index("date")
    rows = []
    integrity = {}
    for case in cases:
        trades = load_backtest_data(case["backtest"], strategy="Phase4RFFStrategy")
        if trades.empty:
            trades = trades.copy()
            trades["signal_date"] = pd.Series(dtype="datetime64[ns, UTC]")
            trades["trend_regime"] = pd.Series(dtype="string")
            trades["volatility_regime"] = pd.Series(dtype="string")
            trades["joint_regime"] = pd.Series(dtype="string")
        else:
            trades = trades.copy()
            trades["signal_date"] = pd.to_datetime(trades["open_date"], utc=True) - pd.Timedelta(
                hours=1
            )
            labels = regime_lookup.reindex(trades["signal_date"])[
                ["trend_regime", "volatility_regime", "joint_regime"]
            ].reset_index(drop=True)
            for column in labels:
                trades[column] = labels[column].to_numpy()
        unmapped = int(trades["joint_regime"].isna().sum()) if not trades.empty else 0
        for axis, regime in REGIME_CELLS:
            matched = trades.loc[trades[_regime_column(axis)] == regime]
            metrics = _trade_metrics(matched, config)
            rows.append(
                {
                    "source_id": case["source_id"],
                    "phase": case["phase"],
                    "source_type": case["source_type"],
                    "model_name": case["model_name"],
                    "classification": case["classification"],
                    "seed": case["seed"],
                    "target_pn_ratio": case["target_pn_ratio"],
                    "actual_pn_ratio": case["actual_pn_ratio"],
                    "feature_count": case["feature_count"],
                    "regime_axis": axis,
                    "regime": regime,
                    "minimum_interpretable_trades": config.minimum_interpretable_trades,
                    "trade_sample_adequate": metrics["trade_count"]
                    >= config.minimum_interpretable_trades,
                    "overall_profit_total": case["overall_profit_total"],
                    **metrics,
                }
            )
        axis_contributions = {
            axis: sum(
                float(row["profit_total_contribution"])
                for row in rows
                if row["source_id"] == case["source_id"] and row["regime_axis"] == axis
            )
            for axis in ("trend", "volatility", "joint")
        }
        integrity[case["source_id"]] = {
            "trade_count": len(trades),
            "unmapped_trade_count": unmapped,
            "axis_profit_contributions": axis_contributions,
            "overall_profit_total": case["overall_profit_total"],
            "all_axis_contributions_match_overall": all(
                math.isclose(
                    contribution,
                    case["overall_profit_total"],
                    rel_tol=0,
                    abs_tol=1e-12,
                )
                for contribution in axis_contributions.values()
            ),
        }
    return rows, integrity


def assess_regime_curves(
    prediction_rows: list[dict[str, Any]],
    config: Phase13Config,
) -> list[dict[str, Any]]:
    phase10_map = pd.read_csv(config.phase10_map)
    training_lookup = {
        (int(row.seed), float(row.target_pn_ratio)): float(row.train_interpolated_window_fraction)
        for row in phase10_map.itertuples()
        if row.representation == "market_rff"
    }
    assessments = []
    for axis, regime in REGIME_CELLS:
        seeds = config.seeds if axis != "joint" else (config.seeds[0],)
        for seed in seeds:
            matched = [
                row
                for row in prediction_rows
                if row["source_type"] == "market_rff"
                and row["seed"] == seed
                and row["regime_axis"] == axis
                and row["regime"] == regime
                and (axis == "joint" or row["target_pn_ratio"] in config.robustness_ratios)
            ]
            curve_rows = []
            for row in matched:
                copy = dict(row)
                copy["train_interpolated_window_fraction"] = training_lookup[
                    (int(row["seed"]), float(row["target_pn_ratio"]))
                ]
                copy["oos_mse"] = row["oos_mse"]
                copy["zero_mse"] = row["zero_mse"]
                curve_rows.append(copy)
            assessments.append(
                {
                    "regime_axis": axis,
                    "regime": regime,
                    "seed": seed,
                    "observation_count": (
                        int(curve_rows[0]["oos_observation_count"]) if curve_rows else 0
                    ),
                    **assess_flat_curve(curve_rows),
                }
            )
    return assessments


def summarize_economic_regimes(economic_rows: list[dict[str, Any]]) -> dict[str, Any]:
    marginal = [row for row in economic_rows if row["regime_axis"] != "joint"]
    market_rff = [row for row in marginal if row["source_type"] == "market_rff"]
    active_sources = {
        row["source_id"]: float(row["overall_profit_total"])
        for row in marginal
        if row["model_name"] != "zero_return"
    }
    positive_rff = [
        row
        for row in market_rff
        if float(row["profit_total_contribution"]) > 0
        and int(row["trade_count"]) >= int(row["minimum_interpretable_trades"])
    ]
    best_positive = (
        max(positive_rff, key=lambda row: float(row["profit_total_contribution"]))
        if positive_rff
        else None
    )
    return {
        "interpretation": "descriptive trade PnL attribution; excluded from model inference",
        "active_source_count": len(active_sources),
        "active_source_count_losing_overall": sum(value < 0 for value in active_sources.values()),
        "market_rff_marginal_regime_cell_count": len(market_rff),
        "market_rff_positive_adequate_cell_count": len(positive_rff),
        "market_rff_positive_adequate_cell_fraction": (
            len(positive_rff) / len(market_rff) if market_rff else math.nan
        ),
        "best_positive_market_rff_cell": (
            {
                "source_id": best_positive["source_id"],
                "regime_axis": best_positive["regime_axis"],
                "regime": best_positive["regime"],
                "trade_count": best_positive["trade_count"],
                "profit_total_contribution": best_positive["profit_total_contribution"],
                "profit_factor": best_positive["profit_factor"],
                "overall_profit_total": best_positive["overall_profit_total"],
            }
            if best_positive
            else None
        ),
        "isolated_regime_profit_rescues_any_overall_model": any(
            value >= 0 for value in active_sources.values()
        ),
    }


def compare_regimes(
    prediction_rows: list[dict[str, Any]],
    curve_assessments: list[dict[str, Any]],
    config: Phase13Config,
) -> dict[str, Any]:
    comparisons = []
    for axis, regime in REGIME_CELLS:
        matched = [
            row for row in prediction_rows if row["regime_axis"] == axis and row["regime"] == regime
        ]
        simple = [row for row in matched if row["source_type"] == "simple_baseline"]
        rff = [row for row in matched if row["source_type"] == "market_rff"]
        zero = next(row for row in simple if row["model_name"] == "zero_return")
        best_simple = min(simple, key=lambda row: float(row["oos_mse"]))
        best_learnable = min(
            (row for row in simple if row["model_name"] != "zero_return"),
            key=lambda row: float(row["oos_mse"]),
        )
        best_rff = min(rff, key=lambda row: float(row["oos_mse"]))
        comparisons.append(
            {
                "regime_axis": axis,
                "regime": regime,
                "observation_count": int(zero["oos_observation_count"]),
                "best_simple_model": best_simple["model_name"],
                "best_simple_mse": best_simple["oos_mse"],
                "best_learnable_model": best_learnable["model_name"],
                "best_learnable_mse_over_zero": float(best_learnable["mse_over_zero"]),
                "best_rff_seed": best_rff["seed"],
                "best_rff_pn_ratio": best_rff["actual_pn_ratio"],
                "best_rff_feature_count": best_rff["feature_count"],
                "best_rff_mse_over_zero": best_rff["mse_over_zero"],
                "best_rff_mse_over_best_simple": float(best_rff["oos_mse"])
                / float(best_simple["oos_mse"]),
                "rff_case_count_beating_zero": sum(
                    float(row["oos_mse"]) < float(row["zero_mse"]) for row in rff
                ),
                "rff_case_count_significantly_better_zero_fdr": sum(
                    bool(row["significantly_better_than_zero_fdr"]) for row in rff
                ),
                "rff_case_count_significantly_worse_zero_fdr": sum(
                    bool(row["significantly_worse_than_zero_fdr"]) for row in rff
                ),
            }
        )
    marginal_assessments = [row for row in curve_assessments if row["regime_axis"] != "joint"]
    majority = math.ceil(len(config.seeds) / 2)
    pattern_counts = {
        f"{axis}:{regime}": sum(
            bool(row.get("double_descent_pattern_detected"))
            for row in marginal_assessments
            if row["regime_axis"] == axis and row["regime"] == regime
        )
        for axis, regime in MARGINAL_REGIMES
    }
    useful_counts = {
        f"{axis}:{regime}": sum(
            bool(row.get("largest_model_beats_zero_mse"))
            and bool(row.get("second_descent_beats_best_underparameterized"))
            for row in marginal_assessments
            if row["regime_axis"] == axis and row["regime"] == regime
        )
        for axis, regime in MARGINAL_REGIMES
    }
    return {
        "primary_metric": "chronological OOS MSE within causally known regimes",
        "regime_comparisons": comparisons,
        "double_descent_seed_counts_by_marginal_regime": pattern_counts,
        "useful_benign_overfitting_seed_counts_by_marginal_regime": useful_counts,
        "majority_seed_requirement": majority,
        "double_descent_persists_in_every_marginal_regime": all(
            count >= majority for count in pattern_counts.values()
        ),
        "double_descent_is_regime_concentrated": any(
            count >= majority for count in pattern_counts.values()
        )
        and not all(count >= majority for count in pattern_counts.values()),
        "useful_benign_overfitting_detected_in_any_marginal_regime": any(
            count >= majority for count in useful_counts.values()
        ),
        "any_market_rff_significantly_beats_zero_after_fdr": any(
            row["source_type"] == "market_rff"
            and row["regime_axis"] != "joint"
            and row["significantly_better_than_zero_fdr"]
            for row in prediction_rows
        ),
        "multiple_testing_family": {
            "method": "Benjamini-Hochberg",
            "alpha": config.fdr_alpha,
            "hypotheses": sum(
                row["regime_axis"] != "joint" and row["model_name"] != "zero_return"
                for row in prediction_rows
            ),
            "test": "two-sided Bartlett-HAC mean squared-error differential versus zero",
            "hac_lag_hours": config.hac_lag_hours,
        },
    }


def evaluate_phase13_gate(
    config: Phase13Config,
    cases: list[dict[str, Any]],
    regime_audit: dict[str, Any],
    prediction_rows: list[dict[str, Any]],
    prediction_integrity: dict[str, dict[str, Any]],
    economic_rows: list[dict[str, Any]],
    economic_integrity: dict[str, dict[str, Any]],
    curve_assessments: list[dict[str, Any]],
) -> dict[str, Any]:
    prediction_cases = [case for case in cases if case["classification"] == "prediction"]
    marginal_counts = [
        regime_audit["regime_counts"][f"{axis}:{regime}"] for axis, regime in MARGINAL_REGIMES
    ]
    joint_counts = [
        regime_audit["regime_counts"][f"{axis}:{regime}"] for axis, regime in JOINT_REGIMES
    ]
    expected_prediction_rows = len(prediction_cases) * len(REGIME_CELLS)
    expected_economic_rows = len(cases) * len(REGIME_CELLS)
    checks = {
        "phase10_market_rff_case_count": sum(case["source_type"] == "market_rff" for case in cases)
        == PHASE10_MARKET_RFF_CASE_COUNT,
        "phase12_case_count": sum(case["phase"] == 12 for case in cases) == PHASE12_CASE_COUNT,
        "complete_prediction_metric_map": len(prediction_rows) == expected_prediction_rows,
        "complete_economic_metric_map": len(economic_rows) == expected_economic_rows,
        "source_artifacts_exist": all(
            case["prediction_directory"].is_dir() and case["backtest"].is_file() for case in cases
        ),
        "market_history_has_no_gaps_or_duplicates": regime_audit["missing_hour_count"] == 0
        and regime_audit["duplicate_timestamp_count"] == 0,
        "every_oos_target_classified": regime_audit["classified_valid_target_rows"] == 8_759,
        "marginal_regime_samples_are_adequate": min(marginal_counts)
        >= config.minimum_marginal_observations,
        "joint_regime_samples_are_adequate": min(joint_counts) >= config.minimum_joint_observations,
        "prediction_sources_have_identical_coverage": bool(prediction_integrity)
        and len({int(item["valid_classified_rows"]) for item in prediction_integrity.values()}) == 1
        and all(
            item["duplicate_prediction_dates"] == 0 and item["all_predictions_accepted"]
            for item in prediction_integrity.values()
        ),
        "all_trade_entries_mapped_to_signal_regime": bool(economic_integrity)
        and all(item["unmapped_trade_count"] == 0 for item in economic_integrity.values()),
        "regime_profit_contributions_reproduce_backtests": bool(economic_integrity)
        and all(
            item["all_axis_contributions_match_overall"] for item in economic_integrity.values()
        ),
        "fee_assumption_preserved": config.fee_per_side == 0.001
        and all(
            math.isclose(float(row["mean_fee_per_side"]), config.fee_per_side, abs_tol=1e-15)
            for row in economic_rows
            if int(row["trade_count"]) > 0
        ),
        "curve_assessment_map_complete": len(curve_assessments)
        == len(MARGINAL_REGIMES) * len(config.seeds) + len(JOINT_REGIMES),
        "regimes_are_diagnostic_not_adaptive": True,
        "holdout_was_not_used": _parse_timerange(config.timerange)[1]
        <= datetime.strptime(config.holdout_start, "%Y%m%d").replace(tzinfo=UTC),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "expected_prediction_source_count": len(prediction_cases),
        "observed_prediction_source_count": len(prediction_integrity),
        "expected_economic_source_count": len(cases),
        "observed_economic_source_count": len(economic_integrity),
    }


def _write_plot(
    path: Path,
    prediction_rows: list[dict[str, Any]],
    comparisons: dict[str, Any],
    config: Phase13Config,
) -> bool:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return False
    reference = [
        row
        for row in prediction_rows
        if row["source_type"] == "market_rff"
        and row["seed"] == config.seeds[0]
        and row["regime_axis"] != "joint"
    ]
    if not reference:
        return False
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Market RFF MSE / zero by trend regime",
            "Market RFF MSE / zero by volatility regime",
            "Best RFF versus best simple baseline",
            "RFF double-descent detections across seeds",
        ),
    )
    for axis, regimes, column in (
        ("trend", TREND_REGIMES, 1),
        ("volatility", VOLATILITY_REGIMES, 2),
    ):
        for regime in regimes:
            matched = sorted(
                (
                    row
                    for row in reference
                    if row["regime_axis"] == axis and row["regime"] == regime
                ),
                key=lambda row: float(row["actual_pn_ratio"]),
            )
            figure.add_trace(
                go.Scatter(
                    x=[row["actual_pn_ratio"] for row in matched],
                    y=[row["mse_over_zero"] for row in matched],
                    mode="lines+markers",
                    name=regime,
                    legendgroup=regime,
                    showlegend=axis == "trend",
                ),
                row=1,
                col=column,
            )
    marginal = [row for row in comparisons["regime_comparisons"] if row["regime_axis"] != "joint"]
    labels = [f"{row['regime_axis']}:{row['regime']}" for row in marginal]
    figure.add_trace(
        go.Bar(
            x=labels,
            y=[row["best_rff_mse_over_best_simple"] for row in marginal],
            name="Best RFF / best simple MSE",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    pattern_counts = comparisons["double_descent_seed_counts_by_marginal_regime"]
    figure.add_trace(
        go.Bar(
            x=list(pattern_counts),
            y=list(pattern_counts.values()),
            name="Seeds detecting double descent",
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    figure.update_xaxes(type="log", title_text="P/N", row=1, col=1)
    figure.update_xaxes(type="log", title_text="P/N", row=1, col=2)
    figure.add_hline(y=1.0, line_dash="dash", row=1, col=1)
    figure.add_hline(y=1.0, line_dash="dash", row=1, col=2)
    figure.add_hline(y=1.0, line_dash="dash", row=2, col=1)
    figure.update_xaxes(tickangle=-30, row=2)
    figure.update_layout(
        title="Phase 13 - Causal market-regime decomposition",
        template="plotly_white",
        height=950,
        width=1500,
    )
    figure.write_html(path, include_plotlyjs="cdn")
    return True


def run_phase13(config: Phase13Config) -> dict[str, Any]:
    config.validate()
    config.output_directory.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    regimes, regime_audit = load_regime_frame(config)
    cases = load_source_cases(config)
    prediction_rows, prediction_integrity = evaluate_predictions(cases, regimes, config)
    economic_rows, economic_integrity = evaluate_economics(cases, regimes, config)
    curve_assessments = assess_regime_curves(prediction_rows, config)
    comparisons = compare_regimes(prediction_rows, curve_assessments, config)
    economic_diagnostics = summarize_economic_regimes(economic_rows)
    gate = evaluate_phase13_gate(
        config,
        cases,
        regime_audit,
        prediction_rows,
        prediction_integrity,
        economic_rows,
        economic_integrity,
        curve_assessments,
    )
    assignment_path = config.output_directory / "regime_assignments.csv"
    prediction_path = config.output_directory / "prediction_regime_metrics.csv"
    economic_path = config.output_directory / "economic_regime_metrics.csv"
    curve_path = config.output_directory / "regime_curve_assessments.csv"
    comparison_path = config.output_directory / "regime_comparisons.csv"
    plot_path = config.output_directory / "market_regimes.html"
    summary_path = config.output_directory / "summary.json"
    regimes.loc[
        regimes["realized_forward_return"].notna(),
        [
            "date",
            "close",
            "trailing_log_return",
            "trailing_realized_volatility",
            "volatility_reference",
            "trend_score",
            "trend_regime",
            "volatility_regime",
            "joint_regime",
            "realized_forward_return",
        ],
    ].to_csv(assignment_path, index=False)
    _write_csv(prediction_path, prediction_rows)
    _write_csv(economic_path, economic_rows)
    _write_csv(curve_path, curve_assessments)
    _write_csv(comparison_path, comparisons["regime_comparisons"])
    plot_written = _write_plot(plot_path, prediction_rows, comparisons, config)
    config_payload = asdict(config)
    for key in (
        "data_directory",
        "output_directory",
        "phase10_summary",
        "phase10_map",
        "phase12_summary",
    ):
        config_payload[key] = str(config_payload[key])
    summary = {
        "phase": 13,
        "objective": (
            "Determine whether double descent or apparent predictability is regime-specific"
        ),
        "scope": "development-only causal regime decomposition; no retraining; 2026 holdout sealed",
        "run_id": run_id,
        "design": {
            "trend_definition": (
                "trailing 30-day log return divided by trailing 30-day realized volatility; "
                "bull >= 0.5, bear <= -0.5, otherwise sideways"
            ),
            "volatility_definition": (
                "trailing 30-day realized volatility above/below the prior 60-day rolling median"
            ),
            "regime_known_at_prediction_time": True,
            "regime_used_to_change_predictions_or_trades": False,
            "parameters_selected_after_viewing_regime_results": False,
            "joint_regimes_are_descriptive": True,
            "trading_used_for_inference": False,
            "holdout_used": False,
        },
        "config": config_payload,
        "regime_audit": regime_audit,
        "source_cases": [
            {key: str(value) if isinstance(value, Path) else value for key, value in case.items()}
            for case in cases
        ],
        "prediction_integrity": prediction_integrity,
        "economic_integrity": economic_integrity,
        "curve_assessments": curve_assessments,
        "comparisons": comparisons,
        "economic_diagnostics": economic_diagnostics,
        "gate": gate,
        "compute": {
            "prediction_source_count": len(prediction_integrity),
            "economic_source_count": len(economic_integrity),
            "prediction_regime_cell_count": len(prediction_rows),
            "economic_regime_cell_count": len(economic_rows),
            "new_model_fit_count": 0,
        },
        "limitations": {
            "single_asset": "BTC only until Phase 14 replication",
            "period_length": "one development year may contain limited independent regimes",
            "regime_thresholds": "fixed simple labels are not uniquely correct market states",
            "cost_model": "Freqtrade fee model; no new slippage or market-impact model",
            "economic_conditioning": "trades attributed to the regime at signal time",
        },
        "artifacts": {
            "summary": str(summary_path),
            "regime_assignments": str(assignment_path),
            "prediction_regime_metrics": str(prediction_path),
            "economic_regime_metrics": str(economic_path),
            "regime_curve_assessments": str(curve_path),
            "regime_comparisons": str(comparison_path),
            "interactive_plot": str(plot_path) if plot_written else None,
        },
    }
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    return summary
