"""Phase 16: frozen economic and execution-cost robustness for Phase 15 trades."""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from freqtrade.data.btanalysis import load_backtest_data, load_backtest_stats
from research.double_descent.phase10 import PHASE10_ROBUSTNESS_RATIOS, _json_safe
from research.double_descent.phase12 import PHASE12_BASELINES
from research.double_descent.phase15 import PHASE15_ASSETS, PHASE15_STRATEGY


PHASE16_PRIMARY_STUDY = "native_15m"
PHASE16_EXPECTED_RFF_CASES = 63
PHASE16_EXPECTED_BASELINE_CASES = 24
PHASE16_EXPECTED_CASES = PHASE16_EXPECTED_RFF_CASES + PHASE16_EXPECTED_BASELINE_CASES
PHASE16_MINIMUM_TRADES = 30
PHASE16_STARTING_BALANCE = 10_000.0
PHASE16_NATIVE_REPRODUCTION_TOLERANCE = 1e-5


@dataclass(frozen=True)
class CostScenario:
    name: str
    fee_per_side: float
    slippage_per_side: float
    funding_multiplier: float
    role: str

    @property
    def total_cost_per_side(self) -> float:
        return self.fee_per_side + self.slippage_per_side


PHASE16_COST_SCENARIOS = (
    CostScenario("price_only", 0.0, 0.0, 0.0, "decomposition_control"),
    CostScenario("funding_only", 0.0, 0.0, 1.0, "decomposition_control"),
    CostScenario("optimistic", 0.0005, 0.00025, 1.0, "cost_sensitivity"),
    CostScenario("phase15_reference", 0.001, 0.0, 1.0, "native_reproduction"),
    CostScenario("conservative", 0.001, 0.0005, 1.0, "cost_sensitivity"),
    CostScenario("stress", 0.002, 0.001, 1.0, "cost_sensitivity"),
)


@dataclass(frozen=True)
class EconomicCase:
    case_id: str
    asset: str
    pair: str
    model_family: str
    model_name: str
    seed: int | None
    target_pn_ratio: float | None
    feature_count: int
    backtest_path: Path


@dataclass(frozen=True)
class Phase16Config:
    phase15_summary: Path = Path("user_data/research_results/double_descent/phase15/summary.json")
    output_directory: Path = Path("user_data/research_results/double_descent/phase16")
    project_root: Path = Path()
    scenarios: tuple[CostScenario, ...] = PHASE16_COST_SCENARIOS
    starting_balance: float = PHASE16_STARTING_BALANCE
    minimum_trades: int = PHASE16_MINIMUM_TRADES
    native_reproduction_tolerance: float = PHASE16_NATIVE_REPRODUCTION_TOLERANCE
    holdout_start: str = "20260101"

    def validate(self) -> dict[str, Any]:
        if not self.phase15_summary.is_file():
            raise FileNotFoundError("Phase 16 requires the completed Phase 15 summary")
        phase15 = json.loads(self.phase15_summary.read_text(encoding="utf-8"))
        if phase15.get("phase") != 15 or phase15.get("gate", {}).get("passed") is not True:
            raise ValueError("Phase 15 did not pass its integrity gate")
        if phase15.get("design", {}).get("holdout_used") is not False:
            raise ValueError("Phase 15 does not certify a sealed holdout")
        if self.scenarios != PHASE16_COST_SCENARIOS:
            raise ValueError("Phase 16 freezes the complete predeclared cost grid")
        if self.starting_balance != PHASE16_STARTING_BALANCE:
            raise ValueError("Phase 16 freezes the Phase 15 starting balance")
        if self.minimum_trades != PHASE16_MINIMUM_TRADES:
            raise ValueError("Phase 16 freezes the minimum-trade evidence threshold")
        if self.native_reproduction_tolerance != PHASE16_NATIVE_REPRODUCTION_TOLERANCE:
            raise ValueError("Phase 16 freezes the native PnL reproduction tolerance")
        if len({scenario.name for scenario in self.scenarios}) != len(self.scenarios):
            raise ValueError("cost scenario names must be unique")
        if any(
            scenario.fee_per_side < 0
            or scenario.slippage_per_side < 0
            or scenario.funding_multiplier < 0
            for scenario in self.scenarios
        ):
            raise ValueError("cost and funding assumptions must be non-negative")
        holdout = datetime.strptime(self.holdout_start, "%Y%m%d").replace(tzinfo=UTC)
        for asset in PHASE15_ASSETS:
            end = datetime.strptime(asset.timerange.split("-")[1], "%Y%m%d").replace(tzinfo=UTC)
            if end > holdout:
                raise ValueError("Phase 16 may not enter the sealed holdout")
        return phase15


def _resolve_artifact(config: Phase16Config, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config.project_root / path


def _latest_backtest(path: Path) -> Path:
    candidates = sorted(path.rglob("*.zip"), key=lambda item: item.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no exported backtest under {path}")
    return candidates[-1]


def discover_phase15_cases(config: Phase16Config) -> list[EconomicCase]:
    """Discover every primary Phase 15 case without selecting on economic performance."""
    phase15_root = config.phase15_summary.parent
    cases: list[EconomicCase] = []
    for asset in PHASE15_ASSETS:
        cell = phase15_root / "runs" / asset.alias / PHASE16_PRIMARY_STUDY
        rff_root = cell / "rff" / "runs" / "market_rff"
        for summary_path in sorted(rff_root.glob("seed-*/summary.json")):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("gate", {}).get("passed") is not True:
                raise ValueError(f"Phase 15 substudy did not pass: {summary_path}")
            seed = int(summary["seed"])
            for result in summary.get("results", []):
                if result.get("success") is not True:
                    raise ValueError(f"Phase 15 contains a failed RFF case: {summary_path}")
                ratio = float(result["target_pn_ratio"])
                backtest = _resolve_artifact(config, result["artifacts"]["backtest"])
                if not backtest.is_file():
                    raise FileNotFoundError(f"missing Phase 15 backtest: {backtest}")
                cases.append(
                    EconomicCase(
                        case_id=f"{asset.alias}-rff-s{seed}-pn{ratio:.5f}",
                        asset=asset.alias,
                        pair=asset.pair,
                        model_family="rff",
                        model_name="market_rff",
                        seed=seed,
                        target_pn_ratio=ratio,
                        feature_count=int(result["feature_count"]),
                        backtest_path=backtest,
                    )
                )
        baseline_root = cell / "baselines" / "runs"
        for baseline in PHASE12_BASELINES:
            run_root = baseline_root / baseline
            config_path = run_root / "config.json"
            if not config_path.is_file():
                raise FileNotFoundError(f"missing Phase 15 baseline config: {config_path}")
            generated = json.loads(config_path.read_text(encoding="utf-8"))
            parameters = generated["freqai"]["model_training_parameters"]
            if (
                parameters.get("baseline") != baseline
                or generated.get("timeframe") != "15m"
                or generated["exchange"]["pair_whitelist"] != [asset.pair]
                or generated["freqai"]["feature_parameters"]["label_period_candles"] != 1
            ):
                raise ValueError(f"baseline configuration drifted: {config_path}")
            cases.append(
                EconomicCase(
                    case_id=f"{asset.alias}-baseline-{baseline}",
                    asset=asset.alias,
                    pair=asset.pair,
                    model_family="baseline",
                    model_name=baseline,
                    seed=None,
                    target_pn_ratio=None,
                    feature_count=int(parameters["feature_count"]),
                    backtest_path=_latest_backtest(run_root / "backtest"),
                )
            )
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Phase 16 discovered duplicate case identifiers")
    return cases


def decompose_trade_pnl(
    trades: pd.DataFrame,
    scenario: CostScenario,
) -> pd.DataFrame:
    """Reprice frozen positions while preserving their original amount and timing."""
    result = trades.copy()
    if result.empty:
        for column in (
            "open_notional",
            "close_notional",
            "price_pnl_abs",
            "funding_pnl_abs",
            "fee_abs",
            "slippage_abs",
            "net_pnl_abs",
        ):
            result[column] = pd.Series(dtype="float64")
        return result
    direction = np.where(result["is_short"].astype(bool), -1.0, 1.0)
    result["open_notional"] = result["amount"] * result["open_rate"]
    result["close_notional"] = result["amount"] * result["close_rate"]
    result["price_pnl_abs"] = (
        direction * result["amount"] * (result["close_rate"] - result["open_rate"])
    )
    result["funding_pnl_abs"] = result["funding_fees"].fillna(0.0) * scenario.funding_multiplier
    two_sided_notional = result["open_notional"] + result["close_notional"]
    result["fee_abs"] = two_sided_notional * scenario.fee_per_side
    result["slippage_abs"] = two_sided_notional * scenario.slippage_per_side
    result["net_pnl_abs"] = (
        result["price_pnl_abs"]
        + result["funding_pnl_abs"]
        - result["fee_abs"]
        - result["slippage_abs"]
    )
    return result


def _daily_risk_metrics(
    trades: pd.DataFrame,
    starting_balance: float,
    start: datetime,
    end: datetime,
) -> dict[str, float | bool]:
    days = pd.date_range(start, end, inclusive="left", freq="1D", tz="UTC")
    daily_pnl = pd.Series(0.0, index=days)
    if not trades.empty:
        closes = pd.to_datetime(trades["close_date"], utc=True).dt.floor("1D")
        grouped = trades.assign(close_day=closes).groupby("close_day")["net_pnl_abs"].sum()
        daily_pnl = daily_pnl.add(grouped, fill_value=0.0).reindex(days, fill_value=0.0)
    equity = starting_balance + daily_pnl.cumsum()
    prior_equity = equity.shift(1, fill_value=starting_balance)
    solvent = bool((prior_equity > 0).all() and (equity > 0).all())
    returns = daily_pnl / prior_equity.where(prior_equity > 0)
    finite = returns[np.isfinite(returns)]
    if len(finite) > 1 and float(finite.std(ddof=1)) > 0:
        sharpe = float(finite.mean() / finite.std(ddof=1) * math.sqrt(365.0))
    else:
        sharpe = 0.0
    downside = finite[finite < 0]
    downside_deviation = math.sqrt(float(np.mean(np.square(downside)))) if len(downside) else 0.0
    sortino = (
        float(finite.mean() / downside_deviation * math.sqrt(365.0))
        if downside_deviation > 0
        else 0.0
    )
    running_peak = equity.cummax().clip(lower=starting_balance)
    drawdowns = equity / running_peak - 1.0
    return {
        "daily_sharpe": sharpe,
        "daily_sortino": sortino,
        "realized_max_drawdown": float(-drawdowns.min()) if len(drawdowns) else 0.0,
        "ending_balance": float(equity.iloc[-1]) if len(equity) else starting_balance,
        "insolvent_under_frozen_exposure": not solvent,
    }


def summarize_repriced_trades(
    trades: pd.DataFrame,
    scenario: CostScenario,
    starting_balance: float,
    timerange: str,
) -> dict[str, Any]:
    start_text, end_text = timerange.split("-", 1)
    start = datetime.strptime(start_text, "%Y%m%d").replace(tzinfo=UTC)
    end = datetime.strptime(end_text, "%Y%m%d").replace(tzinfo=UTC)
    repriced = decompose_trade_pnl(trades, scenario)
    net = repriced["net_pnl_abs"] if not repriced.empty else pd.Series(dtype="float64")
    wins = net[net > 0]
    losses = net[net < 0]
    profit_factor = (
        float(wins.sum() / abs(losses.sum()))
        if float(losses.sum()) < 0
        else (math.inf if float(wins.sum()) > 0 else 0.0)
    )
    net_profit = float(net.sum()) if len(net) else 0.0
    risk = _daily_risk_metrics(repriced, starting_balance, start, end)
    duration_days = (end - start).days
    ending_balance = float(risk["ending_balance"])
    annualized_return = (
        float((ending_balance / starting_balance) ** (365.0 / duration_days) - 1.0)
        if ending_balance > 0 and duration_days > 0
        else -1.0
    )
    return {
        "scenario": scenario.name,
        "scenario_role": scenario.role,
        "fee_per_side": scenario.fee_per_side,
        "slippage_per_side": scenario.slippage_per_side,
        "total_cost_per_side": scenario.total_cost_per_side,
        "funding_multiplier": scenario.funding_multiplier,
        "trade_count": len(repriced),
        "long_trade_count": int((~repriced["is_short"].astype(bool)).sum()) if len(repriced) else 0,
        "short_trade_count": int(repriced["is_short"].astype(bool).sum()) if len(repriced) else 0,
        "win_rate": float((net > 0).mean()) if len(net) else 0.0,
        "profit_factor": profit_factor,
        "price_pnl_abs": float(repriced["price_pnl_abs"].sum()) if len(repriced) else 0.0,
        "funding_pnl_abs": float(repriced["funding_pnl_abs"].sum()) if len(repriced) else 0.0,
        "fee_abs": float(repriced["fee_abs"].sum()) if len(repriced) else 0.0,
        "slippage_abs": float(repriced["slippage_abs"].sum()) if len(repriced) else 0.0,
        "net_profit_abs": net_profit,
        "net_return": net_profit / starting_balance,
        "annualized_return": annualized_return,
        "two_sided_turnover_multiple": (
            float((repriced["open_notional"] + repriced["close_notional"]).sum()) / starting_balance
            if len(repriced)
            else 0.0
        ),
        "average_holding_hours": float(repriced["trade_duration"].mean() / 60.0)
        if len(repriced)
        else 0.0,
        "minimum_trade_count_met": len(repriced) >= PHASE16_MINIMUM_TRADES,
        **risk,
    }


def _strategy_stats(backtest_path: Path) -> dict[str, Any]:
    payload = load_backtest_stats(backtest_path)
    strategy = payload["strategy"][PHASE15_STRATEGY]
    return strategy


def _prepare_trade_frame(trades: pd.DataFrame, backtest_path: Path) -> pd.DataFrame:
    required = {
        "amount",
        "open_rate",
        "close_rate",
        "is_short",
        "funding_fees",
        "profit_abs",
        "close_date",
        "trade_duration",
    }
    if trades.empty:
        return pd.DataFrame(
            {
                **{
                    column: pd.Series(dtype="float64")
                    for column in required - {"is_short", "close_date"}
                },
                "is_short": pd.Series(dtype="bool"),
                "close_date": pd.Series(dtype="datetime64[ns, UTC]"),
            }
        )
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"backtest is missing columns {sorted(missing)}: {backtest_path}")
    return trades.sort_values("close_date").reset_index(drop=True)


def evaluate_case(config: Phase16Config, case: EconomicCase) -> list[dict[str, Any]]:
    trades = _prepare_trade_frame(load_backtest_data(case.backtest_path), case.backtest_path)
    stats = _strategy_stats(case.backtest_path)
    starting_balance = float(stats["starting_balance"])
    if not math.isclose(starting_balance, config.starting_balance, abs_tol=1e-9):
        raise ValueError(f"unexpected starting balance in {case.backtest_path}")
    asset = next(item for item in PHASE15_ASSETS if item.alias == case.asset)
    rows = []
    for scenario in config.scenarios:
        metrics = summarize_repriced_trades(trades, scenario, starting_balance, asset.timerange)
        row = {
            **asdict(case),
            **metrics,
            "backtest_path": str(case.backtest_path),
            "native_freqtrade_profit_abs": float(stats["profit_total_abs"]),
            "native_freqtrade_return": float(stats["profit_total"]),
            "native_freqtrade_sharpe": float(stats["sharpe"]),
            "native_freqtrade_sortino": float(stats["sortino"]),
            "native_freqtrade_max_drawdown": float(stats["max_drawdown_account"]),
        }
        if scenario.name == "phase15_reference":
            reconstructed = decompose_trade_pnl(trades, scenario)
            errors = (
                reconstructed["net_pnl_abs"] - trades["profit_abs"]
                if len(trades)
                else pd.Series(dtype="float64")
            )
            row["native_max_trade_pnl_error"] = float(errors.abs().max()) if len(errors) else 0.0
            row["native_total_pnl_error"] = abs(
                float(reconstructed["net_pnl_abs"].sum()) - float(stats["profit_total_abs"])
            )
        else:
            row["native_max_trade_pnl_error"] = None
            row["native_total_pnl_error"] = None
        rows.append(row)
    return rows


def _scenario_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for scenario in PHASE16_COST_SCENARIOS:
        matched = [row for row in rows if row["scenario"] == scenario.name]
        output.append(
            {
                "scenario": scenario.name,
                "case_count": len(matched),
                "positive_return_case_count": sum(row["net_return"] > 0 for row in matched),
                "positive_sharpe_case_count": sum(row["daily_sharpe"] > 0 for row in matched),
                "profit_factor_above_one_case_count": sum(
                    row["profit_factor"] > 1 for row in matched
                ),
                "minimum_trades_met_case_count": sum(
                    row["minimum_trade_count_met"] for row in matched
                ),
                "insolvent_case_count": sum(
                    row["insolvent_under_frozen_exposure"] for row in matched
                ),
                "median_net_return": statistics.median(row["net_return"] for row in matched),
                "median_daily_sharpe": statistics.median(row["daily_sharpe"] for row in matched),
            }
        )
    return output


def assess_trading_double_ascent(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assessments = []
    robust = set(PHASE10_ROBUSTNESS_RATIOS)
    for asset in (item.alias for item in PHASE15_ASSETS):
        for scenario in PHASE16_COST_SCENARIOS:
            for seed in sorted(
                {
                    int(row["seed"])
                    for row in rows
                    if row["asset"] == asset
                    and row["model_family"] == "rff"
                    and row["seed"] is not None
                }
            ):
                matched = {
                    float(row["target_pn_ratio"]): row
                    for row in rows
                    if row["asset"] == asset
                    and row["scenario"] == scenario.name
                    and row["model_family"] == "rff"
                    and int(row["seed"]) == seed
                    and float(row["target_pn_ratio"]) in robust
                }
                if set(matched) != robust:
                    raise ValueError(
                        f"incomplete robust RFF curve for {asset}/{scenario.name}/{seed}"
                    )
                low = matched[0.1]
                interpolation = matched[1.0]
                high = matched[50.0]
                assessments.append(
                    {
                        "asset": asset,
                        "scenario": scenario.name,
                        "seed": seed,
                        "low_pn_sharpe": low["daily_sharpe"],
                        "interpolation_sharpe": interpolation["daily_sharpe"],
                        "high_pn_sharpe": high["daily_sharpe"],
                        "trading_double_ascent_shape": (
                            interpolation["daily_sharpe"] < low["daily_sharpe"]
                            and high["daily_sharpe"] > interpolation["daily_sharpe"]
                        ),
                        "useful_high_complexity_economics": (
                            high["daily_sharpe"] > 0
                            and high["net_return"] > 0
                            and high["daily_sharpe"] > low["daily_sharpe"]
                            and high["minimum_trade_count_met"]
                        ),
                    }
                )
    return assessments


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return False
    figure = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=tuple(f"{asset.label} net return" for asset in PHASE15_ASSETS)
        + tuple(f"{asset.label} daily Sharpe" for asset in PHASE15_ASSETS),
    )
    for column, asset in enumerate(PHASE15_ASSETS, start=1):
        for scenario in PHASE16_COST_SCENARIOS:
            series = []
            for ratio in PHASE10_ROBUSTNESS_RATIOS:
                matched = [
                    row
                    for row in rows
                    if row["asset"] == asset.alias
                    and row["scenario"] == scenario.name
                    and row["model_family"] == "rff"
                    and float(row["target_pn_ratio"]) == ratio
                ]
                series.append(
                    (
                        ratio,
                        statistics.median(row["net_return"] for row in matched),
                        statistics.median(row["daily_sharpe"] for row in matched),
                    )
                )
            for panel_row, index in ((1, 1), (2, 2)):
                figure.add_trace(
                    go.Scatter(
                        x=[item[0] for item in series],
                        y=[item[index] for item in series],
                        mode="lines+markers",
                        name=scenario.name,
                        legendgroup=scenario.name,
                        showlegend=column == 1 and panel_row == 1,
                    ),
                    row=panel_row,
                    col=column,
                )
        figure.update_xaxes(type="log", title_text="P/N", row=1, col=column)
        figure.update_xaxes(type="log", title_text="P/N", row=2, col=column)
        figure.add_hline(y=0.0, line_dash="dash", row=1, col=column)
        figure.add_hline(y=0.0, line_dash="dash", row=2, col=column)
    figure.update_layout(
        title="Phase 16 - Frozen-trade cost and funding sensitivity",
        template="plotly_white",
        width=1600,
        height=950,
    )
    figure.write_html(path, include_plotlyjs="cdn")
    return True


def evaluate_phase16_gate(
    config: Phase16Config,
    cases: list[EconomicCase],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    native = [row for row in rows if row["scenario"] == "phase15_reference"]
    expected_rows = PHASE16_EXPECTED_CASES * len(config.scenarios)
    checks = {
        "complete_predeclared_case_count": len(cases) == PHASE16_EXPECTED_CASES,
        "complete_predeclared_rff_case_count": sum(case.model_family == "rff" for case in cases)
        == PHASE16_EXPECTED_RFF_CASES,
        "complete_predeclared_baseline_case_count": sum(
            case.model_family == "baseline" for case in cases
        )
        == PHASE16_EXPECTED_BASELINE_CASES,
        "every_case_has_every_cost_scenario": len(rows) == expected_rows,
        "case_identifiers_are_unique": len({case.case_id for case in cases}) == len(cases),
        "native_trade_pnl_reproduced": len(native) == len(cases)
        and all(
            row["native_max_trade_pnl_error"] <= config.native_reproduction_tolerance
            for row in native
        ),
        "native_total_pnl_reproduced": len(native) == len(cases)
        and all(
            row["native_total_pnl_error"] <= config.native_reproduction_tolerance for row in native
        ),
        "all_metrics_finite": bool(rows)
        and all(
            math.isfinite(float(row[key]))
            for row in rows
            for key in (
                "net_return",
                "daily_sharpe",
                "daily_sortino",
                "realized_max_drawdown",
                "two_sided_turnover_multiple",
            )
        ),
        "funding_is_explicit": all("funding_pnl_abs" in row for row in rows),
        "trading_did_not_select_cases": len(cases) == PHASE16_EXPECTED_CASES,
        "holdout_was_not_used": True,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "expected_case_count": PHASE16_EXPECTED_CASES,
        "observed_case_count": len(cases),
        "expected_result_row_count": expected_rows,
        "observed_result_row_count": len(rows),
    }


def run_phase16(config: Phase16Config) -> dict[str, Any]:
    phase15 = config.validate()
    config.output_directory.mkdir(parents=True, exist_ok=True)
    cases = discover_phase15_cases(config)
    rows = [row for case in cases for row in evaluate_case(config, case)]
    scenario_summary = _scenario_summary(rows)
    double_ascent = assess_trading_double_ascent(rows)
    gate = evaluate_phase16_gate(config, cases, rows)
    result_path = config.output_directory / "economic_results.csv"
    scenario_path = config.output_directory / "scenario_summary.csv"
    ascent_path = config.output_directory / "trading_double_ascent.csv"
    manifest_path = config.output_directory / "case_manifest.csv"
    plot_path = config.output_directory / "execution_cost_robustness.html"
    summary_path = config.output_directory / "summary.json"
    _write_csv(result_path, rows)
    _write_csv(scenario_path, scenario_summary)
    _write_csv(ascent_path, double_ascent)
    _write_csv(
        manifest_path,
        [{**asdict(case), "backtest_path": str(case.backtest_path)} for case in cases],
    )
    plot_written = _write_plot(plot_path, rows)
    reference_rows = [row for row in rows if row["scenario"] == "phase15_reference"]
    stress_rows = [row for row in rows if row["scenario"] == "stress"]
    summary = {
        "phase": 16,
        "objective": (
            "Measure economic robustness of every Phase 15 primary case under frozen "
            "execution costs"
        ),
        "run_id": datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ"),
        "design": {
            "source_phase": 15,
            "source_summary": str(config.phase15_summary),
            "primary_study": PHASE16_PRIMARY_STUDY,
            "case_selection_used_trading_results": False,
            "positions_and_timestamps_are_frozen": True,
            "position_amounts_are_frozen": True,
            "cost_scenarios": [asdict(scenario) for scenario in config.scenarios],
            "funding_source": "funding_fees exported by native Freqtrade futures backtests",
            "holdout_used": False,
        },
        "phase15_gate": phase15["gate"],
        "gate": gate,
        "scenario_summary": scenario_summary,
        "trading_double_ascent": {
            "assessment_count": len(double_ascent),
            "shape_count_by_scenario": {
                scenario.name: sum(
                    row["trading_double_ascent_shape"]
                    for row in double_ascent
                    if row["scenario"] == scenario.name
                )
                for scenario in config.scenarios
            },
            "useful_count_by_scenario": {
                scenario.name: sum(
                    row["useful_high_complexity_economics"]
                    for row in double_ascent
                    if row["scenario"] == scenario.name
                )
                for scenario in config.scenarios
            },
        },
        "economic_evidence": {
            "reference_positive_case_count": sum(row["net_return"] > 0 for row in reference_rows),
            "stress_positive_case_count": sum(row["net_return"] > 0 for row in stress_rows),
            "reference_positive_with_minimum_trades_count": sum(
                row["net_return"] > 0 and row["minimum_trade_count_met"] for row in reference_rows
            ),
            "stress_positive_with_minimum_trades_count": sum(
                row["net_return"] > 0 and row["minimum_trade_count_met"] for row in stress_rows
            ),
            "interpretation": (
                "Descriptive economic robustness only; predictive OOS metrics remain the primary "
                "evidence for double descent"
            ),
        },
        "limitations": {
            "slippage": "deterministic per-side sensitivity, not order-book replay",
            "market_impact": (
                "not modeled; stress costs are a sensitivity rather than a capacity estimate"
            ),
            "drawdown": (
                "custom scenario drawdown uses realized daily trade PnL, not intratrade "
                "mark-to-market"
            ),
            "frozen_exposure": (
                "position amounts from the reference backtest are preserved across scenarios; "
                "this isolates cost effects but does not resize after alternative PnL paths"
            ),
            "gold": "PAXG is a tokenized gold proxy with a shorter 2025 evaluation interval",
        },
        "artifacts": {
            "summary": str(summary_path),
            "economic_results": str(result_path),
            "scenario_summary": str(scenario_path),
            "trading_double_ascent": str(ascent_path),
            "case_manifest": str(manifest_path),
            "interactive_plot": str(plot_path) if plot_written else None,
        },
    }
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    return summary
