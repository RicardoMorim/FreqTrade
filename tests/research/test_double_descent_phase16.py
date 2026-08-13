import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.double_descent.phase10 import PHASE10_ROBUSTNESS_RATIOS
from research.double_descent.phase16 import (
    PHASE16_COST_SCENARIOS,
    PHASE16_EXPECTED_BASELINE_CASES,
    PHASE16_EXPECTED_CASES,
    PHASE16_EXPECTED_RFF_CASES,
    PHASE16_MINIMUM_TRADES,
    Phase16Config,
    _prepare_trade_frame,
    assess_trading_double_ascent,
    decompose_trade_pnl,
    summarize_repriced_trades,
)


def _trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount": [2.0, 1.0],
            "open_rate": [100.0, 120.0],
            "close_rate": [110.0, 100.0],
            "is_short": [False, True],
            "funding_fees": [-1.0, 2.0],
            "profit_abs": [18.58, 21.78],
            "close_date": pd.to_datetime(
                ["2025-01-02T00:00:00Z", "2025-01-03T00:00:00Z"], utc=True
            ),
            "trade_duration": [60, 120],
        }
    )


def _passed_phase15(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "phase": 15,
                "gate": {"passed": True},
                "design": {"holdout_used": False},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_phase16_design_is_frozen_and_includes_stress_costs(tmp_path: Path) -> None:
    summary = _passed_phase15(tmp_path / "phase15.json")
    config = Phase16Config(phase15_summary=summary)

    config.validate()

    assert PHASE16_EXPECTED_RFF_CASES == 63
    assert PHASE16_EXPECTED_BASELINE_CASES == 24
    assert PHASE16_EXPECTED_CASES == 87
    assert PHASE16_MINIMUM_TRADES == 30
    assert [scenario.name for scenario in PHASE16_COST_SCENARIOS] == [
        "price_only",
        "funding_only",
        "optimistic",
        "phase15_reference",
        "conservative",
        "stress",
    ]
    assert PHASE16_COST_SCENARIOS[-1].total_cost_per_side == pytest.approx(0.003)


def test_phase16_rejects_failed_phase15(tmp_path: Path) -> None:
    summary = tmp_path / "phase15.json"
    summary.write_text(
        json.dumps(
            {
                "phase": 15,
                "gate": {"passed": False},
                "design": {"holdout_used": False},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="did not pass"):
        Phase16Config(phase15_summary=summary).validate()


def test_native_reference_exactly_reconstructs_long_and_short_pnl() -> None:
    scenario = next(item for item in PHASE16_COST_SCENARIOS if item.name == "phase15_reference")

    repriced = decompose_trade_pnl(_trades(), scenario)

    np.testing.assert_allclose(repriced["price_pnl_abs"], [20.0, 20.0])
    np.testing.assert_allclose(repriced["funding_pnl_abs"], [-1.0, 2.0])
    np.testing.assert_allclose(repriced["fee_abs"], [0.42, 0.22])
    np.testing.assert_allclose(repriced["net_pnl_abs"], _trades()["profit_abs"])


def test_cost_stress_changes_only_cost_decomposition() -> None:
    price_only = PHASE16_COST_SCENARIOS[0]
    stress = PHASE16_COST_SCENARIOS[-1]

    clean = decompose_trade_pnl(_trades(), price_only)
    stressed = decompose_trade_pnl(_trades(), stress)

    np.testing.assert_allclose(clean["price_pnl_abs"], stressed["price_pnl_abs"])
    assert clean["fee_abs"].sum() == 0
    assert clean["funding_pnl_abs"].sum() == 0
    assert stressed["fee_abs"].sum() > 0
    assert stressed["slippage_abs"].sum() > 0
    assert stressed["net_pnl_abs"].sum() < clean["net_pnl_abs"].sum()


def test_economic_summary_tracks_turnover_funding_and_risk() -> None:
    scenario = PHASE16_COST_SCENARIOS[3]

    result = summarize_repriced_trades(_trades(), scenario, 10_000.0, "20250101-20250201")

    assert result["trade_count"] == 2
    assert result["long_trade_count"] == 1
    assert result["short_trade_count"] == 1
    assert result["funding_pnl_abs"] == 1.0
    assert result["net_profit_abs"] == pytest.approx(40.36)
    assert result["net_return"] == pytest.approx(0.004036)
    assert result["two_sided_turnover_multiple"] == pytest.approx(0.064)
    assert result["minimum_trade_count_met"] is False
    assert math.isfinite(result["daily_sharpe"])


def test_no_trade_baseline_is_a_valid_zero_activity_case(tmp_path: Path) -> None:
    trades = _prepare_trade_frame(pd.DataFrame(), tmp_path / "empty-backtest.zip")
    result = summarize_repriced_trades(
        trades, PHASE16_COST_SCENARIOS[3], 10_000.0, "20250101-20250201"
    )

    assert set(("close_date", "is_short", "funding_fees")) <= set(trades.columns)
    assert result["trade_count"] == 0
    assert result["net_return"] == 0.0
    assert result["two_sided_turnover_multiple"] == 0.0
    assert result["minimum_trade_count_met"] is False


def test_double_ascent_is_assessed_only_on_common_seed_grid() -> None:
    rows = []
    sharpes = {0.1: 0.4, 1.0: -0.2, 1.02: -0.1, 5.0: 0.3, 50.0: 0.8}
    for asset in ("btc", "eth", "gold"):
        for scenario in PHASE16_COST_SCENARIOS:
            for seed in (1, 2, 3):
                for ratio in PHASE10_ROBUSTNESS_RATIOS:
                    rows.append(
                        {
                            "asset": asset,
                            "scenario": scenario.name,
                            "model_family": "rff",
                            "seed": seed,
                            "target_pn_ratio": ratio,
                            "daily_sharpe": sharpes[ratio],
                            "net_return": 0.1 if ratio == 50 else 0.0,
                            "minimum_trade_count_met": True,
                        }
                    )

    assessments = assess_trading_double_ascent(rows)

    assert len(assessments) == 3 * len(PHASE16_COST_SCENARIOS) * 3
    assert all(row["trading_double_ascent_shape"] for row in assessments)
    assert all(row["useful_high_complexity_economics"] for row in assessments)
