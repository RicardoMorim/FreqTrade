import math
from pathlib import Path

import numpy as np
import pandas as pd

from research.double_descent.phase13 import (
    FDR_ALPHA,
    HAC_LAG_HOURS,
    JOINT_REGIMES,
    MARGINAL_REGIMES,
    TREND_LOOKBACK_HOURS,
    TREND_SCORE_THRESHOLD,
    VOLATILITY_REFERENCE_HOURS,
    Phase13Config,
    benjamini_hochberg,
    causal_regime_features,
    hac_mean_test,
)


def _synthetic_market(size: int = 500) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=size, freq="1h", tz="UTC")
    returns = np.resize(np.array([0.002, 0.001, -0.0005, 0.0015]), size)
    close = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame({"date": dates, "close": close})


def test_phase13_design_is_frozen() -> None:
    assert TREND_LOOKBACK_HOURS == 720
    assert VOLATILITY_REFERENCE_HOURS == 1_440
    assert TREND_SCORE_THRESHOLD == 0.5
    assert HAC_LAG_HOURS == 24
    assert FDR_ALPHA == 0.05
    assert len(MARGINAL_REGIMES) == 5
    assert len(JOINT_REGIMES) == 6


def test_causal_regimes_do_not_change_when_future_prices_change() -> None:
    market = _synthetic_market()
    original = causal_regime_features(
        market,
        trend_lookback_hours=24,
        volatility_reference_hours=48,
        trend_score_threshold=0.5,
    )
    changed = market.copy()
    changed.loc[400:, "close"] *= np.linspace(1.0, 5.0, len(changed) - 400)
    revised = causal_regime_features(
        changed,
        trend_lookback_hours=24,
        volatility_reference_hours=48,
        trend_score_threshold=0.5,
    )

    columns = ["trend_score", "volatility_reference", "trend_regime", "volatility_regime"]
    pd.testing.assert_frame_equal(original.loc[:399, columns], revised.loc[:399, columns])


def test_last_forward_target_requires_a_next_in_scope_candle() -> None:
    market = _synthetic_market(100)
    regimes = causal_regime_features(
        market,
        trend_lookback_hours=12,
        volatility_reference_hours=24,
        trend_score_threshold=0.5,
    )

    assert np.isfinite(regimes["realized_forward_return"].iloc[-2])
    assert np.isnan(regimes["realized_forward_return"].iloc[-1])


def test_regime_labels_cover_bull_bear_sideways_and_volatility_states() -> None:
    up = np.full(120, 0.01)
    flat = np.resize(np.array([0.002, -0.002]), 120)
    down = np.full(120, -0.01)
    high_vol = np.resize(np.array([0.04, -0.04]), 140)
    returns = np.concatenate([up, flat, down, high_vol])
    market = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=len(returns), freq="1h", tz="UTC"),
            "close": 100.0 * np.exp(np.cumsum(returns)),
        }
    )
    regimes = causal_regime_features(
        market,
        trend_lookback_hours=24,
        volatility_reference_hours=48,
        trend_score_threshold=0.5,
    )

    assert {"bull", "bear", "sideways"}.issubset(set(regimes["trend_regime"].dropna()))
    assert {"low_volatility", "high_volatility"}.issubset(
        set(regimes["volatility_regime"].dropna())
    )


def test_hac_mean_test_handles_constant_and_directional_loss_differences() -> None:
    constant = hac_mean_test(np.zeros(100), maximum_lag=5)
    worse = hac_mean_test(np.full(100, 0.01), maximum_lag=5)

    assert constant == {"mean": 0.0, "standard_error": 0.0, "t_stat": 0.0, "p": 1.0}
    assert worse["mean"] > 0
    assert worse["p"] == 0.0
    assert math.isinf(worse["t_stat"])


def test_benjamini_hochberg_is_monotone_in_sorted_p_values() -> None:
    adjusted = benjamini_hochberg([0.001, 0.01, 0.2, np.nan])

    np.testing.assert_allclose(adjusted[:3], [0.003, 0.015, 0.2])
    assert np.isnan(adjusted[3])


def test_full_run_rejects_post_hoc_regime_threshold(tmp_path: Path) -> None:
    config = Phase13Config(
        data_directory=tmp_path,
        trend_score_threshold=0.7,
    )

    try:
        config._validate_design()
    except ValueError as exc:
        assert "trend threshold" in str(exc)
    else:
        raise AssertionError("Phase 13 accepted a post-hoc trend threshold")
