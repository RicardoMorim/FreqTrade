from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from research.double_descent.freqai.Phase12BaselineRegressor import (
    Phase12BaselineRegressor,
    _BaselinePredictor,
)
from research.double_descent.freqai.Phase15TimeframeStrategy import (
    Phase15TimeframeStrategy,
)
from research.double_descent.phase4 import (
    Phase4Config,
    _load_evaluation_market_data,
    build_freqtrade_config,
)
from research.double_descent.phase15 import (
    PHASE15_ASSETS,
    PHASE15_BENCHMARK_RATIOS,
    PHASE15_CONTROL_BASELINES,
    PHASE15_CONTROL_RATIOS,
    PHASE15_EFFECTIVE_N_TOLERANCE,
    PHASE15_EXPECTED_FULL_CASE_COUNT,
    PHASE15_EXPECTED_FULL_FIT_COUNT,
    PHASE15_EXPECTED_N,
    PHASE15_STUDIES,
    Phase15Config,
    _baseline_names,
    _phase10_config,
    _phase12_config,
    _rff_tasks,
    build_download_command,
    selected_assets,
)


def _config(tmp_path: Path, **overrides) -> Phase15Config:
    values = {
        "data_directory": tmp_path,
        "python_executable": str(tmp_path / "python.exe"),
        "cuda_python_executable": str(tmp_path / "cuda.exe"),
    }
    values.update(overrides)
    return Phase15Config(**values)


def _market_frame(rows: int = 1_000) -> pd.DataFrame:
    rng = np.random.default_rng(15)
    close = 2_000 * np.exp(np.cumsum(rng.normal(0, 0.001, rows)))
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=rows, freq="15min", tz="UTC"),
            "open": close * (1 + rng.normal(0, 0.0002, rows)),
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": rng.lognormal(5, 0.2, rows),
        }
    )


def test_phase15_design_includes_gold_and_large_predictor_map() -> None:
    assert [asset.alias for asset in PHASE15_ASSETS] == ["btc", "eth", "gold"]
    assert PHASE15_ASSETS[-1].pair == "PAXG/USDT:USDT"
    assert PHASE15_ASSETS[-1].asset_class == "tokenized_gold_proxy"
    assert PHASE15_ASSETS[-1].timerange == "20250704-20260101"
    assert PHASE15_ASSETS[-1].expected_training_windows == 7
    assert PHASE15_ASSETS[-1].comparison_scope == "coverage_limited_diagnostic"
    assert PHASE15_EXPECTED_N == {"native_15m": 8_639, "matched_1h_control": 8_636}
    assert PHASE15_EXPECTED_FULL_CASE_COUNT == 117
    assert PHASE15_EXPECTED_FULL_FIT_COUNT == 1_287
    assert round(50 * PHASE15_EXPECTED_N["native_15m"]) == 431_950


def test_download_command_requests_every_asset_and_futures_candle_type(tmp_path: Path) -> None:
    command = build_download_command(_config(tmp_path))

    pair_index = command.index("--pairs")
    timeframe_index = command.index("--timeframes")
    assert command[pair_index + 1 : timeframe_index] == [
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        "PAXG/USDT:USDT",
    ]
    candle_index = command.index("--candle-types")
    assert command[candle_index + 1 : candle_index + 4] == [
        "futures",
        "funding_rate",
        "mark",
    ]


def test_primary_and_control_case_counts_are_predeclared(tmp_path: Path) -> None:
    config = _config(tmp_path)
    primary, control = PHASE15_STUDIES

    assert sum(len(ratios) for _, ratios in _rff_tasks(config, primary)) == 21
    assert sum(len(ratios) for _, ratios in _rff_tasks(config, control)) == 5
    assert _baseline_names(config, primary) == config.baselines
    assert _baseline_names(config, control) == PHASE15_CONTROL_BASELINES
    assert PHASE15_BENCHMARK_RATIOS == (0.1, 1.0, 50.0)
    assert PHASE15_CONTROL_RATIOS == (0.1, 1.0, 1.02, 5.0, 50.0)


def test_full_design_cannot_drop_gold_after_results_are_seen(tmp_path: Path) -> None:
    config = _config(tmp_path, asset_aliases=("btc", "eth"))

    try:
        config._validate_design()
    except ValueError as exc:
        assert "assets" in str(exc)
    else:
        raise AssertionError("Phase 15 accepted a full design without gold")


def test_selected_assets_preserve_declared_order(tmp_path: Path) -> None:
    assert [asset.alias for asset in selected_assets(_config(tmp_path))] == [
        "btc",
        "eth",
        "gold",
    ]


def test_gold_engines_preserve_limited_period_and_target_scaling(tmp_path: Path) -> None:
    config = _config(tmp_path)
    gold = PHASE15_ASSETS[-1]
    native, control = PHASE15_STUDIES

    rff = _phase10_config(config, gold, native, 8_639, tmp_path / "rff")
    baseline = _phase12_config(config, gold, control, 8_636, tmp_path / "baseline")

    assert rff.timerange == gold.timerange
    assert rff.minimum_training_windows == 6
    assert rff.effective_n_tolerance == PHASE15_EFFECTIVE_N_TOLERANCE
    assert rff.label_period_candles == 1
    assert baseline.timerange == gold.timerange
    assert baseline.minimum_training_windows == 6
    assert baseline.effective_n_tolerance == PHASE15_EFFECTIVE_N_TOLERANCE
    assert baseline.label_period_candles == 4
    assert baseline.momentum_24h_scale == 24.0


def test_15m_strategy_has_25_causal_clock_time_matched_features() -> None:
    original = _market_frame()
    changed = original.copy()
    changed.loc[len(changed) - 1, ["open", "high", "low", "close", "volume"]] *= 1.4

    original_features = Phase15TimeframeStrategy.feature_engineering_standard(
        None, original.copy(), {}
    )
    changed_features = Phase15TimeframeStrategy.feature_engineering_standard(
        None, changed.copy(), {}
    )
    columns = [column for column in original_features if column.startswith("%")]

    assert len(columns) == 25
    assert "%-return_1h" in columns
    assert Phase15TimeframeStrategy.timeframe == "15m"
    assert Phase15TimeframeStrategy.startup_candle_count == 801
    pd.testing.assert_frame_equal(
        original_features.loc[: len(original) - 2, columns],
        changed_features.loc[: len(original) - 2, columns],
    )


def test_target_period_distinguishes_native_and_matched_one_hour_control() -> None:
    frame = _market_frame(20)
    native = SimpleNamespace(freqai_info={"feature_parameters": {"label_period_candles": 1}})
    matched = SimpleNamespace(freqai_info={"feature_parameters": {"label_period_candles": 4}})

    native_target = Phase15TimeframeStrategy.set_freqai_targets(native, frame.copy(), {})
    matched_target = Phase15TimeframeStrategy.set_freqai_targets(matched, frame.copy(), {})

    expected_native = frame["close"].shift(-1) / frame["close"] - 1
    expected_matched = frame["close"].shift(-4) / frame["close"] - 1
    np.testing.assert_allclose(
        native_target["&-forward_return"].iloc[:-1], expected_native.iloc[:-1]
    )
    np.testing.assert_allclose(
        matched_target["&-forward_return"].iloc[:-4], expected_matched.iloc[:-4]
    )


def test_generic_phase4_config_records_15m_label_horizon(tmp_path: Path) -> None:
    config = Phase4Config(
        data_directory=tmp_path,
        cuda_python_executable=str(tmp_path / "cuda.exe"),
        timeframe="15m",
        effective_n=8_636,
        ratios=(1.0,),
        allow_external_design=True,
        label_period_candles=4,
        indicator_periods_candles=(56,),
        strategy_name="Phase15TimeframeStrategy",
    )
    config._validate_design()
    generated = build_freqtrade_config(
        config,
        8_636,
        "phase15-test",
        tmp_path / "metrics.jsonl",
        "test-run",
    )

    assert generated["timeframe"] == "15m"
    assert generated["freqai"]["feature_parameters"]["label_period_candles"] == 4
    assert generated["freqai"]["feature_parameters"]["indicator_periods_candles"] == [56]


def test_native_target_keeps_one_hour_momentum_comparator(monkeypatch, tmp_path: Path) -> None:
    frame = _market_frame(12)
    monkeypatch.setattr("research.double_descent.phase4.load_pair_history", lambda **_: frame)
    config = Phase4Config(
        data_directory=tmp_path,
        cuda_python_executable=str(tmp_path / "cuda.exe"),
        timeframe="15m",
        timerange="20250101-20250102",
        effective_n=8_639,
        ratios=(1.0,),
        allow_external_design=True,
        label_period_candles=1,
    )

    market = _load_evaluation_market_data(config)

    expected_target = frame["close"].shift(-1) / frame["close"] - 1
    expected_momentum = frame["close"].pct_change(4)
    np.testing.assert_allclose(
        market["realized_forward_return"].iloc[:-1], expected_target.iloc[:-1]
    )
    np.testing.assert_allclose(
        market["momentum_1h_prediction"].iloc[4:], expected_momentum.iloc[4:]
    )


def test_native_15m_momentum_scales_daily_return_to_one_bar() -> None:
    raw = pd.DataFrame({"%-return_24h": [0.096, -0.048]})
    model = _BaselinePredictor("momentum_24h", momentum_24h_scale=96.0)

    prediction = Phase12BaselineRegressor._raw_momentum_prediction("momentum_24h", raw, model)

    np.testing.assert_allclose(prediction, [0.001, -0.0005])
