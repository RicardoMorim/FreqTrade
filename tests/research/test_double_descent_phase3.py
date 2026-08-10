from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from research.double_descent.freqai.Phase3EffectiveNStrategy import (
    Phase3EffectiveNStrategy,
)
from research.double_descent.phase3 import (
    Phase3Config,
    aggregate_measurements,
    build_freqtrade_config,
    evaluate_phase3_gate,
)


def _measurement(period: int, start_hour: int, effective_n: int) -> dict:
    return {
        "train_period_days": period,
        "pair": "BTC/USDT:USDT",
        "raw_start": f"2025-01-{start_hour + 1:02d}T00:00:00+00:00",
        "raw_end": f"2025-02-{start_hour + 1:02d}T00:00:00+00:00",
        "raw_rows": effective_n + 73,
        "filtered_rows_before_split": effective_n,
        "final_train_rows": effective_n,
        "final_test_rows": 0,
        "pipeline_removed_rows": 0,
        "model_feature_count": 25,
        "nonfinite_train_feature_values": 0,
        "nonfinite_train_label_values": 0,
    }


def test_freqtrade_config_freezes_sample_count_controls(tmp_path: Path) -> None:
    config = Phase3Config(data_directory=tmp_path)
    generated = build_freqtrade_config(
        config,
        train_period_days=90,
        identifier="phase3-test",
        metrics_path=tmp_path / "measurements.jsonl",
        run_id="test-run",
    )

    freqai = generated["freqai"]
    features = freqai["feature_parameters"]
    split = freqai["data_split_parameters"]
    assert freqai["train_period_days"] == 90
    assert generated["entry_pricing"]["use_order_book"] is True
    assert generated["exit_pricing"]["use_order_book"] is True
    assert split == {"test_size": 0, "shuffle": False}
    assert features["weight_factor"] == 0
    assert features["include_shifted_candles"] == 0
    assert features["principal_component_analysis"] is False
    assert features["use_SVM_to_remove_outliers"] is False
    assert features["DI_threshold"] == 0
    assert features["label_period_candles"] == 1


def test_market_features_are_causal_and_only_the_target_uses_the_next_candle() -> None:
    rng = np.random.default_rng(42)
    rows = 300
    close = 50_000 * np.exp(np.cumsum(rng.normal(0, 0.002, rows)))
    dataframe = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC"),
            "open": close * (1 + rng.normal(0, 0.0005, rows)),
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": rng.lognormal(4, 0.3, rows),
        }
    )
    changed_future = dataframe.copy()
    changed_future.loc[rows - 1, ["open", "high", "low", "close", "volume"]] *= 1.5

    original_features = Phase3EffectiveNStrategy.feature_engineering_standard(
        None, dataframe.copy(), {}
    )
    changed_features = Phase3EffectiveNStrategy.feature_engineering_standard(
        None, changed_future.copy(), {}
    )
    feature_columns = [column for column in original_features if column.startswith("%")]
    pd.testing.assert_frame_equal(
        original_features.loc[: rows - 2, feature_columns],
        changed_features.loc[: rows - 2, feature_columns],
    )

    strategy = SimpleNamespace(freqai_info={"feature_parameters": {"label_period_candles": 1}})
    original_target = Phase3EffectiveNStrategy.set_freqai_targets(strategy, dataframe.copy(), {})
    changed_target = Phase3EffectiveNStrategy.set_freqai_targets(
        strategy, changed_future.copy(), {}
    )
    assert (
        original_target["&-forward_return"].iloc[-2] != changed_target["&-forward_return"].iloc[-2]
    )


def test_aggregation_builds_grid_from_observed_median_n() -> None:
    measurements = [_measurement(30, 0, 99), _measurement(30, 1, 101)]

    aggregate = aggregate_measurements(measurements, ratios=(0.5, 1.0, 2.0))

    assert len(aggregate) == 1
    assert aggregate[0]["effective_n_median"] == 100
    assert [row["feature_count"] for row in aggregate[0]["suggested_p_grid"]] == [
        50,
        100,
        200,
    ]


def test_phase3_gate_requires_real_measurements_for_every_window(tmp_path: Path) -> None:
    config = Phase3Config(
        data_directory=tmp_path,
        train_periods_days=(30,),
        minimum_windows_per_period=2,
    )
    measurements = [_measurement(30, 0, 648), _measurement(30, 1, 647)]
    data_audit = {"passed": True}
    successful_runs = [{"train_period_days": 30, "success": True}]

    passed = evaluate_phase3_gate(config, data_audit, successful_runs, measurements)
    failed = evaluate_phase3_gate(config, data_audit, successful_runs, [])

    assert passed["passed"] is True
    assert failed["passed"] is False
    assert failed["checks"]["minimum_windows_observed"] is False
