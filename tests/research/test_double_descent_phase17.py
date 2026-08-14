import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.double_descent.freqai.Phase4CudaRFFRegressor import _CudaRFFPredictor
from research.double_descent.phase17 import (
    PHASE17_EXPECTED_NATIVE_CASES,
    PHASE17_NATIVE_MINIMUM,
    PHASE17_NATIVE_TARGET,
    PHASE17_REQUIRED_SIGNALS,
    Phase17Case,
    Phase17Config,
    audit_case_config,
    audit_shared_causal_source,
    build_diagnostic_config,
    build_lookahead_command,
    parse_lookahead_csv,
    representative_case_ids,
)


def _case(path: Path, **overrides) -> Phase17Case:
    values = {
        "case_id": "btc-native_15m-rff-s20260810-pn0.10000",
        "asset": "btc",
        "pair": "BTC/USDT:USDT",
        "study": "native_15m",
        "study_role": "primary",
        "label_period_candles": 1,
        "model_family": "rff",
        "model_name": "market_rff",
        "seed": 20260810,
        "target_pn_ratio": 0.1,
        "feature_count": 864,
        "source_config": path,
        "native_eligible": True,
        "coverage_mode": "native_representative_by_shared_rff_code",
    }
    values.update(overrides)
    return Phase17Case(**values)


def _source_config(path: Path) -> Path:
    payload = {
        "timeframe": "15m",
        "exchange": {"pair_whitelist": ["BTC/USDT:USDT"]},
        "entry_pricing": {"price_side": "same"},
        "exit_pricing": {"price_side": "same"},
        "freqai": {
            "enabled": True,
            "identifier": "phase15-original",
            "save_backtest_models": False,
            "feature_parameters": {
                "label_period_candles": 1,
                "include_shifted_candles": 0,
                "shuffle_after_split": False,
            },
            "data_split_parameters": {"test_size": 0, "shuffle": False},
            "model_training_parameters": {
                "feature_count": 864,
                "dtype": "float64",
                "representation": "market_rff",
                "phase4_metrics_path": "old.jsonl",
                "phase4_run_id": "old",
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _config(tmp_path: Path) -> Phase17Config:
    return Phase17Config(
        data_directory=tmp_path / "data",
        output_directory=tmp_path / "phase17",
        project_root=tmp_path,
        python_executable="python.exe",
        strategy_directory=Path("research/double_descent/freqai"),
        model_directory=Path("research/double_descent/freqai"),
    )


def test_phase17_native_protocol_is_frozen_and_complete() -> None:
    assert PHASE17_REQUIRED_SIGNALS == 20
    assert PHASE17_NATIVE_MINIMUM == 19
    assert PHASE17_NATIVE_TARGET == 20
    assert len(representative_case_ids()) == PHASE17_EXPECTED_NATIVE_CASES == 9
    assert len(set(representative_case_ids())) == PHASE17_EXPECTED_NATIVE_CASES


def test_shared_phase15_features_are_prefix_causal() -> None:
    audit = audit_shared_causal_source()

    assert audit["passed"] is True
    assert audit["feature_count"] == 25
    assert audit["prefix_mismatches"] == []
    assert audit["target_horizon_checks"] == {"1": True, "4": True}


def test_case_config_audit_rejects_shifted_inputs(tmp_path: Path) -> None:
    source = _source_config(tmp_path / "source.json")
    case = _case(source)
    assert audit_case_config(case)["passed"] is True

    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["freqai"]["feature_parameters"]["include_shifted_candles"] = 1
    source.write_text(json.dumps(payload), encoding="utf-8")

    audit = audit_case_config(case)
    assert audit["passed"] is False
    assert audit["checks"]["no_shifted_input_candles"] is False


def test_diagnostic_config_is_isolated_and_does_not_mutate_source(tmp_path: Path) -> None:
    source = _source_config(tmp_path / "source.json")
    original = source.read_text(encoding="utf-8")
    case = _case(source)
    diagnostic = build_diagnostic_config(_config(tmp_path), case, tmp_path / "case")

    assert source.read_text(encoding="utf-8") == original
    assert diagnostic["entry_pricing"]["price_side"] == "other"
    assert diagnostic["exit_pricing"]["price_side"] == "other"
    assert diagnostic["freqai"]["identifier"].startswith("double-descent-phase17-")
    parameters = diagnostic["freqai"]["model_training_parameters"]
    assert parameters["phase4_metrics_path"].endswith("training_diagnostics.jsonl")
    assert parameters["phase4_run_id"].startswith("phase17-")
    assert parameters["phase17_close_cuda_worker_after_predict"] is True


def test_phase17_rff_predictor_closes_worker_after_sliced_inference() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.closed = False

        def predict(self, inputs):
            return np.zeros(len(inputs))

        def close(self) -> None:
            self.closed = True

    client = FakeClient()
    predictor = _CudaRFFPredictor(client, close_after_predict=True)

    prediction = predictor.predict(pd.DataFrame({"feature": [1.0, 2.0]}))

    np.testing.assert_array_equal(prediction, [0.0, 0.0])
    assert client.closed is True


def test_lookahead_command_uses_native_tool_and_frozen_signal_floor(tmp_path: Path) -> None:
    source = _source_config(tmp_path / "source.json")
    config = _config(tmp_path)
    command = build_lookahead_command(
        config,
        _case(source),
        tmp_path / "config.json",
        tmp_path / "lookahead.csv",
        tmp_path / "freqtrade.log",
    )

    assert command[1:4] == ["-m", "freqtrade", "lookahead-analysis"]
    assert command[command.index("--minimum-trade-amount") + 1] == "19"
    assert command[command.index("--targeted-trade-amount") + 1] == "20"
    assert command[command.index("--timerange") + 1] == "20250101-20250201"
    assert command[command.index("--freqaimodel") + 1] == "Phase4CudaRFFRegressor"


def _write_result(path: Path, **overrides) -> None:
    row = {
        "filename": "Phase15TimeframeStrategy.py",
        "strategy": "Phase15TimeframeStrategy",
        "has_bias": "False",
        "total_signals": "20",
        "biased_entry_signals": "0",
        "biased_exit_signals": "0",
        "biased_indicators": "",
    }
    row.update(overrides)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def test_freqai_target_indicator_is_ignored_but_raw_result_is_retained(tmp_path: Path) -> None:
    result_path = tmp_path / "lookahead.csv"
    _write_result(
        result_path,
        has_bias="True",
        biased_indicators="&-forward_return",
    )

    result = parse_lookahead_csv(result_path, 20, 0, False)

    assert result["passed"] is True
    assert result["raw_freqtrade_has_bias"] is True
    assert result["ignored_freqai_target_indicators"] == ["&-forward_return"]
    assert result["actionable_biased_indicators"] == []


def test_freqai_runtime_columns_are_ignored_only_when_trades_match(tmp_path: Path) -> None:
    result_path = tmp_path / "lookahead.csv"
    _write_result(
        result_path,
        has_bias="True",
        biased_indicators="do_predict,enter_long,exit_short",
    )

    result = parse_lookahead_csv(result_path, 20, 0, False)

    assert result["passed"] is True
    assert result["ignored_freqai_runtime_indicators"] == [
        "do_predict",
        "enter_long",
        "exit_short",
    ]
    assert result["actionable_biased_indicators"] == []


def test_entry_mismatch_or_non_target_indicator_fails(tmp_path: Path) -> None:
    result_path = tmp_path / "lookahead.csv"
    _write_result(
        result_path,
        has_bias="True",
        biased_entry_signals="1",
        biased_indicators="%-return_1h,&-forward_return",
    )

    result = parse_lookahead_csv(result_path, 20, 0, False)

    assert result["passed"] is False
    assert result["checks"]["no_biased_entries"] is False
    assert result["actionable_biased_indicators"] == ["%-return_1h"]
