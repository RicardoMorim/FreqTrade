import json
from pathlib import Path

from research.double_descent.phase17 import Phase17Case, representative_case_ids
from research.double_descent.phase18 import (
    PHASE18_CONFIGURED_STARTUP,
    PHASE18_MARKET_FEATURES,
    PHASE18_RELATIVE_TOLERANCE,
    PHASE18_STARTUP_CANDLES,
    Phase18Config,
    build_diagnostic_config,
    build_recursive_command,
    parse_recursive_output,
    parse_recursive_table,
)


def _case(source_config: Path, model_family: str = "rff") -> Phase17Case:
    return Phase17Case(
        case_id="btc-native_15m-rff-s20260810-pn0.10000",
        asset="btc",
        pair="BTC/USDT:USDT",
        study="native_15m",
        study_role="primary",
        label_period_candles=1,
        model_family=model_family,
        model_name="market_rff" if model_family == "rff" else "market_ols",
        seed=20260810 if model_family == "rff" else None,
        target_pn_ratio=0.1 if model_family == "rff" else None,
        feature_count=864 if model_family == "rff" else 25,
        source_config=source_config,
        native_eligible=True,
        coverage_mode="native_representative_by_shared_rff_code",
    )


def _config(tmp_path: Path) -> Phase18Config:
    phase15 = tmp_path / "phase15.json"
    phase17 = tmp_path / "phase17.json"
    phase15.write_text(
        json.dumps(
            {
                "phase": 15,
                "gate": {"passed": True},
                "design": {"holdout_used": False},
            }
        ),
        encoding="utf-8",
    )
    phase17.write_text(
        json.dumps(
            {
                "phase": 17,
                "protocol_version": 2,
                "gate": {"passed": True},
                "design": {"holdout_used": False},
            }
        ),
        encoding="utf-8",
    )
    return Phase18Config(
        data_directory=tmp_path,
        phase15_summary=phase15,
        phase17_summary=phase17,
        output_directory=tmp_path / "phase18",
        project_root=tmp_path,
        python_executable="python-test",
    )


def test_phase18_protocol_is_frozen(tmp_path):
    config = _config(tmp_path)
    config.validate()
    assert PHASE18_STARTUP_CANDLES == (199, 399, 499, 801, 999, 1999, 2494)
    assert PHASE18_CONFIGURED_STARTUP == 801
    assert PHASE18_RELATIVE_TOLERANCE == 1e-6
    assert len(PHASE18_MARKET_FEATURES) == 25
    assert len(representative_case_ids()) == 9


def test_recursive_table_parser_handles_exact_nan_and_variance():
    output = """
                 Recursive Analysis
+----------------------------------------------------+
| Indicators |    199 | 801 (from strategy) |    999 |
|------------+--------+---------------------+--------|
|     %-rsi  | 0.078% |              0.001% |      - |
|  unstable  |   nan% |             -0.250% | 0.010% |
+----------------------------------------------------+
"""
    parsed = parse_recursive_table(output)
    assert parsed["%-rsi"][199] == 0.00078
    assert parsed["%-rsi"][801] == 0.00001
    assert parsed["%-rsi"][999] == 0.0
    assert parsed["unstable"][199] is None
    assert parsed["unstable"][801] == -0.0025


def test_no_variance_and_known_freqai_tail_findings_pass():
    log = "\n".join(
        [
            *(
                f"Calculating indicators using startup candle of {value}."
                for value in PHASE18_STARTUP_CANDLES
            ),
            "Start checking for recursive bias",
            "No variance on indicator(s) found due to recursive formula.",
            "Start checking for lookahead bias on indicators only",
            "=> found lookahead in indicator &-forward_return",
            "=> found lookahead in indicator &-forward_return_mean",
            "=> found lookahead in indicator &-forward_return_std",
            "=> found lookahead in indicator do_predict",
        ]
    )
    result = parse_recursive_output("", log, returncode=0, timed_out=False)
    assert result["passed"] is True
    assert result["max_abs_market_feature_variance_at_configured_startup"] == 0.0
    assert result["actionable_indicator_lookahead"] == []


def test_unknown_lookahead_or_recursive_variance_fails():
    output = """
| Indicators | 199 | 801 (from strategy) | 999 |
| %-rsi_14   |  -  |              0.010% |  -  |
"""
    log = "\n".join(
        [
            *(
                f"Calculating indicators using startup candle of {value}."
                for value in PHASE18_STARTUP_CANDLES
            ),
            "Start checking for recursive bias",
            "Start checking for lookahead bias on indicators only",
            "=> found lookahead in indicator suspicious_unknown",
        ]
    )
    result = parse_recursive_output(output, log, returncode=0, timed_out=False)
    assert result["passed"] is False
    assert result["actionable_recursive_indicators"] == ["%-rsi_14"]
    assert result["actionable_indicator_lookahead"] == ["suspicious_unknown"]


def test_diagnostic_config_is_isolated_and_closes_rff_worker(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "backtest_cache": "day",
                "freqai": {
                    "identifier": "phase15-source",
                    "save_backtest_models": True,
                    "model_training_parameters": {
                        "phase4_metrics_path": "old.jsonl",
                        "phase4_run_id": "old-run",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    config = _config(tmp_path)
    case = _case(source)
    diagnostic = build_diagnostic_config(config, case, tmp_path / "case")
    parameters = diagnostic["freqai"]["model_training_parameters"]
    assert diagnostic["backtest_cache"] == "none"
    assert diagnostic["freqai"]["identifier"].startswith("double-descent-phase18-btc-")
    assert diagnostic["freqai"]["save_backtest_models"] is False
    assert parameters["phase17_close_cuda_worker_after_predict"] is True
    assert parameters["phase4_run_id"].startswith("phase18-")
    original = json.loads(source.read_text(encoding="utf-8"))
    assert original["freqai"]["identifier"] == "phase15-source"


def test_recursive_command_uses_native_cli_and_complete_grid(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    config = _config(tmp_path)
    case = _case(source, model_family="baseline")
    command = build_recursive_command(
        config, case, tmp_path / "diagnostic.json", tmp_path / "freqtrade.log"
    )
    assert command[:4] == ["python-test", "-m", "freqtrade", "recursive-analysis"]
    assert command[command.index("--freqaimodel") + 1] == "Phase12BaselineRegressor"
    startup_index = command.index("--startup-candle") + 1
    logfile_index = command.index("--logfile")
    assert command[startup_index:logfile_index] == [str(value) for value in PHASE18_STARTUP_CANDLES]
