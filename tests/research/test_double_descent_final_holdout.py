from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from research.double_descent.final_holdout import (
    FINAL_ASSETS,
    FINAL_DATA_TIMERANGE,
    FINAL_EXPECTED_BASELINE_CASES,
    FINAL_EXPECTED_CASES,
    FINAL_EXPECTED_COST_ROWS,
    FINAL_EXPECTED_ECONOMIC_CASES,
    FINAL_EXPECTED_FITS,
    FINAL_EXPECTED_RFF_CASES,
    FINAL_EXPECTED_WINDOWS,
    FINAL_HOLDOUT_TIMERANGE,
    FinalHoldoutConfig,
    _baseline_names,
    _rff_tasks,
    build_download_command,
    evaluate_final_gate,
    freeze_final_holdout_protocol,
    validate_frozen_protocol,
)
from research.double_descent.phase12 import PHASE12_BASELINES
from research.double_descent.phase15 import PHASE15_CONTROL_BASELINES, PHASE15_STUDIES
from research.double_descent.phase16 import PHASE16_COST_SCENARIOS


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _config(tmp_path: Path) -> FinalHoldoutConfig:
    summaries = {}
    for phase in (15, 16, 17, 18, 20):
        summaries[phase] = _write_json(
            tmp_path / f"phase{phase}.json",
            {
                "phase": phase,
                "gate": {"passed": True},
                "design": {"holdout_used": False},
            },
        )
    summaries[19] = _write_json(
        tmp_path / "phase19.json",
        {
            "phase": 19,
            "gate": {
                "passed": False,
                "checks": {"full_unfiltered_protocol_executed": True},
            },
            "design": {"holdout_used": False},
        },
    )
    references = {
        name: _write_json(tmp_path / f"{name}.json", {"name": name})
        for name in ("phase9", "gamma", "phase10", "phase10_map", "phase11")
    }
    data = tmp_path / "data"
    data.mkdir()
    return FinalHoldoutConfig(
        data_directory=data,
        output_directory=tmp_path / "output",
        protocol_path=tmp_path / "protocol.json",
        report_path=tmp_path / "report.md",
        phase15_summary=summaries[15],
        phase16_summary=summaries[16],
        phase17_summary=summaries[17],
        phase18_summary=summaries[18],
        phase19_summary=summaries[19],
        phase20_summary=summaries[20],
        phase9_summary=references["phase9"],
        frozen_gamma_file=references["gamma"],
        phase10_summary=references["phase10"],
        phase10_map=references["phase10_map"],
        phase11_summary=references["phase11"],
        python_executable=sys.executable,
        cuda_python_executable=sys.executable,
    )


def test_final_design_counts_are_frozen() -> None:
    rff_per_asset = sum(
        sum(len(ratios) for _, ratios in _rff_tasks(study)) for study in PHASE15_STUDIES
    )
    baselines_per_asset = sum(len(_baseline_names(study)) for study in PHASE15_STUDIES)
    assert len(FINAL_ASSETS) * rff_per_asset == FINAL_EXPECTED_RFF_CASES
    assert len(FINAL_ASSETS) * baselines_per_asset == FINAL_EXPECTED_BASELINE_CASES
    assert FINAL_EXPECTED_CASES == 117
    assert FINAL_EXPECTED_FITS == 117 * FINAL_EXPECTED_WINDOWS
    assert FINAL_EXPECTED_COST_ROWS == FINAL_EXPECTED_ECONOMIC_CASES * len(PHASE16_COST_SCENARIOS)
    assert _baseline_names(PHASE15_STUDIES[0]) == PHASE12_BASELINES
    assert _baseline_names(PHASE15_STUDIES[1]) == PHASE15_CONTROL_BASELINES


def test_protocol_accepts_completed_but_strictly_failed_phase19(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(
        "research.double_descent.final_holdout._git_output",
        lambda *arguments: "abc123",
    )
    protocol = freeze_final_holdout_protocol(config)
    assert protocol["holdout_accessed_when_frozen"] is False
    assert protocol["holdout"]["timerange"] == FINAL_HOLDOUT_TIMERANGE
    assert protocol["development_status"]["19"]["gate_passed"] is False
    assert protocol["expected_counts"]["rolling_fits"] == FINAL_EXPECTED_FITS
    assert validate_frozen_protocol(config) == protocol


def test_protocol_detects_reference_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(
        "research.double_descent.final_holdout._git_output",
        lambda *arguments: "abc123",
    )
    freeze_final_holdout_protocol(config)
    config.phase15_summary.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Phase 15"):
        validate_frozen_protocol(config)


def test_download_command_cannot_drift_from_ytd_cutoff(tmp_path: Path) -> None:
    config = _config(tmp_path)
    command = build_download_command(config)
    assert command[command.index("--timerange") + 1] == FINAL_DATA_TIMERANGE
    assert command[command.index("--timeframes") + 1] == "15m"
    assert [asset.pair for asset in FINAL_ASSETS] == [
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        "PAXG/USDT:USDT",
    ]


def test_integrity_gate_does_not_require_positive_performance(tmp_path: Path) -> None:
    generated = _write_json(tmp_path / "config.json", {"timeframe": "15m"})
    log = tmp_path / "run.log"
    log.write_text(f"COMMAND: --timerange {FINAL_HOLDOUT_TIMERANGE}", encoding="utf-8")

    def raw_result() -> dict:
        return {
            "success": True,
            "artifacts": {"config": str(generated), "log": str(log)},
        }

    row = {
        "train_training_window_count": FINAL_EXPECTED_WINDOWS,
        "oos_mse": 1e9,
        "oos_mae": 1e6,
        "oos_r2": -1e9,
        "trading_sharpe": -100.0,
    }
    prediction_tests = [
        {"duplicate_prediction_dates": 0, "all_predictions_accepted": True} for _ in range(114)
    ]
    cost_rows = [
        {
            "scenario": "phase15_reference",
            "native_max_trade_pnl_error": 0.0,
            "native_total_pnl_error": 0.0,
        }
        for _ in range(FINAL_EXPECTED_ECONOMIC_CASES)
    ]
    cost_rows.extend(
        {"scenario": "other"}
        for _ in range(FINAL_EXPECTED_COST_ROWS - FINAL_EXPECTED_ECONOMIC_CASES)
    )
    gate = evaluate_final_gate(
        [raw_result() for _ in range(FINAL_EXPECTED_RFF_CASES)],
        [dict(row) for _ in range(FINAL_EXPECTED_RFF_CASES)],
        [raw_result() for _ in range(FINAL_EXPECTED_BASELINE_CASES)],
        [dict(row) for _ in range(FINAL_EXPECTED_BASELINE_CASES)],
        prediction_tests,
        cost_rows,
    )
    assert gate["passed"] is True
    assert gate["checks"]["performance_excluded_from_integrity_gate"] is True
