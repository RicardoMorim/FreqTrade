"""Phase 17: native Freqtrade lookahead-bias audit for the frozen Phase 15 design."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from research.double_descent.freqai.Phase15TimeframeStrategy import (
    Phase15TimeframeStrategy,
)
from research.double_descent.phase12 import PHASE12_BASELINES
from research.double_descent.phase15 import (
    PHASE15_ASSETS,
    PHASE15_CONTROL_BASELINES,
    PHASE15_EXPECTED_FULL_CASE_COUNT,
    PHASE15_STRATEGY,
    PHASE15_STUDIES,
)


PHASE17_REQUIRED_SIGNALS = 20
# Freqtrade's CSV exporter uses `total_signals > minimum_trade_amount`.
PHASE17_NATIVE_MINIMUM = PHASE17_REQUIRED_SIGNALS - 1
PHASE17_NATIVE_TARGET = PHASE17_REQUIRED_SIGNALS
PHASE17_HOLDOUT_START = "20260101"
PHASE17_PROTOCOL_VERSION = 2
PHASE17_SOURCE_ONLY_BASELINES = ("zero_return", "historical_mean", "buy_and_hold")
PHASE17_ACTIVE_BASELINES = (
    "market_ols",
    "market_ridge",
    "momentum_1h",
    "momentum_24h",
    "volatility_adjusted_momentum_24h",
)
PHASE17_EXPECTED_NATIVE_CASES = 9
PHASE17_DERIVED_RUNTIME_COLUMNS = (
    "do_predict",
    "enter_long",
    "enter_short",
    "exit_long",
    "exit_short",
)
PHASE17_DIAGNOSTIC_TIMERANGES = {
    "btc": "20250101-20250201",
    "eth": "20250101-20250201",
    "gold": "20250704-20250804",
}


@dataclass(frozen=True)
class Phase17Case:
    case_id: str
    asset: str
    pair: str
    study: str
    study_role: str
    label_period_candles: int
    model_family: str
    model_name: str
    seed: int | None
    target_pn_ratio: float | None
    feature_count: int
    source_config: Path
    native_eligible: bool
    coverage_mode: str


@dataclass(frozen=True)
class Phase17Config:
    data_directory: Path
    phase15_summary: Path = Path("user_data/research_results/double_descent/phase15/summary.json")
    output_directory: Path = Path("user_data/research_results/double_descent/phase17")
    project_root: Path = Path()
    python_executable: str = "python"
    strategy_directory: Path = Path("research/double_descent/freqai")
    model_directory: Path = Path("research/double_descent/freqai")
    models_directory: Path = Path("user_data/models")
    required_signals: int = PHASE17_REQUIRED_SIGNALS
    minimum_trade_amount: int = PHASE17_NATIVE_MINIMUM
    targeted_trade_amount: int = PHASE17_NATIVE_TARGET
    subprocess_timeout_seconds: int = 3_600
    resume: bool = True

    def validate(self) -> dict[str, Any]:
        if not self.phase15_summary.is_file():
            raise FileNotFoundError("Phase 17 requires the completed Phase 15 summary")
        phase15 = json.loads(self.phase15_summary.read_text(encoding="utf-8"))
        if phase15.get("phase") != 15 or phase15.get("gate", {}).get("passed") is not True:
            raise ValueError("Phase 15 did not pass its complete integrity gate")
        if phase15.get("design", {}).get("holdout_used") is not False:
            raise ValueError("Phase 15 does not certify a sealed holdout")
        if self.required_signals != PHASE17_REQUIRED_SIGNALS:
            raise ValueError("Phase 17 freezes the required number of audited signals")
        if self.minimum_trade_amount != PHASE17_NATIVE_MINIMUM:
            raise ValueError("Phase 17 freezes the native minimum trade amount")
        if self.targeted_trade_amount != PHASE17_NATIVE_TARGET:
            raise ValueError("Phase 17 freezes the native targeted trade amount")
        if self.targeted_trade_amount < self.required_signals:
            raise ValueError("targeted trades cannot be smaller than the evidence floor")
        holdout = datetime.strptime(PHASE17_HOLDOUT_START, "%Y%m%d").replace(tzinfo=UTC)
        for timerange in PHASE17_DIAGNOSTIC_TIMERANGES.values():
            end = datetime.strptime(timerange.split("-")[1], "%Y%m%d").replace(tzinfo=UTC)
            if end > holdout:
                raise ValueError("Phase 17 may not enter the sealed holdout")
        return phase15


def _resolve_artifact(config: Phase17Config, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config.project_root / path


def _study(name: str):
    return next(item for item in PHASE15_STUDIES if item.name == name)


def _asset(alias: str):
    return next(item for item in PHASE15_ASSETS if item.alias == alias)


def _rff_case_id(asset: str, study: str, seed: int, ratio: float) -> str:
    return f"{asset}-{study}-rff-s{seed}-pn{ratio:.5f}"


def _baseline_case_id(asset: str, study: str, baseline: str) -> str:
    return f"{asset}-{study}-baseline-{baseline}"


def discover_phase15_cases(config: Phase17Config) -> list[Phase17Case]:
    """Inventory every Phase 15 case without selecting on prediction or PnL."""
    root = config.phase15_summary.parent / "runs"
    cases: list[Phase17Case] = []
    for asset in PHASE15_ASSETS:
        for study in PHASE15_STUDIES:
            cell = root / asset.alias / study.name
            rff_root = cell / "rff" / "runs" / "market_rff"
            for summary_path in sorted(rff_root.glob("seed-*/summary.json")):
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if summary.get("gate", {}).get("passed") is not True:
                    raise ValueError(f"Phase 15 RFF substudy did not pass: {summary_path}")
                seed = int(summary["seed"])
                for result in summary.get("results", []):
                    if result.get("success") is not True:
                        raise ValueError(f"Phase 15 contains a failed case: {summary_path}")
                    ratio = float(result["target_pn_ratio"])
                    source = _resolve_artifact(config, result["artifacts"]["config"])
                    cases.append(
                        Phase17Case(
                            case_id=_rff_case_id(asset.alias, study.name, seed, ratio),
                            asset=asset.alias,
                            pair=asset.pair,
                            study=study.name,
                            study_role=study.role,
                            label_period_candles=study.label_period_candles,
                            model_family="rff",
                            model_name="market_rff",
                            seed=seed,
                            target_pn_ratio=ratio,
                            feature_count=int(result["feature_count"]),
                            source_config=source,
                            native_eligible=True,
                            coverage_mode="native_representative_by_shared_rff_code",
                        )
                    )
            baselines = (
                PHASE12_BASELINES if study.name == "native_15m" else PHASE15_CONTROL_BASELINES
            )
            for baseline in baselines:
                source = cell / "baselines" / "runs" / baseline / "config.json"
                cases.append(
                    Phase17Case(
                        case_id=_baseline_case_id(asset.alias, study.name, baseline),
                        asset=asset.alias,
                        pair=asset.pair,
                        study=study.name,
                        study_role=study.role,
                        label_period_candles=study.label_period_candles,
                        model_family="baseline",
                        model_name=baseline,
                        seed=None,
                        target_pn_ratio=None,
                        feature_count=25,
                        source_config=source,
                        native_eligible=baseline not in PHASE17_SOURCE_ONLY_BASELINES,
                        coverage_mode=(
                            "source_only_insufficient_trades"
                            if baseline in PHASE17_SOURCE_ONLY_BASELINES
                            else "native_representative_by_shared_baseline_branch"
                        ),
                    )
                )
    identifiers = [case.case_id for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Phase 17 discovered duplicate Phase 15 case identifiers")
    return cases


def representative_case_ids() -> tuple[str, ...]:
    ids = [_rff_case_id(asset, "native_15m", 20260810, 0.1) for asset in ("btc", "eth", "gold")]
    ids.append(_rff_case_id("btc", "matched_1h_control", 20260810, 0.1))
    ids.extend(
        _baseline_case_id("btc", "native_15m", baseline) for baseline in PHASE17_ACTIVE_BASELINES
    )
    return tuple(ids)


def select_representative_cases(cases: list[Phase17Case]) -> list[Phase17Case]:
    by_id = {case.case_id: case for case in cases}
    required = representative_case_ids()
    missing = [case_id for case_id in required if case_id not in by_id]
    if missing:
        raise ValueError(f"missing predeclared Phase 17 representatives: {missing}")
    return [by_id[case_id] for case_id in required]


def audit_case_config(case: Phase17Case) -> dict[str, Any]:
    if not case.source_config.is_file():
        return {"case_id": case.case_id, "passed": False, "error": "missing source config"}
    generated = json.loads(case.source_config.read_text(encoding="utf-8"))
    freqai = generated.get("freqai", {})
    features = freqai.get("feature_parameters", {})
    split = freqai.get("data_split_parameters", {})
    parameters = freqai.get("model_training_parameters", {})
    checks = {
        "freqai_enabled": freqai.get("enabled") is True,
        "timeframe_is_15m": generated.get("timeframe") == "15m",
        "pair_matches_cell": generated.get("exchange", {}).get("pair_whitelist") == [case.pair],
        "label_horizon_matches_cell": features.get("label_period_candles")
        == case.label_period_candles,
        "no_shifted_input_candles": features.get("include_shifted_candles") == 0,
        "training_rows_not_shuffled": features.get("shuffle_after_split") is False,
        "data_split_not_shuffled": split.get("shuffle") is False,
        "no_internal_test_holdout": split.get("test_size") == 0,
        "input_dimension_is_25": int(parameters.get("input_feature_count", 25)) == 25,
        "model_family_matches": (
            parameters.get("representation") == "market_rff"
            if case.model_family == "rff"
            else parameters.get("baseline") == case.model_name
        ),
        "feature_count_matches": int(parameters.get("feature_count", -1)) == case.feature_count,
        "float64_training": parameters.get("dtype") == "float64",
    }
    return {
        "case_id": case.case_id,
        "source_config": str(case.source_config),
        "passed": all(checks.values()),
        "checks": checks,
    }


def _market_frame(rows: int = 1_700) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    close = 50_000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.001, rows)))
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=rows, freq="15min", tz="UTC"),
            "open": close * (1.0 + rng.normal(0.0, 0.0002, rows)),
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": rng.lognormal(5.0, 0.2, rows),
        }
    )


def audit_shared_causal_source() -> dict[str, Any]:
    """Test prefix invariance and the quarantined forward target definition."""
    frame = _market_frame()
    full = Phase15TimeframeStrategy.feature_engineering_standard(None, frame.copy(), {})
    feature_columns = [column for column in full if column.startswith("%")]
    mismatches: list[str] = []
    for cut in (1_000, 1_350, len(frame) - 1):
        prefix = Phase15TimeframeStrategy.feature_engineering_standard(
            None, frame.iloc[:cut].copy(), {}
        )
        for column in feature_columns:
            left = full[column].iloc[:cut].to_numpy(dtype=float)
            right = prefix[column].to_numpy(dtype=float)
            if not np.allclose(left, right, rtol=0.0, atol=0.0, equal_nan=True):
                mismatches.append(f"{column}@{cut}")
    target_checks = {}
    for horizon in (1, 4):
        strategy = SimpleNamespace(
            freqai_info={"feature_parameters": {"label_period_candles": horizon}}
        )
        targeted = Phase15TimeframeStrategy.set_freqai_targets(strategy, frame.copy(), {})
        expected = frame["close"].shift(-horizon) / frame["close"] - 1.0
        target_checks[str(horizon)] = bool(
            np.allclose(
                targeted["&-forward_return"].to_numpy(dtype=float),
                expected.to_numpy(dtype=float),
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )
        )
    checks = {
        "exactly_25_market_features": len(feature_columns) == 25,
        "market_features_are_prefix_invariant": not mismatches,
        "forward_target_is_quarantined_to_ampersand_column": all(target_checks.values()),
        "signal_strategy_uses_native_freqai_prediction_contract": PHASE15_STRATEGY
        == "Phase15TimeframeStrategy",
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "feature_count": len(feature_columns),
        "prefix_mismatches": mismatches,
        "target_horizon_checks": target_checks,
        "target_column": "&-forward_return",
        "note": (
            "The forward target is expected to differ in truncated data and is excluded only "
            "when Freqtrade reports it as an indicator; entry and exit mismatches still fail."
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def prepare_phase17(config: Phase17Config) -> dict[str, Any]:
    phase15 = config.validate()
    config.output_directory.mkdir(parents=True, exist_ok=True)
    cases = discover_phase15_cases(config)
    representatives = select_representative_cases(cases)
    config_audits = [audit_case_config(case) for case in cases]
    source_audit = audit_shared_causal_source()
    source_only = [case for case in cases if not case.native_eligible]
    checks = {
        "phase15_gate_passed": phase15["gate"]["passed"] is True,
        "complete_phase15_case_inventory": len(cases) == PHASE15_EXPECTED_FULL_CASE_COUNT,
        "all_phase15_configs_pass_static_audit": all(row["passed"] for row in config_audits),
        "shared_feature_source_is_causal": source_audit["passed"] is True,
        "complete_predeclared_native_representatives": len(representatives)
        == PHASE17_EXPECTED_NATIVE_CASES,
        "only_inactive_baselines_are_source_only": bool(source_only)
        and all(case.model_name in PHASE17_SOURCE_ONLY_BASELINES for case in source_only),
        "diagnostic_periods_end_before_holdout": all(
            timerange.split("-")[1] <= PHASE17_HOLDOUT_START
            for timerange in PHASE17_DIAGNOSTIC_TIMERANGES.values()
        ),
        "trading_results_did_not_select_representatives": True,
    }
    manifest_path = config.output_directory / "case_manifest.csv"
    audit_path = config.output_directory / "config_audit.csv"
    preparation_path = config.output_directory / "preparation.json"
    _write_csv(
        manifest_path,
        [
            {
                **asdict(case),
                "source_config": str(case.source_config),
                "native_representative": case.case_id in representative_case_ids(),
            }
            for case in cases
        ],
    )
    _write_csv(audit_path, config_audits)
    summary = {
        "phase": 17,
        "stage": "preparation",
        "protocol_version": PHASE17_PROTOCOL_VERSION,
        "objective": "Audit the frozen Phase 15 pipeline for lookahead bias",
        "design": {
            "complete_case_inventory": len(cases),
            "native_representative_case_ids": list(representative_case_ids()),
            "native_representative_count": len(representatives),
            "native_required_signals": config.required_signals,
            "native_minimum_trade_amount": config.minimum_trade_amount,
            "native_targeted_trade_amount": config.targeted_trade_amount,
            "diagnostic_timeranges": PHASE17_DIAGNOSTIC_TIMERANGES,
            "source_only_baselines": list(PHASE17_SOURCE_ONLY_BASELINES),
            "source_only_case_count": len(source_only),
            "selection_used_prediction_or_trading_results": False,
            "holdout_used": False,
        },
        "source_audit": source_audit,
        "gate": {"passed": all(checks.values()), "checks": checks},
        "artifacts": {
            "preparation": str(preparation_path),
            "case_manifest": str(manifest_path),
            "config_audit": str(audit_path),
        },
    }
    preparation_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def diagnostic_identifier(case: Phase17Case) -> str:
    digest = hashlib.sha256(case.case_id.encode()).hexdigest()[:12]
    return f"double-descent-phase17-{case.asset}-{digest}"


def build_diagnostic_config(
    config: Phase17Config, case: Phase17Case, case_directory: Path
) -> dict[str, Any]:
    generated = json.loads(case.source_config.read_text(encoding="utf-8"))
    diagnostic = deepcopy(generated)
    diagnostic["entry_pricing"]["price_side"] = "other"
    diagnostic["exit_pricing"]["price_side"] = "other"
    freqai = diagnostic["freqai"]
    freqai["identifier"] = diagnostic_identifier(case)
    freqai["save_backtest_models"] = False
    parameters = freqai["model_training_parameters"]
    metrics_path = (case_directory / "training_diagnostics.jsonl").resolve()
    for key in tuple(parameters):
        if key.endswith("_metrics_path"):
            parameters[key] = str(metrics_path)
        elif key.endswith("_run_id"):
            parameters[key] = f"phase17-{case.case_id}"
    if case.model_family == "rff":
        parameters["phase17_close_cuda_worker_after_predict"] = True
    return diagnostic


def model_name_for_case(case: Phase17Case) -> str:
    return "Phase4CudaRFFRegressor" if case.model_family == "rff" else "Phase12BaselineRegressor"


def build_lookahead_command(
    config: Phase17Config,
    case: Phase17Case,
    diagnostic_config: Path,
    export_csv: Path,
    log_path: Path,
) -> list[str]:
    return [
        config.python_executable,
        "-m",
        "freqtrade",
        "lookahead-analysis",
        "--config",
        str(diagnostic_config),
        "--strategy",
        PHASE15_STRATEGY,
        "--strategy-path",
        str(config.strategy_directory),
        "--freqaimodel",
        model_name_for_case(case),
        "--freqaimodel-path",
        str(config.model_directory),
        "--timerange",
        PHASE17_DIAGNOSTIC_TIMERANGES[case.asset],
        "--timeframe",
        "15m",
        "--pairs",
        case.pair,
        "--datadir",
        str(config.data_directory),
        "--minimum-trade-amount",
        str(config.minimum_trade_amount),
        "--targeted-trade-amount",
        str(config.targeted_trade_amount),
        "--lookahead-analysis-exportfilename",
        str(export_csv),
        "--logfile",
        str(log_path),
        "--no-color",
    ]


def _source_fingerprint(config: Phase17Config, case: Phase17Case) -> str:
    paths = (
        case.source_config,
        config.strategy_directory / "Phase3EffectiveNStrategy.py",
        config.strategy_directory / "Phase4RFFStrategy.py",
        config.strategy_directory / "Phase15TimeframeStrategy.py",
        config.model_directory / "Phase4CudaRFFRegressor.py",
        config.model_directory / "Phase12BaselineRegressor.py",
        config.project_root / "freqtrade/optimize/analysis/lookahead.py",
        config.project_root / "freqtrade/optimize/analysis/lookahead_helpers.py",
    )
    digest = hashlib.sha256(f"phase17:{PHASE17_PROTOCOL_VERSION}".encode())
    for path in paths:
        digest.update(path.resolve().as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"cannot parse boolean value {value!r}")


def parse_lookahead_csv(
    path: Path,
    required_signals: int,
    returncode: int | None,
    timed_out: bool,
) -> dict[str, Any]:
    if not path.is_file():
        return {
            "passed": False,
            "csv_produced": False,
            "error": "Freqtrade produced no lookahead CSV row",
        }
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        return {
            "passed": False,
            "csv_produced": True,
            "error": f"expected one lookahead row, found {len(rows)}",
        }
    row = rows[0]
    indicators = [
        value.strip() for value in (row.get("biased_indicators") or "").split(",") if value.strip()
    ]
    total = int(row["total_signals"])
    entry = int(row["biased_entry_signals"])
    exit_ = int(row["biased_exit_signals"])
    raw_has_bias = _parse_bool(row["has_bias"])
    ignored_targets = [value for value in indicators if value.startswith("&")]
    ignored_runtime = (
        [value for value in indicators if value in PHASE17_DERIVED_RUNTIME_COLUMNS]
        if entry == 0 and exit_ == 0
        else []
    )
    ignored = set(ignored_targets + ignored_runtime)
    actionable = [value for value in indicators if value not in ignored]
    checks = {
        "returncode_zero": returncode == 0,
        "not_timed_out": not timed_out,
        "required_signals_tested": total >= required_signals,
        "no_biased_entries": entry == 0,
        "no_biased_exits": exit_ == 0,
        "no_actionable_biased_indicators": not actionable,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "csv_produced": True,
        "raw_freqtrade_has_bias": raw_has_bias,
        "total_signals": total,
        "biased_entry_signals": entry,
        "biased_exit_signals": exit_,
        "biased_indicators": indicators,
        "ignored_freqai_target_indicators": ignored_targets,
        "ignored_freqai_runtime_indicators": ignored_runtime,
        "actionable_biased_indicators": actionable,
        "freqtrade_false_positive_adjudication_applied": bool(ignored),
    }


def _clean_diagnostic_models(config: Phase17Config, identifier: str) -> None:
    root = (config.project_root / config.models_directory).resolve()
    target = (root / identifier).resolve()
    if target.parent != root or not identifier.startswith("double-descent-phase17-"):
        raise ValueError("refusing to remove a non-Phase17 model directory")
    if target.exists():
        shutil.rmtree(target)


def run_native_case(config: Phase17Config, case: Phase17Case) -> dict[str, Any]:
    case_directory = config.output_directory / "runs" / case.case_id
    case_directory.mkdir(parents=True, exist_ok=True)
    result_path = case_directory / "result.json"
    fingerprint = _source_fingerprint(config, case)
    if config.resume and result_path.is_file():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if cached.get("fingerprint") == fingerprint and cached.get("passed") is True:
            print(f"PHASE17 RECOVERED {case.case_id}", flush=True)
            return cached
    diagnostic_path = case_directory / "config.json"
    export_path = case_directory / "lookahead.csv"
    log_path = case_directory / "freqtrade.log"
    process_log_path = case_directory / "process.log"
    metrics_path = case_directory / "training_diagnostics.jsonl"
    for path in (export_path, log_path, process_log_path, metrics_path):
        path.unlink(missing_ok=True)
    diagnostic = build_diagnostic_config(config, case, case_directory)
    diagnostic_path.write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
    command = build_lookahead_command(
        config, case, diagnostic_path, export_path.resolve(), log_path.resolve()
    )
    print(f"PHASE17 START {case.case_id}", flush=True)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    started = datetime.now(tz=UTC)
    try:
        completed = subprocess.run(
            command,
            cwd=config.project_root,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=config.subprocess_timeout_seconds,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        returncode = None
        stdout = (
            exc.stdout.decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        timed_out = True
    wall_seconds = (datetime.now(tz=UTC) - started).total_seconds()
    process_log_path.write_text(
        f"COMMAND: {subprocess.list2cmdline(command)}\n\n{stdout}\n\nSTDERR:\n{stderr}",
        encoding="utf-8",
    )
    parsed = parse_lookahead_csv(export_path, config.required_signals, returncode, timed_out)
    result = {
        **asdict(case),
        **parsed,
        "source_config": str(case.source_config),
        "fingerprint": fingerprint,
        "identifier": diagnostic["freqai"]["identifier"],
        "diagnostic_timerange": PHASE17_DIAGNOSTIC_TIMERANGES[case.asset],
        "model_adapter": model_name_for_case(case),
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_seconds": wall_seconds,
        "artifacts": {
            "config": str(diagnostic_path),
            "csv": str(export_path),
            "freqtrade_log": str(log_path),
            "process_log": str(process_log_path),
        },
        "stderr_tail": stderr[-2_000:],
    }
    result_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    _clean_diagnostic_models(config, diagnostic["freqai"]["identifier"])
    print(
        f"PHASE17 DONE {case.case_id} passed={result['passed']} "
        f"signals={result.get('total_signals', 0)} seconds={wall_seconds:.1f}",
        flush=True,
    )
    return result


def _coverage_summary(cases: list[Phase17Case]) -> dict[str, Any]:
    source_only = [case for case in cases if not case.native_eligible]
    return {
        "all_phase15_case_count": len(cases),
        "native_eligible_case_count": sum(case.native_eligible for case in cases),
        "native_representative_case_count": PHASE17_EXPECTED_NATIVE_CASES,
        "source_only_case_count": len(source_only),
        "source_only_case_ids": [case.case_id for case in source_only],
        "rff_dimensions_are_source_equivalent": True,
        "rff_seeds_are_source_equivalent": True,
        "assets_share_one_feature_and_signal_implementation": True,
        "target_horizons_are_both_natively_represented": True,
    }


def run_phase17(
    config: Phase17Config,
    selected_case_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    preparation = prepare_phase17(config)
    if preparation["gate"]["passed"] is not True:
        raise ValueError("Phase 17 preparation did not pass")
    cases = discover_phase15_cases(config)
    representatives = select_representative_cases(cases)
    requested = set(selected_case_ids)
    unknown = requested - set(representative_case_ids())
    if unknown:
        raise ValueError(f"unknown or non-representative case ids: {sorted(unknown)}")
    to_run = [case for case in representatives if not requested or case.case_id in requested]
    results = []
    checkpoint_path = config.output_directory / "checkpoint.json"
    for case in to_run:
        results.append(run_native_case(config, case))
        checkpoint_path.write_text(
            json.dumps(
                {
                    "phase": 17,
                    "completed_case_ids": [row["case_id"] for row in results],
                    "passed_case_ids": [row["case_id"] for row in results if row["passed"]],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    # A filtered invocation is deliberately partial and cannot pass the final gate.
    result_by_id = {row["case_id"]: row for row in results}
    if not requested:
        for case in representatives:
            result_path = config.output_directory / "runs" / case.case_id / "result.json"
            if case.case_id not in result_by_id and result_path.is_file():
                result_by_id[case.case_id] = json.loads(result_path.read_text(encoding="utf-8"))
    complete_results = [
        result_by_id[case_id] for case_id in representative_case_ids() if case_id in result_by_id
    ]
    full_invocation = not requested
    checks = {
        "preparation_passed": preparation["gate"]["passed"] is True,
        "complete_native_representative_count": len(complete_results)
        == PHASE17_EXPECTED_NATIVE_CASES,
        "all_native_representatives_passed": len(complete_results) == PHASE17_EXPECTED_NATIVE_CASES
        and all(row["passed"] for row in complete_results),
        "all_required_signals_tested": len(complete_results) == PHASE17_EXPECTED_NATIVE_CASES
        and all(
            int(row.get("total_signals", 0)) >= config.required_signals for row in complete_results
        ),
        "no_biased_entry_or_exit_signals": len(complete_results) == PHASE17_EXPECTED_NATIVE_CASES
        and all(
            int(row.get("biased_entry_signals", -1)) == 0
            and int(row.get("biased_exit_signals", -1)) == 0
            for row in complete_results
        ),
        "no_actionable_biased_indicators": len(complete_results) == PHASE17_EXPECTED_NATIVE_CASES
        and all(not row.get("actionable_biased_indicators") for row in complete_results),
        "all_117_cases_have_declared_coverage": len(cases) == PHASE15_EXPECTED_FULL_CASE_COUNT,
        "holdout_was_not_used": True,
        "full_unfiltered_protocol_executed": full_invocation,
    }
    result_csv = config.output_directory / "native_results.csv"
    summary_path = config.output_directory / "summary.json"
    _write_csv(result_csv, complete_results)
    summary = {
        "phase": 17,
        "stage": "complete" if all(checks.values()) else "partial_or_failed",
        "protocol_version": PHASE17_PROTOCOL_VERSION,
        "objective": "Native Freqtrade lookahead analysis of the frozen Phase 15 pipeline",
        "design": preparation["design"],
        "preparation_gate": preparation["gate"],
        "coverage": _coverage_summary(cases),
        "gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "expected_native_case_count": PHASE17_EXPECTED_NATIVE_CASES,
            "observed_native_case_count": len(complete_results),
        },
        "native_results": complete_results,
        "limitations": {
            "representative_execution": (
                "P/N and seed change numerical size, not the causal source path; all source "
                "configs are audited and representative native runs cover each path."
            ),
            "inactive_baselines": (
                "zero_return, historical_mean, and buy_and_hold do not provide 20 independent "
                "trades for Freqtrade lookahead-analysis and remain source-only audits."
            ),
            "freqai_target": (
                "Raw Freqtrade output is retained. Targets and known FreqAI runtime/signal "
                "columns are adjudicated only when native entry and exit mismatch counts are "
                "both zero; any feature, unknown column, or trade mismatch fails."
            ),
        },
        "artifacts": {
            "summary": str(summary_path),
            "preparation": preparation["artifacts"]["preparation"],
            "case_manifest": preparation["artifacts"]["case_manifest"],
            "config_audit": preparation["artifacts"]["config_audit"],
            "native_results": str(result_csv),
            "checkpoint": str(checkpoint_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary
