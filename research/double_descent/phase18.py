"""Phase 18: native recursive-indicator audit for the frozen Phase 15 pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from freqtrade.configuration import TimeRange
from freqtrade.data.history import load_pair_history
from freqtrade.enums import CandleType
from research.double_descent.phase17 import (
    PHASE17_EXPECTED_NATIVE_CASES,
    PHASE17_HOLDOUT_START,
    PHASE17_PROTOCOL_VERSION,
    Phase17Case,
    Phase17Config,
    audit_case_config,
    audit_shared_causal_source,
    discover_phase15_cases,
    model_name_for_case,
    representative_case_ids,
    select_representative_cases,
)


PHASE18_PROTOCOL_VERSION = 1
PHASE18_STRATEGY = "Phase15TimeframeStrategy"
PHASE18_TIMEFRAME = "15m"
PHASE18_STARTUP_CANDLES = (199, 399, 499, 801, 999, 1999, 2494)
PHASE18_CONFIGURED_STARTUP = 801
PHASE18_MINIMUM_BENCHMARK_CANDLES = 5_000
PHASE18_RELATIVE_TOLERANCE = 1e-6
PHASE18_DIAGNOSTIC_TIMERANGES = {
    "btc": "20250101-20250301",
    "eth": "20250101-20250301",
    "gold": "20250704-20250904",
}
PHASE18_KNOWN_FREQAI_LOOKAHEAD_COLUMNS = ("do_predict",)
PHASE18_MARKET_FEATURES = (
    "%-return_1h",
    "%-return_3h",
    "%-return_6h",
    "%-return_12h",
    "%-return_24h",
    "%-return_72h",
    "%-return_168h",
    "%-volatility_6h",
    "%-volatility_24h",
    "%-volatility_72h",
    "%-volume_change_1h",
    "%-relative_volume_24h",
    "%-candle_range",
    "%-candle_body",
    "%-rsi_14",
    "%-adx_14",
    "%-atr_14_normalized",
    "%-sma_distance_12h",
    "%-sma_distance_24h",
    "%-sma_distance_72h",
    "%-return_volatility_24h",
    "%-hour_sin",
    "%-hour_cos",
    "%-weekday_sin",
    "%-weekday_cos",
)


@dataclass(frozen=True)
class Phase18Config:
    data_directory: Path
    phase15_summary: Path = Path("user_data/research_results/double_descent/phase15/summary.json")
    phase17_summary: Path = Path("user_data/research_results/double_descent/phase17/summary.json")
    output_directory: Path = Path("user_data/research_results/double_descent/phase18")
    project_root: Path = Path()
    python_executable: str = "python"
    strategy_directory: Path = Path("research/double_descent/freqai")
    model_directory: Path = Path("research/double_descent/freqai")
    models_directory: Path = Path("user_data/models")
    subprocess_timeout_seconds: int = 3_600
    resume: bool = True

    def validate(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self.phase15_summary.is_file():
            raise FileNotFoundError("Phase 18 requires the completed Phase 15 summary")
        if not self.phase17_summary.is_file():
            raise FileNotFoundError("Phase 18 requires the completed Phase 17 summary")
        phase15 = json.loads(self.phase15_summary.read_text(encoding="utf-8"))
        phase17 = json.loads(self.phase17_summary.read_text(encoding="utf-8"))
        if phase15.get("phase") != 15 or phase15.get("gate", {}).get("passed") is not True:
            raise ValueError("Phase 15 did not pass its complete integrity gate")
        if phase17.get("phase") != 17 or phase17.get("gate", {}).get("passed") is not True:
            raise ValueError("Phase 17 did not pass its complete integrity gate")
        if phase15.get("design", {}).get("holdout_used") is not False:
            raise ValueError("Phase 15 does not certify a sealed holdout")
        if phase17.get("design", {}).get("holdout_used") is not False:
            raise ValueError("Phase 17 does not certify a sealed holdout")
        if tuple(PHASE18_STARTUP_CANDLES) != (199, 399, 499, 801, 999, 1999, 2494):
            raise ValueError("Phase 18 startup-candle grid is frozen")
        if PHASE18_CONFIGURED_STARTUP not in PHASE18_STARTUP_CANDLES:
            raise ValueError("configured strategy startup must be in the audit grid")
        if len(PHASE18_MARKET_FEATURES) != 25:
            raise ValueError("Phase 18 requires all 25 frozen market features")
        holdout = datetime.strptime(PHASE17_HOLDOUT_START, "%Y%m%d").replace(tzinfo=UTC)
        for timerange in PHASE18_DIAGNOSTIC_TIMERANGES.values():
            _, end = _parse_timerange(timerange)
            if end > holdout:
                raise ValueError("Phase 18 may not enter the sealed holdout")
        return phase15, phase17


def _phase17_config(config: Phase18Config) -> Phase17Config:
    return Phase17Config(
        data_directory=config.data_directory,
        phase15_summary=config.phase15_summary,
        project_root=config.project_root,
        python_executable=config.python_executable,
        strategy_directory=config.strategy_directory,
        model_directory=config.model_directory,
        models_directory=config.models_directory,
    )


def _parse_timerange(timerange: str) -> tuple[datetime, datetime]:
    try:
        start_text, end_text = timerange.split("-", 1)
        start = datetime.strptime(start_text, "%Y%m%d").replace(tzinfo=UTC)
        end = datetime.strptime(end_text, "%Y%m%d").replace(tzinfo=UTC)
    except ValueError as exc:
        raise ValueError("timerange must use YYYYMMDD-YYYYMMDD") from exc
    if end <= start:
        raise ValueError("timerange end must follow its start")
    return start, end


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


def audit_timerange_data(config: Phase18Config, case: Phase17Case) -> dict[str, Any]:
    """Verify that the native benchmark interval contains at least 5,000 real candles."""
    timerange = PHASE18_DIAGNOSTIC_TIMERANGES[case.asset]
    start, end = _parse_timerange(timerange)
    requested = TimeRange(
        starttype="date",
        stoptype="date",
        startts=int(start.timestamp()),
        stopts=int(end.timestamp()),
    )
    dataframe = load_pair_history(
        pair=case.pair,
        timeframe=PHASE18_TIMEFRAME,
        datadir=config.data_directory,
        timerange=requested,
        fill_up_missing=False,
        drop_incomplete=False,
        data_format="feather",
        candle_type=CandleType.FUTURES,
    )
    dataframe = dataframe.loc[(dataframe["date"] >= start) & (dataframe["date"] < end)].copy()
    unique_candles = int(dataframe["date"].nunique())
    expected_candles = int((end - start).total_seconds() / (15 * 60))
    duplicate_candles = int(len(dataframe) - unique_candles)
    missing_candles = max(expected_candles - unique_candles, 0)
    gap_fraction = missing_candles / expected_candles
    checks = {
        "at_least_5000_real_candles": unique_candles >= PHASE18_MINIMUM_BENCHMARK_CANDLES,
        "no_duplicate_timestamps": duplicate_candles == 0,
        "gap_fraction_within_one_percent": gap_fraction <= 0.01,
        "interval_ends_before_holdout": end.strftime("%Y%m%d") <= PHASE17_HOLDOUT_START,
    }
    return {
        "asset": case.asset,
        "pair": case.pair,
        "timerange": timerange,
        "passed": all(checks.values()),
        "checks": checks,
        "expected_candles": expected_candles,
        "unique_candles": unique_candles,
        "duplicate_candles": duplicate_candles,
        "missing_candles": missing_candles,
        "gap_fraction": gap_fraction,
    }


def prepare_phase18(config: Phase18Config) -> dict[str, Any]:
    phase15, phase17 = config.validate()
    config.output_directory.mkdir(parents=True, exist_ok=True)
    phase17_config = _phase17_config(config)
    cases = discover_phase15_cases(phase17_config)
    representatives = select_representative_cases(cases)
    config_audits = [audit_case_config(case) for case in cases]
    source_audit = audit_shared_causal_source()
    data_audits = []
    seen_assets: set[str] = set()
    for case in representatives:
        if case.asset not in seen_assets:
            data_audits.append(audit_timerange_data(config, case))
            seen_assets.add(case.asset)
    checks = {
        "phase15_gate_passed": phase15["gate"]["passed"] is True,
        "phase17_gate_passed": phase17["gate"]["passed"] is True,
        "phase17_protocol_is_frozen": phase17.get("protocol_version") == PHASE17_PROTOCOL_VERSION,
        "complete_phase15_case_inventory": len(cases) == 117,
        "all_phase15_configs_pass_static_audit": all(row["passed"] for row in config_audits),
        "shared_feature_source_is_causal": source_audit["passed"] is True,
        "same_predeclared_representatives_as_phase17": len(representatives)
        == PHASE17_EXPECTED_NATIVE_CASES,
        "all_benchmark_data_audits_passed": len(data_audits) == 3
        and all(row["passed"] for row in data_audits),
        "configured_startup_is_tested": PHASE18_CONFIGURED_STARTUP in PHASE18_STARTUP_CANDLES,
        "exchange_safe_maximum_is_respected": max(PHASE18_STARTUP_CANDLES) <= 2494,
        "diagnostic_periods_end_before_holdout": all(
            timerange.split("-")[1] <= PHASE17_HOLDOUT_START
            for timerange in PHASE18_DIAGNOSTIC_TIMERANGES.values()
        ),
        "prediction_or_trading_results_did_not_select_cases": True,
    }
    manifest_path = config.output_directory / "case_manifest.csv"
    audit_path = config.output_directory / "config_audit.csv"
    data_path = config.output_directory / "data_audit.csv"
    preparation_path = config.output_directory / "preparation.json"
    representative_set = set(representative_case_ids())
    _write_csv(
        manifest_path,
        [
            {
                **asdict(case),
                "source_config": str(case.source_config),
                "native_representative": case.case_id in representative_set,
            }
            for case in cases
        ],
    )
    _write_csv(audit_path, config_audits)
    _write_csv(data_path, data_audits)
    summary = {
        "phase": 18,
        "stage": "preparation",
        "protocol_version": PHASE18_PROTOCOL_VERSION,
        "objective": "Audit recursive indicator stability in the frozen Phase 15 pipeline",
        "design": {
            "complete_case_inventory": len(cases),
            "native_representative_case_ids": list(representative_case_ids()),
            "native_representative_count": len(representatives),
            "startup_candles": list(PHASE18_STARTUP_CANDLES),
            "configured_strategy_startup": PHASE18_CONFIGURED_STARTUP,
            "relative_tolerance": PHASE18_RELATIVE_TOLERANCE,
            "minimum_benchmark_candles": PHASE18_MINIMUM_BENCHMARK_CANDLES,
            "diagnostic_timeranges": PHASE18_DIAGNOSTIC_TIMERANGES,
            "market_feature_count": len(PHASE18_MARKET_FEATURES),
            "selection_used_prediction_or_trading_results": False,
            "holdout_used": False,
        },
        "source_audit": source_audit,
        "data_audits": data_audits,
        "gate": {"passed": all(checks.values()), "checks": checks},
        "artifacts": {
            "preparation": str(preparation_path),
            "case_manifest": str(manifest_path),
            "config_audit": str(audit_path),
            "data_audit": str(data_path),
        },
    }
    preparation_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def diagnostic_identifier(case: Phase17Case) -> str:
    digest = hashlib.sha256(case.case_id.encode()).hexdigest()[:12]
    return f"double-descent-phase18-{case.asset}-{digest}"


def build_diagnostic_config(
    config: Phase18Config, case: Phase17Case, case_directory: Path
) -> dict[str, Any]:
    generated = json.loads(case.source_config.read_text(encoding="utf-8"))
    diagnostic = deepcopy(generated)
    diagnostic["backtest_cache"] = "none"
    freqai = diagnostic["freqai"]
    freqai["identifier"] = diagnostic_identifier(case)
    freqai["save_backtest_models"] = False
    parameters = freqai["model_training_parameters"]
    metrics_path = (case_directory / "training_diagnostics.jsonl").resolve()
    for key in tuple(parameters):
        if key.endswith("_metrics_path"):
            parameters[key] = str(metrics_path)
        elif key.endswith("_run_id"):
            parameters[key] = f"phase18-{case.case_id}"
    if case.model_family == "rff":
        # Lifecycle-only flag validated in Phase 17. It leaves RFF arithmetic unchanged.
        parameters["phase17_close_cuda_worker_after_predict"] = True
    return diagnostic


def build_recursive_command(
    config: Phase18Config,
    case: Phase17Case,
    diagnostic_config: Path,
    log_path: Path,
) -> list[str]:
    return [
        config.python_executable,
        "-m",
        "freqtrade",
        "recursive-analysis",
        "--config",
        str(diagnostic_config),
        "--strategy",
        PHASE18_STRATEGY,
        "--strategy-path",
        str(config.strategy_directory),
        "--freqaimodel",
        model_name_for_case(case),
        "--freqaimodel-path",
        str(config.model_directory),
        "--timerange",
        PHASE18_DIAGNOSTIC_TIMERANGES[case.asset],
        "--timeframe",
        PHASE18_TIMEFRAME,
        "--pairs",
        case.pair,
        "--datadir",
        str(config.data_directory),
        "--startup-candle",
        *(str(value) for value in PHASE18_STARTUP_CANDLES),
        "--logfile",
        str(log_path),
        "--no-color",
    ]


def _table_cells(line: str) -> list[str] | None:
    normalized = line.strip().replace("│", "|").replace("┃", "|")
    if not (normalized.startswith("|") and normalized.endswith("|")):
        return None
    return [cell.strip() for cell in normalized.strip("|").split("|")]


def _parse_percentage(value: str) -> float | None:
    normalized = value.strip().lower()
    if normalized == "-":
        return 0.0
    if normalized == "nan%":
        return None
    if not normalized.endswith("%"):
        raise ValueError(f"unexpected recursive-analysis cell {value!r}")
    return float(normalized[:-1].strip()) / 100.0


def parse_recursive_table(output: str) -> dict[str, dict[int, float | None]]:
    """Parse Freqtrade's no-color Rich table without treating '-' as missing evidence."""
    header: list[int] | None = None
    parsed: dict[str, dict[int, float | None]] = {}
    for line in output.splitlines():
        cells = _table_cells(line)
        if not cells:
            continue
        if cells[0].lower() == "indicators":
            header = [int(re.match(r"\d+", cell).group()) for cell in cells[1:]]
            continue
        if header is None or len(cells) != len(header) + 1:
            continue
        if set(cells[0]) <= {"-", "+", "="}:
            continue
        parsed[cells[0]] = {
            startup: _parse_percentage(value)
            for startup, value in zip(header, cells[1:], strict=True)
        }
    return parsed


def parse_recursive_output(
    output: str,
    log_text: str,
    returncode: int | None,
    timed_out: bool,
) -> dict[str, Any]:
    combined = f"{output}\n{log_text}"
    variances = parse_recursive_table(output)
    executed = sorted(
        {
            int(value)
            for value in re.findall(
                r"Calculating indicators using startup candle of (\d+)\.", combined
            )
        }
    )
    lookahead = sorted(set(re.findall(r"=> found lookahead in indicator ([^\r\n]+)", combined)))
    ignored_targets = [value for value in lookahead if value.startswith("&")]
    ignored_runtime = [
        value for value in lookahead if value in PHASE18_KNOWN_FREQAI_LOOKAHEAD_COLUMNS
    ]
    ignored_lookahead = set(ignored_targets + ignored_runtime)
    actionable_lookahead = [value for value in lookahead if value not in ignored_lookahead]

    configured_values = {
        indicator: values.get(PHASE18_CONFIGURED_STARTUP, 0.0)
        for indicator, values in variances.items()
    }
    actionable_recursive = sorted(
        indicator
        for indicator, value in configured_values.items()
        if value is None or abs(value) > PHASE18_RELATIVE_TOLERANCE
    )
    market_values = {
        indicator: variances.get(indicator, {}).get(PHASE18_CONFIGURED_STARTUP, 0.0)
        for indicator in PHASE18_MARKET_FEATURES
    }
    market_nan = sorted(indicator for indicator, value in market_values.items() if value is None)
    finite_market = [abs(value) for value in market_values.values() if value is not None]
    finite_all = [abs(value) for value in configured_values.values() if value is not None]
    max_market = max(finite_market, default=0.0)
    max_all = max(finite_all, default=0.0)
    recursive_result_detected = bool(variances) or (
        "No variance on indicator(s) found due to recursive formula." in combined
    )
    checks = {
        "returncode_zero": returncode == 0,
        "not_timed_out": not timed_out,
        "all_startup_candles_executed": executed == list(PHASE18_STARTUP_CANDLES),
        "recursive_result_detected": recursive_result_detected,
        "native_recursive_check_completed": "Start checking for recursive bias" in combined,
        "native_indicator_lookahead_check_completed": (
            "Start checking for lookahead bias on indicators only" in combined
        ),
        "market_features_finite_at_strategy_startup": not market_nan,
        "market_features_stable_at_strategy_startup": max_market <= PHASE18_RELATIVE_TOLERANCE,
        "no_actionable_recursive_indicators_at_strategy_startup": not actionable_recursive,
        "no_actionable_indicator_lookahead": not actionable_lookahead,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "startup_candles_executed": executed,
        "recursive_variances": variances,
        "recursive_indicator_count": len(variances),
        "recursive_no_variance_message": not variances and recursive_result_detected,
        "configured_startup": PHASE18_CONFIGURED_STARTUP,
        "relative_tolerance": PHASE18_RELATIVE_TOLERANCE,
        "max_abs_market_feature_variance_at_configured_startup": max_market,
        "max_abs_any_indicator_variance_at_configured_startup": max_all,
        "market_feature_variances_at_configured_startup": market_values,
        "market_feature_nan_at_configured_startup": market_nan,
        "actionable_recursive_indicators": actionable_recursive,
        "indicator_lookahead_findings": lookahead,
        "ignored_freqai_target_lookahead": ignored_targets,
        "ignored_freqai_runtime_lookahead": ignored_runtime,
        "actionable_indicator_lookahead": actionable_lookahead,
    }


def _source_fingerprint(config: Phase18Config, case: Phase17Case) -> str:
    paths = (
        case.source_config,
        config.project_root / "research/double_descent/phase18.py",
        config.strategy_directory / "Phase3EffectiveNStrategy.py",
        config.strategy_directory / "Phase4RFFStrategy.py",
        config.strategy_directory / "Phase15TimeframeStrategy.py",
        config.model_directory / "Phase4CudaRFFRegressor.py",
        config.model_directory / "Phase12BaselineRegressor.py",
        config.project_root / "freqtrade/optimize/analysis/recursive.py",
        config.project_root / "freqtrade/optimize/analysis/recursive_helpers.py",
    )
    digest = hashlib.sha256(f"phase18:{PHASE18_PROTOCOL_VERSION}".encode())
    for path in paths:
        digest.update(path.resolve().as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _clean_diagnostic_models(config: Phase18Config, identifier: str) -> None:
    root = (config.project_root / config.models_directory).resolve()
    target = (root / identifier).resolve()
    if target.parent != root or not identifier.startswith("double-descent-phase18-"):
        raise ValueError("refusing to remove a non-Phase18 model directory")
    if target.exists():
        shutil.rmtree(target)


def run_native_case(config: Phase18Config, case: Phase17Case) -> dict[str, Any]:
    case_directory = config.output_directory / "runs" / case.case_id
    case_directory.mkdir(parents=True, exist_ok=True)
    result_path = case_directory / "result.json"
    fingerprint = _source_fingerprint(config, case)
    if config.resume and result_path.is_file():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if cached.get("fingerprint") == fingerprint and cached.get("passed") is True:
            print(f"PHASE18 RECOVERED {case.case_id}", flush=True)
            return cached
    diagnostic_path = case_directory / "config.json"
    log_path = case_directory / "freqtrade.log"
    process_log_path = case_directory / "process.log"
    metrics_path = case_directory / "training_diagnostics.jsonl"
    for path in (log_path, process_log_path, metrics_path):
        path.unlink(missing_ok=True)
    diagnostic = build_diagnostic_config(config, case, case_directory)
    diagnostic_path.write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
    command = build_recursive_command(config, case, diagnostic_path, log_path.resolve())
    print(f"PHASE18 START {case.case_id}", flush=True)
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
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    parsed = parse_recursive_output(
        f"{stdout}\n{stderr}", log_text, returncode=returncode, timed_out=timed_out
    )
    result = {
        **asdict(case),
        **parsed,
        "source_config": str(case.source_config),
        "fingerprint": fingerprint,
        "identifier": diagnostic["freqai"]["identifier"],
        "diagnostic_timerange": PHASE18_DIAGNOSTIC_TIMERANGES[case.asset],
        "model_adapter": model_name_for_case(case),
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_seconds": wall_seconds,
        "artifacts": {
            "config": str(diagnostic_path),
            "freqtrade_log": str(log_path),
            "process_log": str(process_log_path),
            "training_diagnostics": str(metrics_path),
        },
        "stderr_tail": stderr[-2_000:],
    }
    result_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    _clean_diagnostic_models(config, diagnostic["freqai"]["identifier"])
    print(
        f"PHASE18 DONE {case.case_id} passed={result['passed']} "
        f"recursive_rows={result['recursive_indicator_count']} seconds={wall_seconds:.1f}",
        flush=True,
    )
    return result


def _recursive_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        variances = result.get("recursive_variances", {})
        indicators = sorted(set(PHASE18_MARKET_FEATURES) | set(variances))
        for indicator in indicators:
            for startup in PHASE18_STARTUP_CANDLES:
                rows.append(
                    {
                        "case_id": result["case_id"],
                        "asset": result["asset"],
                        "study": result["study"],
                        "model_name": result["model_name"],
                        "indicator": indicator,
                        "is_market_feature": indicator in PHASE18_MARKET_FEATURES,
                        "startup_candle": startup,
                        "is_configured_startup": startup == PHASE18_CONFIGURED_STARTUP,
                        "relative_variance": variances.get(indicator, {}).get(startup, 0.0),
                    }
                )
    return rows


def _lookahead_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        ignored = set(result.get("ignored_freqai_target_lookahead", [])) | set(
            result.get("ignored_freqai_runtime_lookahead", [])
        )
        for indicator in result.get("indicator_lookahead_findings", []):
            rows.append(
                {
                    "case_id": result["case_id"],
                    "asset": result["asset"],
                    "study": result["study"],
                    "model_name": result["model_name"],
                    "indicator": indicator,
                    "ignored_known_freqai_tail_column": indicator in ignored,
                    "actionable": indicator not in ignored,
                }
            )
    return rows


def run_phase18(
    config: Phase18Config,
    selected_case_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    preparation = prepare_phase18(config)
    if preparation["gate"]["passed"] is not True:
        raise ValueError("Phase 18 preparation did not pass")
    cases = discover_phase15_cases(_phase17_config(config))
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
                    "phase": 18,
                    "completed_case_ids": [row["case_id"] for row in results],
                    "passed_case_ids": [row["case_id"] for row in results if row["passed"]],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
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
        "all_startup_candle_grids_executed": len(complete_results) == PHASE17_EXPECTED_NATIVE_CASES
        and all(
            row.get("startup_candles_executed") == list(PHASE18_STARTUP_CANDLES)
            for row in complete_results
        ),
        "all_market_features_stable_at_configured_startup": len(complete_results)
        == PHASE17_EXPECTED_NATIVE_CASES
        and all(
            row.get("max_abs_market_feature_variance_at_configured_startup", math.inf)
            <= PHASE18_RELATIVE_TOLERANCE
            for row in complete_results
        ),
        "no_actionable_recursive_indicators": len(complete_results) == PHASE17_EXPECTED_NATIVE_CASES
        and all(not row.get("actionable_recursive_indicators") for row in complete_results),
        "no_actionable_indicator_lookahead": len(complete_results) == PHASE17_EXPECTED_NATIVE_CASES
        and all(not row.get("actionable_indicator_lookahead") for row in complete_results),
        "all_117_cases_have_declared_coverage": len(cases) == 117,
        "holdout_was_not_used": True,
        "full_unfiltered_protocol_executed": full_invocation,
    }
    result_csv = config.output_directory / "native_results.csv"
    recursive_csv = config.output_directory / "recursive_variances.csv"
    lookahead_csv = config.output_directory / "indicator_lookahead.csv"
    summary_path = config.output_directory / "summary.json"
    _write_csv(result_csv, complete_results)
    _write_csv(recursive_csv, _recursive_rows(complete_results))
    _write_csv(lookahead_csv, _lookahead_rows(complete_results))
    summary = {
        "phase": 18,
        "stage": "complete" if all(checks.values()) else "partial_or_failed",
        "protocol_version": PHASE18_PROTOCOL_VERSION,
        "objective": "Native Freqtrade recursive analysis of the frozen Phase 15 pipeline",
        "design": preparation["design"],
        "preparation_gate": preparation["gate"],
        "gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "expected_native_case_count": PHASE17_EXPECTED_NATIVE_CASES,
            "observed_native_case_count": len(complete_results),
        },
        "native_results": complete_results,
        "limitations": {
            "native_scope": (
                "recursive-analysis compares the final indicator row only and does not test "
                "entry or exit decisions; Phase 17 supplies the separate signal-level audit."
            ),
            "freqai_history": (
                "FreqAI expands each native startup window by the frozen 90-day training "
                "history. This is the deployed pipeline behavior, not a bare-indicator-only "
                "calculation with exactly N input rows."
            ),
            "native_early_break": (
                "Freqtrade stops comparison after the first startup window with no variance. "
                "All requested windows are still calculated; no variance at the smallest "
                "window is stronger evidence than the configured 801-candle point."
            ),
            "freqai_tail": (
                "Known target and do_predict lookahead findings at the truncated tail are "
                "retained and ignored only because Phase 17 independently found zero entry "
                "and exit mismatches. Any market feature or unknown column fails."
            ),
        },
        "artifacts": {
            "summary": str(summary_path),
            "preparation": preparation["artifacts"]["preparation"],
            "case_manifest": preparation["artifacts"]["case_manifest"],
            "config_audit": preparation["artifacts"]["config_audit"],
            "data_audit": preparation["artifacts"]["data_audit"],
            "native_results": str(result_csv),
            "recursive_variances": str(recursive_csv),
            "indicator_lookahead": str(lookahead_csv),
            "checkpoint": str(checkpoint_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary
