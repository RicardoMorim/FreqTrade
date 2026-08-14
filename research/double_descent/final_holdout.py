"""One-shot 2026 YTD holdout for the frozen supervised double-descent study."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from freqtrade.data.btanalysis import load_backtest_data
from research.double_descent.metrics import prediction_metrics
from research.double_descent.phase3 import Phase3Config, audit_data_coverage, run_phase3
from research.double_descent.phase4 import _load_evaluation_market_data, _parse_timerange
from research.double_descent.phase4 import _read_predictions as read_predictions
from research.double_descent.phase8 import assess_flat_curve
from research.double_descent.phase10 import (
    PHASE10_GAMMA,
    PHASE10_MAP_RATIOS,
    PHASE10_REPRESENTATIONS,
    PHASE10_ROBUSTNESS_RATIOS,
    PHASE10_SEEDS,
    Phase10Config,
    _json_safe,
    _run_substudy,
    _write_csv,
    flatten_substudies,
)
from research.double_descent.phase12 import (
    PHASE12_BASELINES,
    PHASE12_PREDICTION_BASELINES,
    PHASE12_RIDGE_ALPHA,
    Phase12Config,
)
from research.double_descent.phase12 import _phase4_config as phase12_phase4_config
from research.double_descent.phase12 import _recover_case as recover_baseline_case
from research.double_descent.phase12 import _run_case as run_baseline_case
from research.double_descent.phase12 import flatten_results as flatten_baseline_results
from research.double_descent.phase13 import benjamini_hochberg, hac_mean_test
from research.double_descent.phase15 import (
    PHASE15_CONTROL_BASELINES,
    PHASE15_CONTROL_RATIOS,
    PHASE15_INDICATOR_PERIODS,
    PHASE15_STARTUP_CANDLES,
    PHASE15_STRATEGY,
    PHASE15_STUDIES,
    AssetSpec,
    StudySpec,
)
from research.double_descent.phase16 import (
    PHASE16_COST_SCENARIOS,
    PHASE16_NATIVE_REPRODUCTION_TOLERANCE,
    _prepare_trade_frame,
    _strategy_stats,
    decompose_trade_pnl,
    summarize_repriced_trades,
)


FINAL_HOLDOUT_PROTOCOL_VERSION = 1
FINAL_HOLDOUT_START = "20260101"
FINAL_HOLDOUT_END = "20260801"
FINAL_HOLDOUT_TIMERANGE = f"{FINAL_HOLDOUT_START}-{FINAL_HOLDOUT_END}"
FINAL_DATA_TIMERANGE = "20250924-20260801"
FINAL_TIMEFRAME = "15m"
FINAL_EXPECTED_N = {"native_15m": 8_639, "matched_1h_control": 8_636}
FINAL_EXPECTED_WINDOWS = 8
FINAL_EXPECTED_RFF_CASES = 78
FINAL_EXPECTED_BASELINE_CASES = 39
FINAL_EXPECTED_CASES = FINAL_EXPECTED_RFF_CASES + FINAL_EXPECTED_BASELINE_CASES
FINAL_EXPECTED_FITS = FINAL_EXPECTED_CASES * FINAL_EXPECTED_WINDOWS
FINAL_EXPECTED_ECONOMIC_CASES = 87
FINAL_EXPECTED_COST_ROWS = FINAL_EXPECTED_ECONOMIC_CASES * len(PHASE16_COST_SCENARIOS)
FINAL_HAC_LAG_CANDLES = 96
FINAL_FDR_ALPHA = 0.05


FINAL_ASSETS = (
    AssetSpec(
        "btc",
        "BTC/USDT:USDT",
        "BTC",
        "crypto",
        FINAL_HOLDOUT_TIMERANGE,
        FINAL_EXPECTED_WINDOWS,
        FINAL_EXPECTED_WINDOWS,
        "common_2026_ytd_holdout",
    ),
    AssetSpec(
        "eth",
        "ETH/USDT:USDT",
        "ETH",
        "crypto",
        FINAL_HOLDOUT_TIMERANGE,
        FINAL_EXPECTED_WINDOWS,
        FINAL_EXPECTED_WINDOWS,
        "common_2026_ytd_holdout",
    ),
    AssetSpec(
        "gold",
        "PAXG/USDT:USDT",
        "PAXG gold proxy",
        "tokenized_gold_proxy",
        FINAL_HOLDOUT_TIMERANGE,
        FINAL_EXPECTED_WINDOWS,
        FINAL_EXPECTED_WINDOWS,
        "common_2026_ytd_holdout",
    ),
)


PROTOCOL_SOURCE_PATHS = (
    Path("scripts/run_double_descent_final_holdout.py"),
    Path("research/double_descent/final_holdout.py"),
    Path("research/double_descent/metrics.py"),
    Path("research/double_descent/phase3.py"),
    Path("research/double_descent/phase4.py"),
    Path("research/double_descent/phase10.py"),
    Path("research/double_descent/phase12.py"),
    Path("research/double_descent/phase13.py"),
    Path("research/double_descent/phase15.py"),
    Path("research/double_descent/phase16.py"),
    Path("research/double_descent/rff_cuda_worker.py"),
    Path("research/double_descent/freqai/Phase15TimeframeStrategy.py"),
    Path("research/double_descent/freqai/Phase4CudaRFFRegressor.py"),
    Path("research/double_descent/freqai/Phase12BaselineRegressor.py"),
)


@dataclass(frozen=True)
class FinalHoldoutConfig:
    data_directory: Path
    output_directory: Path = Path(
        "user_data/research_results/double_descent/final_holdout_2026_ytd"
    )
    protocol_path: Path = Path("research/double_descent/FINAL_HOLDOUT_PROTOCOL.json")
    report_path: Path = Path("research/double_descent/FINAL_STUDY_REPORT.md")
    phase15_summary: Path = Path("user_data/research_results/double_descent/phase15/summary.json")
    phase16_summary: Path = Path("user_data/research_results/double_descent/phase16/summary.json")
    phase17_summary: Path = Path("user_data/research_results/double_descent/phase17/summary.json")
    phase18_summary: Path = Path("user_data/research_results/double_descent/phase18/summary.json")
    phase19_summary: Path = Path("user_data/research_results/double_descent/phase19/summary.json")
    phase20_summary: Path = Path("user_data/research_results/double_descent/phase20/summary.json")
    phase9_summary: Path = Path("user_data/research_results/double_descent/phase9/summary.json")
    frozen_gamma_file: Path = Path(
        "user_data/research_results/double_descent/phase9/frozen_gamma.json"
    )
    phase10_summary: Path = Path("user_data/research_results/double_descent/phase10/summary.json")
    phase10_map: Path = Path(
        "user_data/research_results/double_descent/phase10/representation_map.csv"
    )
    phase11_summary: Path = Path("user_data/research_results/double_descent/phase11/summary.json")
    python_executable: str = sys.executable
    cuda_python_executable: str = sys.executable
    timeframe: str = FINAL_TIMEFRAME
    timerange: str = FINAL_HOLDOUT_TIMERANGE
    data_timerange: str = FINAL_DATA_TIMERANGE
    train_period_days: int = 90
    backtest_period_days: int = 30
    startup_candles: int = PHASE15_STARTUP_CANDLES
    indicator_periods_candles: tuple[int, ...] = PHASE15_INDICATOR_PERIODS
    map_ratios: tuple[float, ...] = PHASE10_MAP_RATIOS
    robustness_ratios: tuple[float, ...] = PHASE10_ROBUSTNESS_RATIOS
    seeds: tuple[int, ...] = PHASE10_SEEDS
    baselines: tuple[str, ...] = PHASE12_BASELINES
    control_ratios: tuple[float, ...] = PHASE15_CONTROL_RATIOS
    control_baselines: tuple[str, ...] = PHASE15_CONTROL_BASELINES
    gamma: float = PHASE10_GAMMA
    ridge: float = 0.0
    ridge_alpha: float = PHASE12_RIDGE_ALPHA
    rcond: float = 1e-12
    dtype: str = "float64"
    chunk_size: int = 4_096
    fee: float = 0.001
    minimum_training_windows: int = FINAL_EXPECTED_WINDOWS
    effective_n_tolerance: int = 1
    subprocess_timeout_seconds: int = 3_600
    resume: bool = True
    strategy_directory: Path = Path("research/double_descent/freqai")
    model_directory: Path = Path("research/double_descent/freqai")
    models_directory: Path = Path("user_data/models")

    @property
    def preparation_path(self) -> Path:
        return self.output_directory / "preparation.json"

    def validate_frozen_design(self) -> None:
        expected = (
            (self.timeframe, FINAL_TIMEFRAME, "timeframe"),
            (self.timerange, FINAL_HOLDOUT_TIMERANGE, "holdout interval"),
            (self.data_timerange, FINAL_DATA_TIMERANGE, "download interval"),
            (self.train_period_days, 90, "training window"),
            (self.backtest_period_days, 30, "backtest window"),
            (self.startup_candles, PHASE15_STARTUP_CANDLES, "startup candles"),
            (self.indicator_periods_candles, PHASE15_INDICATOR_PERIODS, "indicators"),
            (self.map_ratios, PHASE10_MAP_RATIOS, "P/N map"),
            (self.robustness_ratios, PHASE10_ROBUSTNESS_RATIOS, "robustness map"),
            (self.seeds, PHASE10_SEEDS, "seeds"),
            (self.baselines, PHASE12_BASELINES, "baselines"),
            (self.control_ratios, PHASE15_CONTROL_RATIOS, "control P/N map"),
            (self.control_baselines, PHASE15_CONTROL_BASELINES, "control baselines"),
            (self.gamma, PHASE10_GAMMA, "gamma"),
            (self.ridge, 0.0, "ridgeless setting"),
            (self.ridge_alpha, PHASE12_RIDGE_ALPHA, "baseline Ridge alpha"),
            (self.rcond, 1e-12, "rcond"),
            (self.dtype, "float64", "dtype"),
            (self.fee, 0.001, "fee"),
            (self.minimum_training_windows, FINAL_EXPECTED_WINDOWS, "rolling windows"),
        )
        for actual, frozen, label in expected:
            if actual != frozen:
                raise ValueError(f"final holdout changed the frozen {label}")
        if not Path(self.python_executable).is_file():
            raise FileNotFoundError("Freqtrade Python executable does not exist")
        if not Path(self.cuda_python_executable).is_file():
            raise FileNotFoundError("CUDA Python executable does not exist")
        if tuple(study.name for study in PHASE15_STUDIES) != (
            "native_15m",
            "matched_1h_control",
        ):
            raise ValueError("Phase 15 study definitions changed")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "git command failed")
    return completed.stdout.strip()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def validate_completed_development(config: FinalHoldoutConfig) -> dict[str, Any]:
    summaries = {
        phase: _load_json(path)
        for phase, path in (
            (15, config.phase15_summary),
            (16, config.phase16_summary),
            (17, config.phase17_summary),
            (18, config.phase18_summary),
            (19, config.phase19_summary),
            (20, config.phase20_summary),
        )
    }
    for phase in (15, 16, 17, 18, 20):
        summary = summaries[phase]
        if summary.get("phase") != phase or summary.get("gate", {}).get("passed") is not True:
            raise ValueError(f"Phase {phase} did not complete its integrity gate")
        if summary.get("design", {}).get("holdout_used") is not False:
            raise ValueError(f"Phase {phase} does not certify a sealed holdout")
    phase19 = summaries[19]
    if phase19.get("phase") != 19:
        raise ValueError("invalid Phase 19 summary")
    checks = phase19.get("gate", {}).get("checks", {})
    if not checks.get("full_unfiltered_protocol_executed"):
        raise ValueError("Phase 19 did not complete the numerical protocol")
    if phase19.get("design", {}).get("holdout_used") is not False:
        raise ValueError("Phase 19 does not certify a sealed holdout")
    return summaries


def freeze_final_holdout_protocol(config: FinalHoldoutConfig) -> dict[str, Any]:
    """Write the immutable design manifest without reading holdout market data."""
    config.validate_frozen_design()
    summaries = validate_completed_development(config)
    source_hashes = {}
    for path in PROTOCOL_SOURCE_PATHS:
        if not path.is_file():
            raise FileNotFoundError(path)
        source_hashes[path.as_posix()] = _sha256(path)
    reference_hashes = {
        path.as_posix(): _sha256(path)
        for path in (
            config.phase9_summary,
            config.frozen_gamma_file,
            config.phase10_summary,
            config.phase10_map,
            config.phase11_summary,
            config.phase15_summary,
            config.phase16_summary,
            config.phase17_summary,
            config.phase18_summary,
            config.phase19_summary,
            config.phase20_summary,
        )
    }
    payload = {
        "protocol": "supervised_double_descent_final_holdout",
        "protocol_version": FINAL_HOLDOUT_PROTOCOL_VERSION,
        "frozen_at_utc": datetime.now(tz=UTC).isoformat(),
        "protocol_parent_commit": _git_output("rev-parse", "HEAD"),
        "holdout_accessed_when_frozen": False,
        "holdout": {
            "label": "2026 YTD through the last complete month available at freeze time",
            "timerange": FINAL_HOLDOUT_TIMERANGE,
            "start_inclusive": "2026-01-01T00:00:00Z",
            "end_exclusive": "2026-08-01T00:00:00Z",
            "full_calendar_year": False,
            "evaluation": "prequential 90-day train / 30-day predict rolling windows",
        },
        "design": {
            "assets": [asdict(asset) for asset in FINAL_ASSETS],
            "studies": [asdict(study) for study in PHASE15_STUDIES],
            "timeframe": FINAL_TIMEFRAME,
            "data_timerange": FINAL_DATA_TIMERANGE,
            "train_period_days": config.train_period_days,
            "backtest_period_days": config.backtest_period_days,
            "effective_n": FINAL_EXPECTED_N,
            "expected_windows_per_case": FINAL_EXPECTED_WINDOWS,
            "map_ratios": list(config.map_ratios),
            "robustness_ratios": list(config.robustness_ratios),
            "seeds": list(config.seeds),
            "baselines": list(config.baselines),
            "control_ratios": list(config.control_ratios),
            "control_baselines": list(config.control_baselines),
            "gamma": config.gamma,
            "ridge": config.ridge,
            "ridge_alpha": config.ridge_alpha,
            "dtype": config.dtype,
            "fee_per_side": config.fee,
            "cost_scenarios": [asdict(item) for item in PHASE16_COST_SCENARIOS],
            "primary_inference": "prediction metrics; trading remains downstream",
            "prediction_test": {
                "method": "two-sided Bartlett-HAC MSE loss differential versus zero",
                "lag_candles": FINAL_HAC_LAG_CANDLES,
                "multiple_testing": "Benjamini-Hochberg over all non-zero prediction cases",
                "fdr_alpha": FINAL_FDR_ALPHA,
            },
            "performance_never_changes_integrity_gate": True,
        },
        "expected_counts": {
            "rff_cases": FINAL_EXPECTED_RFF_CASES,
            "baseline_cases": FINAL_EXPECTED_BASELINE_CASES,
            "total_cases": FINAL_EXPECTED_CASES,
            "rolling_fits": FINAL_EXPECTED_FITS,
            "economic_cases": FINAL_EXPECTED_ECONOMIC_CASES,
            "cost_rows": FINAL_EXPECTED_COST_ROWS,
        },
        "development_status": {
            str(phase): {
                "gate_passed": summaries[phase].get("gate", {}).get("passed"),
                "holdout_used": summaries[phase].get("design", {}).get("holdout_used"),
            }
            for phase in summaries
        },
        "phase19_quarantine": (
            "The complete protocol ran, but its strict gate failed one CPU maximum-absolute "
            "tolerance near P/N=1. Final interpolation points remain numerically qualified."
        ),
        "source_sha256": source_hashes,
        "reference_sha256": reference_hashes,
    }
    config.protocol_path.parent.mkdir(parents=True, exist_ok=True)
    config.protocol_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def validate_frozen_protocol(config: FinalHoldoutConfig) -> dict[str, Any]:
    config.validate_frozen_design()
    validate_completed_development(config)
    protocol = _load_json(config.protocol_path)
    if protocol.get("protocol_version") != FINAL_HOLDOUT_PROTOCOL_VERSION:
        raise ValueError("unsupported final holdout protocol")
    if protocol.get("holdout_accessed_when_frozen") is not False:
        raise ValueError("protocol does not certify a sealed holdout at freeze time")
    if protocol.get("holdout", {}).get("timerange") != FINAL_HOLDOUT_TIMERANGE:
        raise ValueError("holdout timerange drifted after freeze")
    for raw_path, expected in protocol.get("source_sha256", {}).items():
        path = Path(raw_path)
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"frozen source changed after protocol freeze: {path}")
    for raw_path, expected in protocol.get("reference_sha256", {}).items():
        path = Path(raw_path)
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"development artifact changed after protocol freeze: {path}")
    return protocol


def build_download_command(config: FinalHoldoutConfig) -> list[str]:
    return [
        config.python_executable,
        "-m",
        "freqtrade",
        "download-data",
        "--exchange",
        "binance",
        "--pairs",
        *(asset.pair for asset in FINAL_ASSETS),
        "--timeframes",
        config.timeframe,
        "--timerange",
        config.data_timerange,
        "--trading-mode",
        "futures",
        "--candle-types",
        "futures",
        "funding_rate",
        "mark",
        "--data-format-ohlcv",
        "feather",
        "--datadir",
        str(config.data_directory),
        "--no-color",
    ]


def download_final_holdout_data(config: FinalHoldoutConfig) -> None:
    validate_frozen_protocol(config)
    config.data_directory.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    completed = subprocess.run(
        build_download_command(config),
        cwd=Path.cwd(),
        env=environment,
        check=False,
        timeout=config.subprocess_timeout_seconds,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"final holdout data download failed with code {completed.returncode}")


def _phase3_config(
    config: FinalHoldoutConfig,
    asset: AssetSpec,
    study: StudySpec,
    output_directory: Path,
) -> Phase3Config:
    return Phase3Config(
        data_directory=config.data_directory,
        output_directory=output_directory,
        python_executable=config.python_executable,
        pair=asset.pair,
        timeframe=config.timeframe,
        timerange=asset.timerange,
        train_periods_days=(config.train_period_days,),
        backtest_period_days=config.backtest_period_days,
        startup_candles=config.startup_candles,
        label_period_candles=study.label_period_candles,
        indicator_periods_candles=config.indicator_periods_candles,
        minimum_windows_per_period=config.minimum_training_windows,
        subprocess_timeout_seconds=config.subprocess_timeout_seconds,
        strategy_name=PHASE15_STRATEGY,
        identifier_prefix=f"double-descent-final-n-{asset.alias}-{study.name}",
        allow_timeframe_variation=True,
        strategy_directory=config.strategy_directory,
        model_directory=config.model_directory,
    )


def _read_or_measure_n(
    config: FinalHoldoutConfig,
    asset: AssetSpec,
    study: StudySpec,
) -> dict[str, Any]:
    output = config.output_directory / "preparation" / "effective_n" / asset.alias / study.name
    summary_path = output / "summary.json"
    if config.resume and summary_path.is_file():
        cached = _load_json(summary_path)
        cached_config = cached.get("config", {})
        if (
            cached.get("gate", {}).get("passed") is True
            and cached_config.get("pair") == asset.pair
            and cached_config.get("timerange") == asset.timerange
            and cached_config.get("label_period_candles") == study.label_period_candles
        ):
            return cached
    return run_phase3(_phase3_config(config, asset, study, output))


def _data_fingerprints(config: FinalHoldoutConfig) -> dict[str, str]:
    prefixes = tuple(asset.pair.replace("/", "_").replace(":", "_") for asset in FINAL_ASSETS)
    files = sorted(
        path for path in config.data_directory.rglob("*.feather") if path.name.startswith(prefixes)
    )
    return {str(path): _sha256(path) for path in files}


def prepare_final_holdout(config: FinalHoldoutConfig) -> dict[str, Any]:
    protocol = validate_frozen_protocol(config)
    config.output_directory.mkdir(parents=True, exist_ok=True)
    audits = {
        asset.alias: audit_data_coverage(
            _phase3_config(
                config,
                asset,
                PHASE15_STUDIES[0],
                config.output_directory / "preparation" / "audit" / asset.alias,
            )
        )
        for asset in FINAL_ASSETS
    }
    measurements: dict[str, dict[str, Any]] = {}
    effective_n_by_asset: dict[str, dict[str, int | None]] = {}
    for asset in FINAL_ASSETS:
        measurements[asset.alias] = {}
        effective_n_by_asset[asset.alias] = {}
        for study in PHASE15_STUDIES:
            summary = _read_or_measure_n(config, asset, study)
            measurements[asset.alias][study.name] = summary
            aggregates = summary.get("aggregate", [])
            effective_n_by_asset[asset.alias][study.name] = (
                int(aggregates[0]["effective_n_median"]) if len(aggregates) == 1 else None
            )
    effective_n = {}
    for study in PHASE15_STUDIES:
        values = {effective_n_by_asset[asset.alias][study.name] for asset in FINAL_ASSETS}
        effective_n[study.name] = values.pop() if len(values) == 1 else None
    rows = [
        (asset, study, measurements[asset.alias][study.name])
        for asset in FINAL_ASSETS
        for study in PHASE15_STUDIES
    ]
    checks = {
        "protocol_was_frozen_before_data_access": protocol["holdout_accessed_when_frozen"] is False,
        "all_data_audits_passed": all(item.get("passed") is True for item in audits.values()),
        "all_effective_n_measurements_passed": all(
            summary.get("gate", {}).get("passed") is True for _, _, summary in rows
        ),
        "effective_n_matches_predeclaration": all(
            effective_n_by_asset[asset.alias][study.name] == FINAL_EXPECTED_N[study.name]
            for asset, study, _ in rows
        ),
        "feature_dimension_is_25": all(
            summary.get("gate", {}).get("observed_feature_counts") == [25] for _, _, summary in rows
        ),
        "rolling_window_count_is_eight": all(
            summary.get("aggregate", [{}])[0].get("window_count") == FINAL_EXPECTED_WINDOWS
            for _, _, summary in rows
        ),
        "data_files_fingerprinted": bool(_data_fingerprints(config)),
    }
    summary = {
        "stage": "preparation",
        "protocol_version": FINAL_HOLDOUT_PROTOCOL_VERSION,
        "holdout_used": True,
        "timerange": FINAL_HOLDOUT_TIMERANGE,
        "effective_n": effective_n,
        "effective_n_by_asset": effective_n_by_asset,
        "data_audits": audits,
        "measurements": {
            asset.alias: {
                study.name: {
                    "gate": measurements[asset.alias][study.name].get("gate"),
                    "aggregate": measurements[asset.alias][study.name].get("aggregate"),
                }
                for study in PHASE15_STUDIES
            }
            for asset in FINAL_ASSETS
        },
        "data_sha256": _data_fingerprints(config),
        "gate": {"passed": all(checks.values()), "checks": checks},
    }
    config.preparation_path.write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return summary


def _load_passed_preparation(config: FinalHoldoutConfig) -> dict[str, Any]:
    summary = _load_json(config.preparation_path)
    if summary.get("gate", {}).get("passed") is not True:
        raise ValueError("final holdout preparation did not pass")
    return summary


def _rff_tasks(study: StudySpec) -> list[tuple[int, tuple[float, ...]]]:
    if study.role == "diagnostic_control":
        return [(PHASE10_SEEDS[0], PHASE15_CONTROL_RATIOS)]
    return [
        (seed, PHASE10_MAP_RATIOS if index == 0 else PHASE10_ROBUSTNESS_RATIOS)
        for index, seed in enumerate(PHASE10_SEEDS)
    ]


def _baseline_names(study: StudySpec) -> tuple[str, ...]:
    return PHASE15_CONTROL_BASELINES if study.role == "diagnostic_control" else PHASE12_BASELINES


def _phase10_config(
    config: FinalHoldoutConfig,
    asset: AssetSpec,
    study: StudySpec,
    effective_n: int,
    output_directory: Path,
) -> Phase10Config:
    return Phase10Config(
        data_directory=config.data_directory,
        output_directory=output_directory,
        phase9_summary=config.phase9_summary,
        frozen_gamma_file=config.frozen_gamma_file,
        python_executable=config.python_executable,
        cuda_python_executable=config.cuda_python_executable,
        pair=asset.pair,
        timeframe=config.timeframe,
        timerange=asset.timerange,
        holdout_start=FINAL_HOLDOUT_END,
        train_period_days=config.train_period_days,
        backtest_period_days=config.backtest_period_days,
        effective_n=effective_n,
        map_ratios=config.map_ratios,
        robustness_ratios=config.robustness_ratios,
        representations=PHASE10_REPRESENTATIONS,
        seeds=config.seeds,
        gamma=config.gamma,
        ridge=config.ridge,
        rcond=config.rcond,
        dtype=config.dtype,
        chunk_size=config.chunk_size,
        fee=config.fee,
        minimum_training_windows=config.minimum_training_windows,
        effective_n_tolerance=config.effective_n_tolerance,
        minimum_seed_count=1,
        subprocess_timeout_seconds=config.subprocess_timeout_seconds,
        resume=config.resume,
        smoke_test=True,
        allow_external_design=True,
        label_period_candles=study.label_period_candles,
        startup_candles=config.startup_candles,
        indicator_periods_candles=config.indicator_periods_candles,
        strategy_name=PHASE15_STRATEGY,
        strategy_directory=config.strategy_directory,
        model_directory=config.model_directory,
        models_directory=config.models_directory,
    )


def _phase12_config(
    config: FinalHoldoutConfig,
    asset: AssetSpec,
    study: StudySpec,
    effective_n: int,
    output_directory: Path,
) -> Phase12Config:
    return Phase12Config(
        data_directory=config.data_directory,
        output_directory=output_directory,
        phase10_summary=config.phase10_summary,
        phase10_map=config.phase10_map,
        phase11_summary=config.phase11_summary,
        python_executable=config.python_executable,
        pair=asset.pair,
        timeframe=config.timeframe,
        timerange=asset.timerange,
        holdout_start=FINAL_HOLDOUT_END,
        train_period_days=config.train_period_days,
        backtest_period_days=config.backtest_period_days,
        effective_n=effective_n,
        baselines=_baseline_names(study),
        ridge_alpha=config.ridge_alpha,
        fee=config.fee,
        minimum_training_windows=config.minimum_training_windows,
        effective_n_tolerance=config.effective_n_tolerance,
        subprocess_timeout_seconds=config.subprocess_timeout_seconds,
        resume=config.resume,
        smoke_test=True,
        allow_external_design=True,
        label_period_candles=study.label_period_candles,
        startup_candles=config.startup_candles,
        indicator_periods_candles=config.indicator_periods_candles,
        strategy_name=PHASE15_STRATEGY,
        momentum_24h_scale=96.0 if study.label_period_candles == 1 else 24.0,
        strategy_directory=config.strategy_directory,
        model_directory=config.model_directory,
        models_directory=config.models_directory,
    )


def _run_baselines(
    engine: Phase12Config,
    run_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    phase4 = phase12_phase4_config(engine)
    phase4.validate()
    market_data = _load_evaluation_market_data(phase4)
    results = []
    for baseline in engine.baselines:
        result = recover_baseline_case(engine, baseline, phase4, market_data)
        if result is None:
            result = run_baseline_case(engine, baseline, phase4, run_id, market_data)
        results.append(result)
    return results, flatten_baseline_results(results)


def _tag_row(row: dict[str, Any], asset: AssetSpec, study: StudySpec) -> dict[str, Any]:
    return {
        "asset": asset.alias,
        "asset_label": asset.label,
        "asset_class": asset.asset_class,
        "pair": asset.pair,
        "asset_timerange": asset.timerange,
        "comparison_scope": asset.comparison_scope,
        "study": study.name,
        "study_role": study.role,
        "target_minutes": study.target_minutes,
        "label_period_candles": study.label_period_candles,
        **row,
    }


def _case_record(
    result: dict[str, Any],
    asset: AssetSpec,
    study: StudySpec,
    model_family: str,
    model_name: str,
) -> dict[str, Any]:
    return {
        "asset": asset.alias,
        "pair": asset.pair,
        "timerange": asset.timerange,
        "study": study.name,
        "study_role": study.role,
        "model_family": model_family,
        "model_name": model_name,
        "seed": result.get("seed"),
        "target_pn_ratio": result.get("target_pn_ratio"),
        "feature_count": result.get("feature_count"),
        "classification": result.get("classification", "prediction"),
        "prediction_directory": result.get("artifacts", {}).get("prediction_directory"),
        "backtest": result.get("artifacts", {}).get("backtest"),
        "identifier": result.get("identifier"),
    }


def _curve_assessments(rff_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assessments = []
    for asset in FINAL_ASSETS:
        for study in PHASE15_STUDIES:
            for seed, ratios in _rff_tasks(study):
                matched = [
                    row
                    for row in rff_rows
                    if row["asset"] == asset.alias
                    and row["study"] == study.name
                    and int(row["seed"]) == seed
                    and float(row["target_pn_ratio"]) in ratios
                ]
                assessments.append(
                    {
                        "asset": asset.alias,
                        "study": study.name,
                        "seed": seed,
                        **assess_flat_curve(matched),
                    }
                )
    return assessments


def compare_holdout_predictions(
    rff_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    curves = _curve_assessments(rff_rows)
    cells = []
    for asset in FINAL_ASSETS:
        for study in PHASE15_STUDIES:
            rff = [
                row
                for row in rff_rows
                if row["asset"] == asset.alias and row["study"] == study.name
            ]
            baselines = [
                row
                for row in baseline_rows
                if row["asset"] == asset.alias and row["study"] == study.name
            ]
            prediction_baselines = [
                row for row in baselines if row["baseline"] in PHASE12_PREDICTION_BASELINES
            ]
            zero = next(row for row in prediction_baselines if row["baseline"] == "zero_return")
            best_simple = min(prediction_baselines, key=lambda row: float(row["oos_mse"]))
            best_rff = min(rff, key=lambda row: float(row["oos_mse"]))
            cell_curves = [
                row for row in curves if row["asset"] == asset.alias and row["study"] == study.name
            ]
            cells.append(
                {
                    "asset": asset.alias,
                    "study": study.name,
                    "seed_count": len(cell_curves),
                    "double_descent_seed_count": sum(
                        bool(row.get("double_descent_pattern_detected")) for row in cell_curves
                    ),
                    "useful_second_descent_seed_count": sum(
                        bool(row.get("largest_model_beats_zero_mse"))
                        and bool(row.get("second_descent_beats_best_underparameterized"))
                        for row in cell_curves
                    ),
                    "rff_case_count_beating_zero": sum(
                        float(row["oos_mse"]) < float(row["zero_mse"]) for row in rff
                    ),
                    "best_simple_baseline": best_simple["baseline"],
                    "best_simple_mse_over_zero": float(best_simple["oos_mse"])
                    / float(zero["oos_mse"]),
                    "best_rff_mse_over_zero": float(best_rff["oos_mse"])
                    / float(best_rff["zero_mse"]),
                    "rff_complexity_justified": float(best_rff["oos_mse"])
                    < float(best_simple["oos_mse"]),
                }
            )
    primary = [row for row in cells if row["study"] == "native_15m"]
    return {
        "primary_metric": "2026 YTD prequential OOS MSE normalized by zero forecast",
        "curve_assessments": curves,
        "asset_study_comparisons": cells,
        "primary_shape_replication_asset_count": sum(
            row["double_descent_seed_count"] >= math.ceil(row["seed_count"] / 2) for row in primary
        ),
        "primary_useful_replication_asset_count": sum(
            row["useful_second_descent_seed_count"] >= math.ceil(row["seed_count"] / 2)
            for row in primary
        ),
        "primary_asset_count": len(primary),
        "trading_excluded_from_prediction_decision": True,
    }


def evaluate_prediction_significance(
    config: FinalHoldoutConfig,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    market_cache: dict[tuple[str, str], pd.DataFrame] = {}
    for case in cases:
        if case["classification"] != "prediction":
            continue
        key = (case["asset"], case["study"])
        if key not in market_cache:
            asset = next(item for item in FINAL_ASSETS if item.alias == case["asset"])
            study = next(item for item in PHASE15_STUDIES if item.name == case["study"])
            baseline = _phase12_config(
                config,
                asset,
                study,
                FINAL_EXPECTED_N[study.name],
                config.output_directory / "significance" / asset.alias / study.name,
            )
            market_cache[key] = _load_evaluation_market_data(phase12_phase4_config(baseline))
        predictions = read_predictions(Path(case["prediction_directory"]))
        start, end = _parse_timerange(case["timerange"])
        predictions = predictions.loc[
            (predictions["date"] >= start) & (predictions["date"] < end)
        ].copy()
        merged = predictions.merge(
            market_cache[key][["date", "realized_forward_return"]],
            on="date",
            how="left",
            validate="many_to_one",
        )
        valid = (
            (merged["do_predict"] == 1)
            & np.isfinite(merged["&-forward_return"])
            & np.isfinite(merged["realized_forward_return"])
        )
        evaluated = merged.loc[valid]
        actual = evaluated["realized_forward_return"].to_numpy(dtype=np.float64)
        predicted = evaluated["&-forward_return"].to_numpy(dtype=np.float64)
        model = prediction_metrics(actual, predicted)
        zero = prediction_metrics(actual, np.zeros_like(actual))
        loss_difference = np.square(predicted - actual) - np.square(actual)
        test = hac_mean_test(loss_difference, FINAL_HAC_LAG_CANDLES)
        rows.append(
            {
                **{key: value for key, value in case.items() if key != "prediction_directory"},
                "prediction_rows": len(predictions),
                "valid_prediction_rows": len(evaluated),
                "duplicate_prediction_dates": int(predictions["date"].duplicated().sum()),
                "all_predictions_accepted": bool((predictions["do_predict"] == 1).all()),
                **{f"oos_{key}": value for key, value in model.items()},
                **{f"zero_{key}": value for key, value in zero.items()},
                "mse_over_zero": float(model["mse"]) / float(zero["mse"]),
                "loss_difference_vs_zero_mean": test["mean"],
                "loss_difference_hac_standard_error": test["standard_error"],
                "loss_difference_hac_t_stat": test["t_stat"],
                "loss_difference_hac_p_unadjusted": test["p"],
            }
        )
    family = [row for row in rows if row["model_name"] != "zero_return"]
    adjusted = benjamini_hochberg(
        [float(row["loss_difference_hac_p_unadjusted"]) for row in family]
    )
    by_identifier = {row["identifier"]: value for row, value in zip(family, adjusted, strict=True)}
    for row in rows:
        q = by_identifier.get(row["identifier"], math.nan)
        row["loss_difference_fdr_q"] = q
        row["significantly_better_than_zero_fdr"] = bool(
            math.isfinite(q) and q <= FINAL_FDR_ALPHA and row["loss_difference_vs_zero_mean"] < 0
        )
        row["significantly_worse_than_zero_fdr"] = bool(
            math.isfinite(q) and q <= FINAL_FDR_ALPHA and row["loss_difference_vs_zero_mean"] > 0
        )
    return rows


def evaluate_costs(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        if case["study"] != "native_15m":
            continue
        backtest_path = Path(case["backtest"])
        trades = _prepare_trade_frame(load_backtest_data(backtest_path), backtest_path)
        stats = _strategy_stats(backtest_path)
        starting_balance = float(stats["starting_balance"])
        for scenario in PHASE16_COST_SCENARIOS:
            metrics = summarize_repriced_trades(
                trades,
                scenario,
                starting_balance,
                case["timerange"],
            )
            row = {
                **{key: value for key, value in case.items() if key != "prediction_directory"},
                **metrics,
                "native_freqtrade_profit_abs": float(stats["profit_total_abs"]),
                "native_freqtrade_return": float(stats["profit_total"]),
                "native_freqtrade_sharpe": float(stats["sharpe"]),
            }
            if scenario.name == "phase15_reference":
                reconstructed = decompose_trade_pnl(trades, scenario)
                errors = (
                    reconstructed["net_pnl_abs"] - trades["profit_abs"]
                    if len(trades)
                    else pd.Series(dtype="float64")
                )
                row["native_max_trade_pnl_error"] = (
                    float(errors.abs().max()) if len(errors) else 0.0
                )
                row["native_total_pnl_error"] = abs(
                    float(reconstructed["net_pnl_abs"].sum()) - float(stats["profit_total_abs"])
                )
            else:
                row["native_max_trade_pnl_error"] = None
                row["native_total_pnl_error"] = None
            rows.append(row)
    return rows


def _cost_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for scenario in PHASE16_COST_SCENARIOS:
        matched = [row for row in rows if row["scenario"] == scenario.name]
        rff = [row for row in matched if row["model_family"] == "rff"]
        output.append(
            {
                "scenario": scenario.name,
                "case_count": len(matched),
                "positive_return_case_count": sum(row["net_return"] > 0 for row in matched),
                "positive_rff_case_count": sum(row["net_return"] > 0 for row in rff),
                "minimum_trades_met_case_count": sum(
                    row["minimum_trade_count_met"] for row in matched
                ),
                "median_net_return": statistics.median(row["net_return"] for row in matched),
                "median_daily_sharpe": statistics.median(row["daily_sharpe"] for row in matched),
            }
        )
    return output


def evaluate_final_gate(
    rff_results: list[dict[str, Any]],
    rff_rows: list[dict[str, Any]],
    baseline_results: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    prediction_tests: list[dict[str, Any]],
    cost_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    all_results = [*rff_results, *baseline_results]
    all_rows = [*rff_rows, *baseline_rows]
    configs = [Path(result["artifacts"]["config"]) for result in all_results]
    generated = [_load_json(path) for path in configs]
    logs = [Path(result["artifacts"]["log"]) for result in all_results]
    reference_costs = [row for row in cost_rows if row["scenario"] == "phase15_reference"]
    checks = {
        "all_rff_cases_passed": all(result.get("success") is True for result in rff_results),
        "all_baseline_cases_passed": all(
            result.get("success") is True for result in baseline_results
        ),
        "complete_rff_case_count": len(rff_rows) == FINAL_EXPECTED_RFF_CASES,
        "complete_baseline_case_count": len(baseline_rows) == FINAL_EXPECTED_BASELINE_CASES,
        "complete_total_case_count": len(all_rows) == FINAL_EXPECTED_CASES,
        "complete_rolling_fit_count": sum(
            int(row["train_training_window_count"]) for row in all_rows
        )
        == FINAL_EXPECTED_FITS,
        "all_commands_use_frozen_holdout": len(logs) == FINAL_EXPECTED_CASES
        and all(
            path.is_file()
            and f"--timerange {FINAL_HOLDOUT_TIMERANGE}" in path.read_text(encoding="utf-8")
            for path in logs
        ),
        "all_configs_use_15m": all(item["timeframe"] == FINAL_TIMEFRAME for item in generated),
        "all_prediction_metrics_finite": all(
            math.isfinite(float(row[key]))
            for row in all_rows
            for key in ("oos_mse", "oos_mae", "oos_r2", "trading_sharpe")
        ),
        "complete_prediction_test_count": len(prediction_tests)
        == FINAL_EXPECTED_RFF_CASES
        + 3 * (len(PHASE12_BASELINES) - 1 + len(PHASE15_CONTROL_BASELINES)),
        "prediction_dates_unique_and_accepted": all(
            row["duplicate_prediction_dates"] == 0 and row["all_predictions_accepted"]
            for row in prediction_tests
        ),
        "complete_cost_row_count": len(cost_rows) == FINAL_EXPECTED_COST_ROWS,
        "reference_cost_reproduces_native": len(reference_costs) == FINAL_EXPECTED_ECONOMIC_CASES
        and all(
            float(row["native_max_trade_pnl_error"]) <= PHASE16_NATIVE_REPRODUCTION_TOLERANCE
            and float(row["native_total_pnl_error"]) <= PHASE16_NATIVE_REPRODUCTION_TOLERANCE
            for row in reference_costs
        ),
        "performance_excluded_from_integrity_gate": True,
        "holdout_opened_only_after_protocol_freeze": True,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "expected_case_count": FINAL_EXPECTED_CASES,
        "observed_case_count": len(all_rows),
        "expected_rolling_fit_count": FINAL_EXPECTED_FITS,
        "observed_rolling_fit_count": sum(
            int(row["train_training_window_count"]) for row in all_rows
        ),
    }


def _write_holdout_plot(path: Path, rff_rows: list[dict[str, Any]]) -> bool:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return False
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Native 15m target", "Matched 1h target control"),
    )
    for column, study in enumerate(PHASE15_STUDIES, start=1):
        for asset in FINAL_ASSETS:
            rows = sorted(
                (
                    row
                    for row in rff_rows
                    if row["asset"] == asset.alias
                    and row["study"] == study.name
                    and int(row["seed"]) == PHASE10_SEEDS[0]
                ),
                key=lambda row: float(row["actual_pn_ratio"]),
            )
            figure.add_trace(
                go.Scatter(
                    x=[row["actual_pn_ratio"] for row in rows],
                    y=[float(row["oos_mse"]) / float(row["zero_mse"]) for row in rows],
                    mode="lines+markers",
                    name=asset.label,
                    legendgroup=asset.alias,
                    showlegend=column == 1,
                ),
                row=1,
                col=column,
            )
        figure.update_xaxes(type="log", title_text="P/N", row=1, col=column)
        figure.update_yaxes(type="log", title_text="OOS MSE / zero", row=1, col=column)
        figure.add_hline(y=1.0, line_dash="dash", row=1, col=column)
    figure.update_layout(
        title="Frozen 2026 YTD final holdout",
        template="plotly_white",
        height=650,
        width=1450,
    )
    figure.write_html(path, include_plotlyjs="cdn")
    return True


def run_final_holdout(config: FinalHoldoutConfig) -> dict[str, Any]:
    protocol = validate_frozen_protocol(config)
    preparation = _load_passed_preparation(config)
    config.output_directory.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ") + "-final-holdout"
    rff_results: list[dict[str, Any]] = []
    rff_rows: list[dict[str, Any]] = []
    baseline_results: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    checkpoints = []
    for asset in FINAL_ASSETS:
        for study in PHASE15_STUDIES:
            cell_run_id = f"{run_id}-{asset.alias}-{study.name}"
            effective_n = int(preparation["effective_n"][study.name])
            cell_root = config.output_directory / "runs" / asset.alias / study.name
            rff_engine = _phase10_config(
                config,
                asset,
                study,
                effective_n,
                cell_root / "rff",
            )
            for seed, ratios in _rff_tasks(study):
                substudy = _run_substudy(
                    rff_engine,
                    "market_rff",
                    seed,
                    ratios,
                    cell_run_id,
                )
                for result, row in zip(
                    substudy["results"],
                    flatten_substudies([substudy]),
                    strict=True,
                ):
                    rff_results.append(result)
                    rff_rows.append(_tag_row(row, asset, study))
                    cases.append(_case_record(result, asset, study, "rff", "market_rff"))
            baseline_engine = _phase12_config(
                config,
                asset,
                study,
                effective_n,
                cell_root / "baselines",
            )
            results, rows = _run_baselines(baseline_engine, cell_run_id)
            for result, row in zip(results, rows, strict=True):
                baseline_results.append(result)
                baseline_rows.append(_tag_row(row, asset, study))
                cases.append(_case_record(result, asset, study, "baseline", result["baseline"]))
            checkpoints.append(
                {
                    "asset": asset.alias,
                    "study": study.name,
                    "completed_at_utc": datetime.now(tz=UTC).isoformat(),
                }
            )
            (config.output_directory / "checkpoint.json").write_text(
                json.dumps(checkpoints, indent=2),
                encoding="utf-8",
            )
    comparisons = compare_holdout_predictions(rff_rows, baseline_rows)
    prediction_tests = evaluate_prediction_significance(config, cases)
    cost_rows = evaluate_costs(cases)
    costs = _cost_summary(cost_rows)
    gate = evaluate_final_gate(
        rff_results,
        rff_rows,
        baseline_results,
        baseline_rows,
        prediction_tests,
        cost_rows,
    )
    rff_path = config.output_directory / "rff_results.csv"
    baseline_path = config.output_directory / "baseline_results.csv"
    curves_path = config.output_directory / "curve_assessments.csv"
    prediction_tests_path = config.output_directory / "prediction_tests.csv"
    costs_path = config.output_directory / "cost_results.csv"
    plot_path = config.output_directory / "final_holdout.html"
    summary_path = config.output_directory / "summary.json"
    _write_csv(rff_path, rff_rows)
    _write_csv(baseline_path, baseline_rows)
    _write_csv(curves_path, comparisons["curve_assessments"])
    _write_csv(prediction_tests_path, prediction_tests)
    _write_csv(costs_path, cost_rows)
    plot_written = _write_holdout_plot(plot_path, rff_rows)
    significant_nonzero = [row for row in prediction_tests if row["model_name"] != "zero_return"]
    summary = {
        "experiment": "supervised_double_descent_final_holdout",
        "stage": "complete",
        "protocol_version": FINAL_HOLDOUT_PROTOCOL_VERSION,
        "run_id": run_id,
        "protocol_commit": _git_output("rev-parse", "HEAD"),
        "protocol_parent_commit": protocol["protocol_parent_commit"],
        "design": {
            "timerange": FINAL_HOLDOUT_TIMERANGE,
            "label": "2026 YTD final holdout",
            "full_calendar_year": False,
            "prequential_rolling_evaluation": True,
            "parameters_selected_on_holdout": False,
            "trading_used_for_prediction_inference": False,
            "holdout_used": True,
            "one_shot_protocol": True,
        },
        "comparisons": comparisons,
        "statistical_tests": {
            "method": "two-sided Bartlett-HAC MSE differential versus zero",
            "lag_candles": FINAL_HAC_LAG_CANDLES,
            "fdr_alpha": FINAL_FDR_ALPHA,
            "nonzero_prediction_case_count": len(significant_nonzero),
            "significantly_better_than_zero_count": sum(
                row["significantly_better_than_zero_fdr"] for row in significant_nonzero
            ),
            "significantly_worse_than_zero_count": sum(
                row["significantly_worse_than_zero_fdr"] for row in significant_nonzero
            ),
        },
        "cost_summary": costs,
        "gate": gate,
        "compute": {
            "rff_case_count": len(rff_rows),
            "baseline_case_count": len(baseline_rows),
            "total_case_count": len(rff_rows) + len(baseline_rows),
            "rolling_model_fit_count": sum(
                int(row["train_training_window_count"]) for row in [*rff_rows, *baseline_rows]
            ),
            "summed_freqtrade_wall_seconds": sum(
                float(row["wall_seconds"]) for row in [*rff_rows, *baseline_rows]
            ),
            "maximum_peak_vram_mib": max(float(row["train_peak_vram_mib"]) for row in rff_rows),
        },
        "limitations": {
            "partial_year": "2026 YTD ends before 2026-08-01; it is not a full-year holdout",
            "prequential": (
                "Later rolling fits may train on earlier 2026 observations, but every prediction "
                "remains strictly chronological and the protocol is never retuned."
            ),
            "gold_proxy": "PAXG perpetual is tokenized gold, not XAU spot or COMEX futures",
            "common_exchange": "all instruments use Binance",
            "cost_model": (
                "deterministic fees/slippage/funding sensitivity; no order-book market impact"
            ),
            "phase19": protocol["phase19_quarantine"],
        },
        "artifacts": {
            "summary": str(summary_path),
            "rff_results": str(rff_path),
            "baseline_results": str(baseline_path),
            "curve_assessments": str(curves_path),
            "prediction_tests": str(prediction_tests_path),
            "cost_results": str(costs_path),
            "interactive_plot": str(plot_path) if plot_written else None,
        },
    }
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return summary
