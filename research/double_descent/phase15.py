"""Phase 15: frozen 15-minute frequency robustness across BTC, ETH, and gold proxy."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research.double_descent.phase3 import (
    Phase3Config,
    audit_data_coverage,
    run_phase3,
)
from research.double_descent.phase4 import _load_evaluation_market_data, _parse_timerange
from research.double_descent.phase8 import assess_flat_curve
from research.double_descent.phase10 import (
    PHASE10_GAMMA,
    PHASE10_MAP_RATIOS,
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
from research.double_descent.phase12 import (
    _phase4_config as _phase12_phase4_config,
)
from research.double_descent.phase12 import (
    _recover_case as _recover_baseline_case,
)
from research.double_descent.phase12 import (
    _run_case as _run_baseline_case,
)
from research.double_descent.phase12 import (
    flatten_results as flatten_baseline_results,
)


PHASE15_TIMEFRAME = "15m"
PHASE15_TIMERANGE = "20250101-20260101"
PHASE15_BENCHMARK_TIMERANGE = "20250101-20250201"
PHASE15_DATA_TIMERANGE = "20240924-20260101"
PHASE15_STRATEGY = "Phase15TimeframeStrategy"
# One extra candle prevents the first rolling window from losing a training row at
# Freqtrade's inclusive startup boundary; the feature history itself needs 800.
PHASE15_STARTUP_CANDLES = 801
PHASE15_INDICATOR_PERIODS = (56,)
PHASE15_BENCHMARK_RATIOS = (0.1, 1.0, 50.0)
PHASE15_CONTROL_RATIOS = (0.1, 1.0, 1.02, 5.0, 50.0)
PHASE15_CONTROL_BASELINES = (
    "zero_return",
    "market_ols",
    "market_ridge",
    "momentum_1h",
    "momentum_24h",
)
PHASE15_EXPECTED_N = {"native_15m": 8_639, "matched_1h_control": 8_636}
PHASE15_EFFECTIVE_N_TOLERANCE = 1
PHASE15_EXPECTED_FULL_CASE_COUNT = 117
PHASE15_EXPECTED_FULL_FIT_COUNT = 1_287


@dataclass(frozen=True)
class AssetSpec:
    alias: str
    pair: str
    label: str
    asset_class: str
    timerange: str
    minimum_training_windows: int
    expected_training_windows: int
    comparison_scope: str


@dataclass(frozen=True)
class StudySpec:
    name: str
    label_period_candles: int
    target_minutes: int
    role: str


PHASE15_ASSETS = (
    AssetSpec(
        "btc",
        "BTC/USDT:USDT",
        "BTC",
        "crypto",
        PHASE15_TIMERANGE,
        10,
        13,
        "full_2025_core",
    ),
    AssetSpec(
        "eth",
        "ETH/USDT:USDT",
        "ETH",
        "crypto",
        PHASE15_TIMERANGE,
        10,
        13,
        "full_2025_core",
    ),
    AssetSpec(
        "gold",
        "PAXG/USDT:USDT",
        "PAXG gold proxy",
        "tokenized_gold_proxy",
        "20250704-20260101",
        6,
        7,
        "coverage_limited_diagnostic",
    ),
)
PHASE15_STUDIES = (
    StudySpec("native_15m", 1, 15, "primary"),
    StudySpec("matched_1h_control", 4, 60, "diagnostic_control"),
)


@dataclass(frozen=True)
class Phase15Config:
    data_directory: Path
    output_directory: Path = Path("user_data/research_results/double_descent/phase15")
    phase14_summary: Path = Path("user_data/research_results/double_descent/phase14/summary.json")
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
    cuda_python_executable: str = ""
    asset_aliases: tuple[str, ...] = tuple(asset.alias for asset in PHASE15_ASSETS)
    study_names: tuple[str, ...] = tuple(study.name for study in PHASE15_STUDIES)
    timeframe: str = PHASE15_TIMEFRAME
    timerange: str = PHASE15_TIMERANGE
    holdout_start: str = "20260101"
    train_period_days: int = 90
    backtest_period_days: int = 30
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
    minimum_training_windows: int = 10
    subprocess_timeout_seconds: int = 3_600
    benchmark_max_case_seconds: float = 3_600.0
    benchmark_vram_fraction_limit: float = 0.90
    resume: bool = True
    smoke_test: bool = False
    strategy_directory: Path = Path("research/double_descent/freqai")
    model_directory: Path = Path("research/double_descent/freqai")
    models_directory: Path = Path("user_data/models")

    @property
    def preparation_path(self) -> Path:
        return self.output_directory / "preparation.json"

    @property
    def benchmark_path(self) -> Path:
        return self.output_directory / "benchmark" / "summary.json"

    def validate(self) -> None:
        if not self.data_directory.is_dir():
            raise FileNotFoundError(f"data directory does not exist: {self.data_directory}")
        if not Path(self.python_executable).is_file():
            raise FileNotFoundError(f"Freqtrade Python does not exist: {self.python_executable}")
        if not Path(self.cuda_python_executable).is_file():
            raise FileNotFoundError(f"CUDA Python does not exist: {self.cuda_python_executable}")
        self._validate_reference()
        self._validate_design()

    def _validate_reference(self) -> None:
        if not self.phase14_summary.is_file():
            raise FileNotFoundError("Phase 15 requires the completed Phase 14 summary")
        summary = json.loads(self.phase14_summary.read_text(encoding="utf-8"))
        if summary.get("phase") != 14 or summary.get("gate", {}).get("passed") is not True:
            raise ValueError("Phase 14 did not pass its integrity gate")
        if summary.get("design", {}).get("holdout_used") is not False:
            raise ValueError("Phase 14 does not certify a sealed holdout")

    def _validate_design(self) -> None:
        if self.timeframe != PHASE15_TIMEFRAME or self.timerange != PHASE15_TIMERANGE:
            raise ValueError("Phase 15 is frozen to the 2025 15-minute development interval")
        if self.train_period_days != 90 or self.backtest_period_days != 30:
            raise ValueError("Phase 15 freezes the 90-day/30-day rolling design")
        if self.gamma != 0.5 or self.ridge != 0 or self.ridge_alpha != 1.0:
            raise ValueError("Phase 15 freezes gamma=0.5 and the prior ridge settings")
        if self.rcond != 1e-12 or self.dtype != "float64" or self.fee != 0.001:
            raise ValueError("Phase 15 freezes numerical precision and 0.1% fee per side")
        if not 0 < self.benchmark_vram_fraction_limit < 1:
            raise ValueError("benchmark VRAM fraction must be in (0, 1)")
        if not self.smoke_test:
            frozen = (
                (self.asset_aliases, tuple(asset.alias for asset in PHASE15_ASSETS), "assets"),
                (self.study_names, tuple(study.name for study in PHASE15_STUDIES), "studies"),
                (self.map_ratios, PHASE10_MAP_RATIOS, "P/N map"),
                (self.robustness_ratios, PHASE10_ROBUSTNESS_RATIOS, "robustness map"),
                (self.seeds, PHASE10_SEEDS, "seeds"),
                (self.baselines, PHASE12_BASELINES, "baselines"),
                (self.control_ratios, PHASE15_CONTROL_RATIOS, "control P/N map"),
                (self.control_baselines, PHASE15_CONTROL_BASELINES, "control baselines"),
            )
            for actual, expected, label in frozen:
                if actual != expected:
                    raise ValueError(f"the full Phase 15 run freezes the {label}")
        _, end = _parse_timerange(self.timerange)
        holdout = datetime.strptime(self.holdout_start, "%Y%m%d").replace(tzinfo=UTC)
        if end > holdout:
            raise ValueError("Phase 15 may not enter the sealed holdout")


def selected_assets(config: Phase15Config) -> tuple[AssetSpec, ...]:
    lookup = {asset.alias: asset for asset in PHASE15_ASSETS}
    unknown = set(config.asset_aliases) - set(lookup)
    if unknown:
        raise ValueError(f"unknown Phase 15 assets: {sorted(unknown)}")
    return tuple(lookup[alias] for alias in config.asset_aliases)


def selected_studies(config: Phase15Config) -> tuple[StudySpec, ...]:
    lookup = {study.name: study for study in PHASE15_STUDIES}
    unknown = set(config.study_names) - set(lookup)
    if unknown:
        raise ValueError(f"unknown Phase 15 studies: {sorted(unknown)}")
    return tuple(lookup[name] for name in config.study_names)


def build_download_command(config: Phase15Config) -> list[str]:
    return [
        config.python_executable,
        "-m",
        "freqtrade",
        "download-data",
        "--exchange",
        "binance",
        "--pairs",
        *(asset.pair for asset in selected_assets(config)),
        "--timeframes",
        config.timeframe,
        "--timerange",
        PHASE15_DATA_TIMERANGE,
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


def download_phase15_data(config: Phase15Config) -> None:
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
        raise RuntimeError(f"Phase 15 data download failed with code {completed.returncode}")


def _phase3_config(
    config: Phase15Config,
    asset: AssetSpec,
    study: StudySpec,
    output_directory: Path,
    timerange: str | None = None,
    minimum_windows: int | None = None,
) -> Phase3Config:
    return Phase3Config(
        data_directory=config.data_directory,
        output_directory=output_directory,
        python_executable=config.python_executable,
        pair=asset.pair,
        timeframe=config.timeframe,
        timerange=timerange or asset.timerange,
        train_periods_days=(config.train_period_days,),
        backtest_period_days=config.backtest_period_days,
        startup_candles=PHASE15_STARTUP_CANDLES,
        label_period_candles=study.label_period_candles,
        indicator_periods_candles=PHASE15_INDICATOR_PERIODS,
        minimum_windows_per_period=(
            minimum_windows if minimum_windows is not None else asset.minimum_training_windows
        ),
        subprocess_timeout_seconds=config.subprocess_timeout_seconds,
        strategy_name=PHASE15_STRATEGY,
        identifier_prefix=f"double-descent-phase15-n-{study.name}",
        allow_timeframe_variation=True,
        strategy_directory=config.strategy_directory,
        model_directory=config.model_directory,
    )


def _read_or_measure_n(
    config: Phase15Config,
    asset: AssetSpec,
    study: StudySpec,
) -> dict[str, Any]:
    output = config.output_directory / "preparation" / "effective_n" / asset.alias / study.name
    summary_path = output / "summary.json"
    if config.resume and summary_path.is_file():
        cached = json.loads(summary_path.read_text(encoding="utf-8"))
        cached_config = cached.get("config", {})
        if (
            cached.get("gate", {}).get("passed") is True
            and cached_config.get("pair") == asset.pair
            and cached_config.get("timeframe") == config.timeframe
            and cached_config.get("timerange") == asset.timerange
            and cached_config.get("startup_candles") == PHASE15_STARTUP_CANDLES
            and cached_config.get("label_period_candles") == study.label_period_candles
        ):
            return cached
    return run_phase3(_phase3_config(config, asset, study, output))


def prepare_phase15(config: Phase15Config) -> dict[str, Any]:
    config.validate()
    config.output_directory.mkdir(parents=True, exist_ok=True)
    audit_study = PHASE15_STUDIES[0]
    data_audits = {
        asset.alias: audit_data_coverage(
            _phase3_config(
                config,
                asset,
                audit_study,
                config.output_directory / "preparation" / "audit" / asset.alias,
            )
        )
        for asset in selected_assets(config)
    }
    measurements: dict[str, dict[str, Any]] = {}
    effective_n_by_asset: dict[str, dict[str, int | None]] = {}
    for asset in selected_assets(config):
        measurements[asset.alias] = {}
        effective_n_by_asset[asset.alias] = {}
        for study in selected_studies(config):
            summary = _read_or_measure_n(config, asset, study)
            measurements[asset.alias][study.name] = summary
            aggregates = summary.get("aggregate", [])
            effective_n_by_asset[asset.alias][study.name] = (
                int(aggregates[0]["effective_n_median"]) if len(aggregates) == 1 else None
            )
    effective_n = {}
    for study in selected_studies(config):
        values = {
            effective_n_by_asset[asset.alias][study.name] for asset in selected_assets(config)
        }
        effective_n[study.name] = values.pop() if len(values) == 1 else None
    measurement_rows = [
        (asset, study, measurements[asset.alias][study.name])
        for asset in selected_assets(config)
        for study in selected_studies(config)
    ]
    checks = {
        "all_asset_data_audits_passed": bool(data_audits)
        and all(audit.get("passed") is True for audit in data_audits.values()),
        "all_effective_n_measurements_passed": bool(measurement_rows)
        and all(
            summary.get("gate", {}).get("passed") is True for _, _, summary in measurement_rows
        ),
        "effective_n_matches_predeclared_values": all(
            effective_n_by_asset[asset.alias][study.name] == PHASE15_EXPECTED_N[study.name]
            for asset, study, _ in measurement_rows
        ),
        "effective_n_variation_within_tolerance": all(
            abs(int(aggregate[key]) - PHASE15_EXPECTED_N[study.name])
            <= PHASE15_EFFECTIVE_N_TOLERANCE
            for _, study, summary in measurement_rows
            for aggregate in summary.get("aggregate", [])
            for key in ("effective_n_min", "effective_n_max")
        ),
        "feature_dimension_is_25": all(
            summary.get("gate", {}).get("observed_feature_counts") == [25]
            for _, _, summary in measurement_rows
        ),
        "rolling_window_counts_match_predeclaration": all(
            summary.get("aggregate", [{}])[0].get("window_count") == asset.expected_training_windows
            for asset, _, summary in measurement_rows
        ),
        "gold_is_labelled_as_tokenized_proxy": any(
            asset.alias == "gold" and asset.asset_class == "tokenized_gold_proxy"
            for asset in selected_assets(config)
        ),
        "holdout_was_not_used": True,
    }
    summary = {
        "phase": 15,
        "stage": "preparation",
        "objective": "Verify 15m data coverage and measure effective N before selecting P",
        "data_audits": data_audits,
        "effective_n": effective_n,
        "effective_n_by_asset": effective_n_by_asset,
        "measurements": {
            asset.alias: {
                study.name: {
                    "summary": measurements[asset.alias][study.name]
                    .get("artifacts", {})
                    .get("summary"),
                    "gate": measurements[asset.alias][study.name].get("gate"),
                    "aggregate": measurements[asset.alias][study.name].get("aggregate"),
                }
                for study in selected_studies(config)
            }
            for asset in selected_assets(config)
        },
        "gate": {"passed": all(checks.values()), "checks": checks},
        "design": {
            "timeframe": config.timeframe,
            "clock_time_matched_features": True,
            "holdout_used": False,
            "asset_periods": {
                asset.alias: {
                    "timerange": asset.timerange,
                    "expected_training_windows": asset.expected_training_windows,
                    "comparison_scope": asset.comparison_scope,
                }
                for asset in selected_assets(config)
            },
            "gold_instrument": "PAXG/USDT:USDT Binance perpetual futures",
            "gold_proxy_limitation": "PAXG is tokenized gold, not XAU spot or COMEX futures",
        },
    }
    config.preparation_path.write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return summary


def _load_passed_preparation(config: Phase15Config) -> dict[str, Any]:
    if not config.preparation_path.is_file():
        raise FileNotFoundError("Run Phase 15 preparation before the benchmark")
    summary = json.loads(config.preparation_path.read_text(encoding="utf-8"))
    if summary.get("gate", {}).get("passed") is not True:
        raise ValueError("Phase 15 preparation did not pass")
    return summary


def _cuda_device_info(config: Phase15Config) -> dict[str, Any]:
    code = (
        "import json, torch; p=torch.cuda.get_device_properties(0); "
        "print(json.dumps({'name': p.name, 'total_vram_mib': p.total_memory / 2**20, "
        "'torch': torch.__version__, 'cuda': torch.version.cuda}))"
    )
    completed = subprocess.run(
        [config.cuda_python_executable, "-c", code],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"CUDA device query failed: {completed.stderr[-1000:]}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _phase10_config(
    config: Phase15Config,
    asset: AssetSpec,
    study: StudySpec,
    effective_n: int,
    output_directory: Path,
    timerange: str | None = None,
    minimum_windows: int | None = None,
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
        timerange=timerange or asset.timerange,
        holdout_start=config.holdout_start,
        train_period_days=config.train_period_days,
        backtest_period_days=config.backtest_period_days,
        effective_n=effective_n,
        map_ratios=config.map_ratios,
        robustness_ratios=config.robustness_ratios,
        seeds=config.seeds,
        gamma=config.gamma,
        ridge=config.ridge,
        rcond=config.rcond,
        dtype=config.dtype,
        chunk_size=config.chunk_size,
        fee=config.fee,
        minimum_training_windows=(
            minimum_windows if minimum_windows is not None else asset.minimum_training_windows
        ),
        effective_n_tolerance=PHASE15_EFFECTIVE_N_TOLERANCE,
        minimum_seed_count=1 if config.smoke_test else 3,
        subprocess_timeout_seconds=config.subprocess_timeout_seconds,
        resume=config.resume,
        smoke_test=True,
        allow_external_design=True,
        label_period_candles=study.label_period_candles,
        startup_candles=PHASE15_STARTUP_CANDLES,
        indicator_periods_candles=PHASE15_INDICATOR_PERIODS,
        strategy_name=PHASE15_STRATEGY,
        strategy_directory=config.strategy_directory,
        model_directory=config.model_directory,
        models_directory=config.models_directory,
    )


def run_phase15_benchmark(config: Phase15Config) -> dict[str, Any]:
    config.validate()
    preparation = _load_passed_preparation(config)
    asset = PHASE15_ASSETS[0]
    study = PHASE15_STUDIES[0]
    effective_n = int(preparation["effective_n"][study.name])
    output = config.output_directory / "benchmark"
    output.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ") + "-benchmark"
    engine = _phase10_config(
        config,
        asset,
        study,
        effective_n,
        output,
        timerange=PHASE15_BENCHMARK_TIMERANGE,
        minimum_windows=1,
    )
    substudy = _run_substudy(
        engine,
        "market_rff",
        config.seeds[0],
        PHASE15_BENCHMARK_RATIOS,
        run_id,
    )
    benchmark_effective_n_values = [
        int(value)
        for result in substudy.get("results", [])
        for value in result.get("training", {}).get("effective_n_values", [])
    ]
    rows = flatten_substudies([substudy])
    device = _cuda_device_info(config)
    maximum_vram = max((float(row["train_peak_vram_mib"]) for row in rows), default=math.inf)
    maximum_seconds = max((float(row["wall_seconds"]) for row in rows), default=math.inf)
    checks = {
        "all_benchmark_cases_passed": substudy.get("gate", {}).get("passed") is True,
        "all_three_compute_boundaries_observed": len(rows) == len(PHASE15_BENCHMARK_RATIOS),
        "effective_n_matches_preparation": bool(benchmark_effective_n_values)
        and all(
            abs(value - effective_n) <= PHASE15_EFFECTIVE_N_TOLERANCE
            for value in benchmark_effective_n_values
        ),
        "maximum_case_within_timeout": maximum_seconds <= config.benchmark_max_case_seconds,
        "peak_vram_within_budget": maximum_vram
        <= float(device["total_vram_mib"]) * config.benchmark_vram_fraction_limit,
        "cuda_float64_used": bool(rows)
        and all(float(row["train_peak_vram_mib"]) > 0 for row in rows),
    }
    summary = {
        "phase": 15,
        "stage": "benchmark",
        "run_id": run_id,
        "design": {
            "asset": asset.pair,
            "study": study.name,
            "timerange": PHASE15_BENCHMARK_TIMERANGE,
            "ratios": PHASE15_BENCHMARK_RATIOS,
            "effective_n": effective_n,
            "maximum_feature_count": round(max(PHASE15_BENCHMARK_RATIOS) * effective_n),
            "holdout_used": False,
        },
        "device": device,
        "rows": rows,
        "gate": {"passed": all(checks.values()), "checks": checks},
        "estimates": {
            "maximum_case_wall_seconds": maximum_seconds,
            "maximum_peak_vram_mib": maximum_vram,
            "full_case_count": PHASE15_EXPECTED_FULL_CASE_COUNT,
            "full_rolling_fit_count": PHASE15_EXPECTED_FULL_FIT_COUNT,
        },
    }
    config.benchmark_path.write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _write_csv(output / "benchmark_results.csv", rows)
    return summary


def _load_passed_benchmark(config: Phase15Config) -> dict[str, Any]:
    if not config.benchmark_path.is_file():
        raise FileNotFoundError("Run the Phase 15 benchmark before the full sweep")
    summary = json.loads(config.benchmark_path.read_text(encoding="utf-8"))
    if summary.get("gate", {}).get("passed") is not True:
        raise ValueError("Phase 15 benchmark did not pass its compute gate")
    return summary


def _rff_tasks(
    config: Phase15Config,
    study: StudySpec,
) -> list[tuple[int, tuple[float, ...]]]:
    if config.smoke_test:
        return [(config.seeds[0], (0.1, 1.0, 5.0))]
    if study.role == "diagnostic_control":
        return [(config.seeds[0], config.control_ratios)]
    return [
        (seed, config.map_ratios if index == 0 else config.robustness_ratios)
        for index, seed in enumerate(config.seeds)
    ]


def _baseline_names(config: Phase15Config, study: StudySpec) -> tuple[str, ...]:
    if config.smoke_test:
        return ("zero_return", "market_ols", "market_ridge")
    return config.control_baselines if study.role == "diagnostic_control" else config.baselines


def _phase12_config(
    config: Phase15Config,
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
        holdout_start=config.holdout_start,
        train_period_days=config.train_period_days,
        backtest_period_days=config.backtest_period_days,
        effective_n=effective_n,
        baselines=_baseline_names(config, study),
        ridge_alpha=config.ridge_alpha,
        fee=config.fee,
        minimum_training_windows=asset.minimum_training_windows,
        effective_n_tolerance=PHASE15_EFFECTIVE_N_TOLERANCE,
        subprocess_timeout_seconds=config.subprocess_timeout_seconds,
        resume=config.resume,
        smoke_test=True,
        allow_external_design=True,
        label_period_candles=study.label_period_candles,
        startup_candles=PHASE15_STARTUP_CANDLES,
        indicator_periods_candles=PHASE15_INDICATOR_PERIODS,
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
    phase4 = _phase12_phase4_config(engine)
    phase4.validate()
    market_data = _load_evaluation_market_data(phase4)
    results = []
    for baseline in engine.baselines:
        result = (
            _recover_baseline_case(engine, baseline, phase4, market_data) if engine.resume else None
        )
        if result is None:
            result = _run_baseline_case(engine, baseline, phase4, run_id, market_data)
        results.append(result)
    return results, flatten_baseline_results(results)


def _tag_rows(
    rows: list[dict[str, Any]],
    asset: AssetSpec,
    study: StudySpec,
) -> list[dict[str, Any]]:
    return [
        {
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
        for row in rows
    ]


def _curve_assessments(
    config: Phase15Config,
    rff_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    assessments = []
    for asset in selected_assets(config):
        for study in selected_studies(config):
            for seed, ratios in _rff_tasks(config, study):
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


def compare_frequency_results(
    config: Phase15Config,
    rff_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    curves = _curve_assessments(config, rff_rows)
    cells = []
    for asset in selected_assets(config):
        for study in selected_studies(config):
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
    primary_cells = [row for row in cells if row["study"] == "native_15m"]
    return {
        "primary_metric": "chronological OOS MSE normalized by each asset's zero forecast",
        "curve_assessments": curves,
        "asset_study_comparisons": cells,
        "primary_shape_replication_asset_count": sum(
            row["double_descent_seed_count"] >= math.ceil(row["seed_count"] / 2)
            for row in primary_cells
        ),
        "primary_useful_replication_asset_count": sum(
            row["useful_second_descent_seed_count"] >= math.ceil(row["seed_count"] / 2)
            for row in primary_cells
        ),
        "primary_asset_count": len(primary_cells),
        "trading_excluded_from_frequency_decision": True,
    }


def _input_dimension_is_25(results: list[dict[str, Any]]) -> bool:
    return bool(results) and all(
        result.get("training", {}).get("input_feature_counts") == [25]
        for result in results
    )


def evaluate_phase15_gate(
    config: Phase15Config,
    preparation: dict[str, Any],
    benchmark: dict[str, Any] | None,
    raw_rff: list[dict[str, Any]],
    rff_rows: list[dict[str, Any]],
    raw_baselines: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    all_rows = [*rff_rows, *baseline_rows]
    expected_rff = sum(
        sum(len(ratios) for _, ratios in _rff_tasks(config, study))
        for study in selected_studies(config)
    ) * len(selected_assets(config))
    expected_baselines = sum(
        len(_baseline_names(config, study)) for study in selected_studies(config)
    ) * len(selected_assets(config))
    expected_total = expected_rff + expected_baselines
    cases_per_asset = sum(
        sum(len(ratios) for _, ratios in _rff_tasks(config, study))
        + len(_baseline_names(config, study))
        for study in selected_studies(config)
    )
    expected_rolling_fits = sum(
        asset.expected_training_windows * cases_per_asset for asset in selected_assets(config)
    )
    observed_rolling_fits = sum(
        int(row["train_training_window_count"]) for row in all_rows
    )
    all_results = [*raw_rff, *raw_baselines]
    generated_configs = [Path(result["artifacts"]["config"]) for result in all_results]
    configurations = [
        json.loads(path.read_text(encoding="utf-8")) for path in generated_configs if path.is_file()
    ]
    checks = {
        "preparation_passed": preparation.get("gate", {}).get("passed") is True,
        "benchmark_passed": config.smoke_test
        or (benchmark is not None and benchmark.get("gate", {}).get("passed") is True),
        "all_rff_cases_passed": bool(raw_rff) and all(result["success"] for result in raw_rff),
        "all_baseline_cases_passed": bool(raw_baselines)
        and all(result["success"] for result in raw_baselines),
        "complete_predeclared_rff_case_count": len(rff_rows) == expected_rff,
        "complete_predeclared_baseline_case_count": len(baseline_rows) == expected_baselines,
        "complete_predeclared_total_case_count": len(all_rows) == expected_total,
        "complete_predeclared_rolling_fit_count": observed_rolling_fits
        == expected_rolling_fits,
        "all_generated_configs_are_15m": len(configurations) == expected_total
        and all(item["timeframe"] == config.timeframe for item in configurations),
        "all_generated_pairs_match_cells": len(configurations) == expected_total
        and {item["exchange"]["pair_whitelist"][0] for item in configurations}
        == {asset.pair for asset in selected_assets(config)},
        "label_horizons_match_predeclaration": len(configurations) == expected_total
        and {
            item["freqai"]["feature_parameters"]["label_period_candles"] for item in configurations
        }
        == {study.label_period_candles for study in selected_studies(config)},
        "effective_n_matches_preparation": bool(all_results)
        and all(
            bool(result["training"].get("effective_n_values"))
            and all(
                abs(value - int(preparation["effective_n"][row["study"]]))
                <= PHASE15_EFFECTIVE_N_TOLERANCE
                for value in result["training"]["effective_n_values"]
            )
            for result, row in zip(all_results, all_rows, strict=True)
        ),
        "input_dimension_is_25": _input_dimension_is_25(all_results),
        "all_prediction_metrics_finite": bool(all_rows)
        and all(
            math.isfinite(float(row[key]))
            for row in all_rows
            for key in ("oos_mse", "oos_mae", "oos_r2", "trading_sharpe")
        ),
        "cuda_float64_used_for_rff": bool(rff_rows)
        and all(float(row["train_peak_vram_mib"]) > 0 for row in rff_rows),
        "fee_preserved": config.fee == 0.001,
        "trading_excluded_from_inference": True,
        "holdout_was_not_used": all(
            _parse_timerange(asset.timerange)[1]
            <= datetime.strptime(config.holdout_start, "%Y%m%d").replace(tzinfo=UTC)
            for asset in selected_assets(config)
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "expected_rff_case_count": expected_rff,
        "observed_rff_case_count": len(rff_rows),
        "expected_baseline_case_count": expected_baselines,
        "observed_baseline_case_count": len(baseline_rows),
        "expected_total_case_count": expected_total,
        "observed_total_case_count": len(all_rows),
        "expected_rolling_fit_count": expected_rolling_fits,
        "observed_rolling_fit_count": observed_rolling_fits,
    }


def _write_plot(
    path: Path,
    config: Phase15Config,
    rff_rows: list[dict[str, Any]],
) -> bool:
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
    for column, study in enumerate(selected_studies(config), start=1):
        for asset in selected_assets(config):
            rows = sorted(
                (
                    row
                    for row in rff_rows
                    if row["asset"] == asset.alias
                    and row["study"] == study.name
                    and int(row["seed"]) == config.seeds[0]
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
        title="Phase 15 - 15m frequency robustness across BTC, ETH, and PAXG",
        template="plotly_white",
        height=650,
        width=1450,
    )
    figure.write_html(path, include_plotlyjs="cdn")
    return True


def run_phase15(config: Phase15Config) -> dict[str, Any]:
    config.validate()
    preparation = _load_passed_preparation(config)
    benchmark = None if config.smoke_test else _load_passed_benchmark(config)
    config.output_directory.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    raw_rff = []
    rff_rows = []
    raw_baselines = []
    baseline_rows = []
    checkpoints = []
    for asset in selected_assets(config):
        for study in selected_studies(config):
            cell_id = f"{asset.alias}-{study.name}"
            cell_run_id = f"{run_id}-{cell_id}"
            effective_n = int(preparation["effective_n"][study.name])
            cell_root = config.output_directory / "runs" / asset.alias / study.name
            rff_engine = _phase10_config(
                config,
                asset,
                study,
                effective_n,
                cell_root / "rff",
            )
            for seed, ratios in _rff_tasks(config, study):
                substudy = _run_substudy(
                    rff_engine,
                    "market_rff",
                    seed,
                    ratios,
                    cell_run_id,
                )
                raw_rff.extend(substudy["results"])
                rff_rows.extend(_tag_rows(flatten_substudies([substudy]), asset, study))
            baseline_engine = _phase12_config(
                config,
                asset,
                study,
                effective_n,
                cell_root / "baselines",
            )
            results, rows = _run_baselines(baseline_engine, cell_run_id)
            raw_baselines.extend(results)
            baseline_rows.extend(_tag_rows(rows, asset, study))
            checkpoints.append(
                {
                    "asset": asset.alias,
                    "study": study.name,
                    "rff_rows": len(
                        [
                            row
                            for row in rff_rows
                            if row["asset"] == asset.alias and row["study"] == study.name
                        ]
                    ),
                    "baseline_rows": len(rows),
                }
            )
            (config.output_directory / "checkpoint.json").write_text(
                json.dumps(checkpoints, indent=2),
                encoding="utf-8",
            )
    comparisons = compare_frequency_results(config, rff_rows, baseline_rows)
    gate = evaluate_phase15_gate(
        config,
        preparation,
        benchmark,
        raw_rff,
        rff_rows,
        raw_baselines,
        baseline_rows,
    )
    rff_path = config.output_directory / "rff_results.csv"
    baseline_path = config.output_directory / "baseline_results.csv"
    curve_path = config.output_directory / "curve_assessments.csv"
    plot_path = config.output_directory / "frequency_robustness.html"
    summary_path = config.output_directory / "summary.json"
    _write_csv(rff_path, rff_rows)
    _write_csv(baseline_path, baseline_rows)
    _write_csv(curve_path, comparisons["curve_assessments"])
    plot_written = _write_plot(plot_path, config, rff_rows)
    config_payload = asdict(config)
    for key, value in list(config_payload.items()):
        if isinstance(value, Path):
            config_payload[key] = str(value)
    summary = {
        "phase": 15,
        "stage": "full" if not config.smoke_test else "smoke",
        "objective": "Test frequency robustness across BTC, ETH, and tokenized gold proxy",
        "run_id": run_id,
        "design": {
            "primary": "15m candles predicting the next 15m return",
            "control": "15m candles predicting the next 1h return",
            "clock_time_matched_features": True,
            "parameters_selected_on_15m_data": False,
            "trading_used_for_inference": False,
            "holdout_used": False,
            "asset_periods": {
                asset.alias: {
                    "timerange": asset.timerange,
                    "expected_training_windows": asset.expected_training_windows,
                    "comparison_scope": asset.comparison_scope,
                }
                for asset in selected_assets(config)
            },
        },
        "config": config_payload,
        "preparation": {
            "path": str(config.preparation_path),
            "effective_n": preparation["effective_n"],
            "gate": preparation["gate"],
        },
        "benchmark": (
            {"path": str(config.benchmark_path), "gate": benchmark["gate"]}
            if benchmark is not None
            else None
        ),
        "comparisons": comparisons,
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
            "gold_proxy": "PAXG perpetual is tokenized gold, not XAU spot or COMEX futures",
            "gold_history": (
                "Binance PAXG futures data begins in March 2025, so its OOS arm is limited "
                "to 2025-07-04 through 2025-12-31 and is reported separately from the "
                "full-year BTC/ETH core"
            ),
            "common_exchange": "all instruments use Binance and share exchange-specific effects",
            "development_period": "one 2025 development year; 2026 remains sealed",
            "cost_model": "0.1% fee per side; no separate market-impact model",
            "control_scope": "the one-hour target control uses one seed and a sparse P/N grid",
        },
        "artifacts": {
            "summary": str(summary_path),
            "rff_results": str(rff_path),
            "baseline_results": str(baseline_path),
            "curve_assessments": str(curve_path),
            "interactive_plot": str(plot_path) if plot_written else None,
            "checkpoint": str(config.output_directory / "checkpoint.json"),
        },
    }
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    return summary
