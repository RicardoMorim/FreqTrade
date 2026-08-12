"""Phase 11: shuffled-label negative control for the financial RFF experiment."""

from __future__ import annotations

import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.double_descent.phase3 import Phase3Config, audit_data_coverage
from research.double_descent.phase4 import (
    Phase4Config,
    _finalize_case,
    _flatten_result,
    _load_evaluation_market_data,
    _parse_timerange,
    _read_training_records,
    build_freqtrade_config,
    evaluate_phase4_gate,
    feature_count_for_ratio,
)
from research.double_descent.phase5 import summarize_values
from research.double_descent.phase8 import assess_flat_curve
from research.double_descent.phase10 import (
    PHASE10_GAMMA,
    PHASE10_MAP_RATIOS,
    PHASE10_ROBUSTNESS_RATIOS,
    PHASE10_SEEDS,
    _json_safe,
    _write_csv,
)


PHASE11_MAP_RATIOS = PHASE10_MAP_RATIOS
PHASE11_ROBUSTNESS_RATIOS = PHASE10_ROBUSTNESS_RATIOS
PHASE11_FEATURE_SEEDS = PHASE10_SEEDS
PHASE11_LABEL_SHUFFLE_SEEDS = (316_486_817, 2_823_025_517, 4_239_233_659)
PHASE11_GAMMA = PHASE10_GAMMA
PHASE11_EXPECTED_CASE_COUNT = 21
MAXIMUM_ABSOLUTE_TRAIN_LABEL_CORRELATION = 0.1

AGGREGATE_METRICS = (
    "oos_mse",
    "oos_mae",
    "oos_r2",
    "oos_information_coefficient",
    "oos_spearman_information_coefficient",
    "oos_directional_accuracy",
    "oos_prediction_standard_deviation",
    "train_train_mse_mean",
    "train_interpolated_window_fraction",
    "train_effective_rank_mean",
    "train_condition_number_maximum",
    "shuffle_mean_absolute_information_coefficient",
    "shuffle_maximum_absolute_information_coefficient",
    "trading_total_trades",
    "trading_profit_total",
    "trading_sharpe",
    "trading_profit_factor",
    "trading_max_drawdown_account",
)


@dataclass(frozen=True)
class Phase11Config:
    data_directory: Path
    output_directory: Path = Path("user_data/research_results/double_descent/phase11")
    phase10_summary: Path = Path("user_data/research_results/double_descent/phase10/summary.json")
    phase10_map: Path = Path(
        "user_data/research_results/double_descent/phase10/representation_map.csv"
    )
    python_executable: str = sys.executable
    cuda_python_executable: str = ""
    pair: str = "BTC/USDT:USDT"
    timeframe: str = "1h"
    timerange: str = "20250101-20260101"
    holdout_start: str = "20260101"
    train_period_days: int = 90
    backtest_period_days: int = 30
    effective_n: int = 2_159
    map_ratios: tuple[float, ...] = PHASE11_MAP_RATIOS
    robustness_ratios: tuple[float, ...] = PHASE11_ROBUSTNESS_RATIOS
    feature_seeds: tuple[int, ...] = PHASE11_FEATURE_SEEDS
    label_shuffle_seeds: tuple[int, ...] = PHASE11_LABEL_SHUFFLE_SEEDS
    gamma: float = PHASE11_GAMMA
    ridge: float = 0.0
    rcond: float = 1e-12
    dtype: str = "float64"
    chunk_size: int = 4_096
    fee: float = 0.001
    minimum_training_windows: int = 10
    minimum_seed_count: int = 3
    maximum_absolute_train_label_correlation: float = MAXIMUM_ABSOLUTE_TRAIN_LABEL_CORRELATION
    subprocess_timeout_seconds: int = 1_800
    resume: bool = True
    smoke_test: bool = False
    strategy_directory: Path = Path("research/double_descent/freqai")
    model_directory: Path = Path("research/double_descent/freqai")
    models_directory: Path = Path("user_data/models")

    def validate(self) -> None:
        self._validate_phase10_reference()
        self._validate_design()
        _phase4_config(
            self,
            feature_seed=self.feature_seeds[0],
            ratios=self.map_ratios,
            output_directory=self.output_directory / "validation",
        ).validate()

    def _validate_phase10_reference(self) -> None:
        if not self.phase10_summary.is_file() or not self.phase10_map.is_file():
            raise FileNotFoundError("Phase 11 requires the completed Phase 10 artifacts")
        summary = json.loads(self.phase10_summary.read_text(encoding="utf-8"))
        if summary.get("phase") != 10 or not summary.get("gate", {}).get("passed"):
            raise ValueError("Phase 10 did not pass its integrity gate")
        reference = summary.get("config", {})
        if reference.get("gamma") != self.gamma or reference.get("ridge") != self.ridge:
            raise ValueError("Phase 11 must reuse the frozen Phase 10 gamma and ridge")
        if summary.get("design", {}).get("holdout_used") is not False:
            raise ValueError("the Phase 10 reference does not certify a sealed holdout")

    def _validate_design(self) -> None:
        for ratios, label in (
            (self.map_ratios, "main map"),
            (self.robustness_ratios, "robustness map"),
        ):
            if not ratios or tuple(sorted(set(ratios))) != ratios or any(x <= 0 for x in ratios):
                raise ValueError(f"{label} must be positive, sorted, and unique")
        if not set(self.robustness_ratios).issubset(self.map_ratios):
            raise ValueError("robustness ratios must be a subset of the main map")
        self._validate_seeds()
        if self.ridge != 0 or self.gamma != PHASE11_GAMMA:
            raise ValueError("Phase 11 freezes gamma=0.5 and ridge=0")
        if not 0 < self.maximum_absolute_train_label_correlation < 1:
            raise ValueError("the shuffled-label correlation bound must be in (0, 1)")
        _, end = _parse_timerange(self.timerange)
        holdout = datetime.strptime(self.holdout_start, "%Y%m%d").replace(tzinfo=UTC)
        if end > holdout:
            raise ValueError("Phase 11 may not enter the sealed holdout")
        self._validate_confirmatory_design()

    def _validate_seeds(self) -> None:
        if len(self.feature_seeds) != len(self.label_shuffle_seeds):
            raise ValueError("feature and label-shuffle seeds must be paired one-to-one")
        if any(seed < 0 for seed in self.feature_seeds):
            raise ValueError("feature seeds must be non-negative")
        if any(seed < 0 for seed in self.label_shuffle_seeds):
            raise ValueError("label-shuffle seeds must be non-negative")
        if len(set(self.feature_seeds)) != len(self.feature_seeds):
            raise ValueError("feature seeds must be unique")
        if len(set(self.label_shuffle_seeds)) != len(self.label_shuffle_seeds):
            raise ValueError("label-shuffle seeds must be unique")
        minimum = 1 if self.smoke_test else 3
        if self.minimum_seed_count < minimum or len(self.feature_seeds) < self.minimum_seed_count:
            raise ValueError(f"Phase 11 requires at least {minimum} paired seed(s)")

    def _validate_confirmatory_design(self) -> None:
        if not self.smoke_test:
            frozen = (
                (self.map_ratios, PHASE11_MAP_RATIOS, "main map"),
                (self.robustness_ratios, PHASE11_ROBUSTNESS_RATIOS, "robustness map"),
                (self.feature_seeds, PHASE11_FEATURE_SEEDS, "feature seeds"),
                (
                    self.label_shuffle_seeds,
                    PHASE11_LABEL_SHUFFLE_SEEDS,
                    "label-shuffle seeds",
                ),
            )
            for actual, expected, label in frozen:
                if actual != expected:
                    raise ValueError(f"the confirmatory run requires the frozen {label}")


def _phase4_config(
    config: Phase11Config,
    feature_seed: int,
    ratios: tuple[float, ...],
    output_directory: Path,
) -> Phase4Config:
    return Phase4Config(
        data_directory=config.data_directory,
        output_directory=output_directory,
        python_executable=config.python_executable,
        cuda_python_executable=config.cuda_python_executable,
        pair=config.pair,
        timeframe=config.timeframe,
        timerange=config.timerange,
        holdout_start=config.holdout_start,
        train_period_days=config.train_period_days,
        backtest_period_days=config.backtest_period_days,
        effective_n=config.effective_n,
        ratios=ratios,
        seed=feature_seed,
        gamma=config.gamma,
        ridge=config.ridge,
        rcond=config.rcond,
        dtype=config.dtype,
        chunk_size=config.chunk_size,
        fee=config.fee,
        minimum_training_windows=config.minimum_training_windows,
        subprocess_timeout_seconds=config.subprocess_timeout_seconds,
        resume=config.resume,
        strategy_directory=config.strategy_directory,
        model_directory=config.model_directory,
        models_directory=config.models_directory,
    )


def _task_grid(config: Phase11Config) -> list[tuple[int, int, tuple[float, ...]]]:
    tasks = []
    for index, (feature_seed, label_seed) in enumerate(
        zip(config.feature_seeds, config.label_shuffle_seeds, strict=True)
    ):
        ratios = config.map_ratios if index == 0 else config.robustness_ratios
        tasks.append((feature_seed, label_seed, ratios))
    return tasks


def _case_paths(
    config: Phase4Config,
    ratio: float,
    label_seed: int,
) -> tuple[str, Path, Path, Path, Path]:
    feature_count = feature_count_for_ratio(ratio, config.effective_n)
    slug = f"pn-{ratio:.5f}-p-{feature_count}-fseed-{config.seed}-lseed-{label_seed}"
    config_path = config.output_directory / "configs" / f"{slug}.json"
    log_path = config.output_directory / "logs" / f"{slug}.log"
    metrics_path = config.output_directory / "training_diagnostics" / f"{slug}.jsonl"
    backtest_directory = config.output_directory / "backtests" / slug
    return slug, config_path, log_path, metrics_path, backtest_directory


def _summarize_label_shuffle(records: list[dict[str, Any]]) -> dict[str, Any]:
    correlations = [abs(float(row["label_shuffle_information_coefficient"])) for row in records]
    return {
        "window_count": len(records),
        "mean_absolute_information_coefficient": statistics.fmean(correlations),
        "maximum_absolute_information_coefficient": max(correlations),
        "maximum_fixed_point_count": max(int(row["label_fixed_point_count"]) for row in records),
        "all_multisets_preserved": all(bool(row["label_multiset_preserved"]) for row in records),
        "all_permutations_non_identity": all(
            not bool(row["label_permutation_identity"]) for row in records
        ),
        "permutation_hashes": {
            str(row["model_window"]): str(row["label_permutation_sha256"]) for row in records
        },
    }


def _enrich_case(
    result: dict[str, Any],
    metrics_path: Path,
    label_seed: int,
    maximum_absolute_correlation: float,
) -> dict[str, Any]:
    records = _read_training_records(metrics_path)
    diagnostics_match = bool(records) and all(
        row.get("phase") == 11
        and row.get("representation") == "market_rff"
        and row.get("labels_shuffled") is True
        and int(row.get("label_shuffle_seed", -1)) == label_seed
        and row.get("label_multiset_preserved") is True
        and row.get("label_permutation_identity") is False
        and int(row.get("label_fixed_point_count", -1)) == 0
        for row in records
    )
    shuffle = _summarize_label_shuffle(records) if diagnostics_match else {}
    correlation_passed = bool(shuffle) and (
        float(shuffle["maximum_absolute_information_coefficient"]) <= maximum_absolute_correlation
    )
    result["label_shuffle_seed"] = label_seed
    result["label_shuffle"] = shuffle
    result["integrity"]["shuffled_label_diagnostics_match"] = diagnostics_match
    result["integrity"]["shuffled_train_label_correlation_bounded"] = correlation_passed
    result["success"] = bool(result["success"] and diagnostics_match and correlation_passed)
    return result


def _recover_case(
    config: Phase4Config,
    ratio: float,
    label_seed: int,
    market_data: pd.DataFrame,
    maximum_absolute_correlation: float,
) -> dict[str, Any] | None:
    _, config_path, log_path, metrics_path, backtest_directory = _case_paths(
        config, ratio, label_seed
    )
    if not config_path.is_file() or not log_path.is_file() or not metrics_path.is_file():
        return None
    feature_count = feature_count_for_ratio(ratio, config.effective_n)
    try:
        generated = json.loads(config_path.read_text(encoding="utf-8"))
        freqai = generated["freqai"]
        parameters = freqai["model_training_parameters"]
        configuration_matches = all(
            (
                parameters["feature_count"] == feature_count,
                parameters["seed"] == config.seed,
                parameters["label_shuffle_seed"] == label_seed,
                parameters["gamma"] == config.gamma,
                parameters["ridge"] == config.ridge,
                parameters["rcond"] == config.rcond,
                parameters["dtype"] == config.dtype,
                freqai["train_period_days"] == config.train_period_days,
                freqai["backtest_period_days"] == config.backtest_period_days,
                generated["fee"] == config.fee,
            )
        )
        if not configuration_matches:
            return None
        result = _finalize_case(
            config,
            ratio,
            freqai["identifier"],
            config_path,
            log_path,
            metrics_path,
            backtest_directory,
            market_data,
            returncode=0,
            timed_out=False,
            wall_seconds=max(log_path.stat().st_mtime - config_path.stat().st_mtime, 0.0),
            stderr="",
            recovered=True,
        )
        result = _enrich_case(result, metrics_path, label_seed, maximum_absolute_correlation)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not result["success"]:
        return None
    print(
        f"PHASE11 RECOVERED feature_seed={config.seed} label_seed={label_seed} "
        f"P/N={ratio:.5f} P={feature_count}",
        flush=True,
    )
    return result


def _run_case(
    config: Phase4Config,
    ratio: float,
    label_seed: int,
    run_id: str,
    market_data: pd.DataFrame,
    maximum_absolute_correlation: float,
) -> dict[str, Any]:
    feature_count = feature_count_for_ratio(ratio, config.effective_n)
    _, config_path, log_path, metrics_path, backtest_directory = _case_paths(
        config, ratio, label_seed
    )
    if backtest_directory.exists():
        shutil.rmtree(backtest_directory)
    for directory in (config_path.parent, log_path.parent, backtest_directory, metrics_path.parent):
        directory.mkdir(parents=True, exist_ok=True)
    metrics_path.unlink(missing_ok=True)
    identifier = f"double-descent-phase11-{run_id}-p{feature_count}-f{config.seed}-l{label_seed}"
    generated = build_freqtrade_config(config, feature_count, identifier, metrics_path, run_id)
    parameters = generated["freqai"]["model_training_parameters"]
    parameters.update(
        {
            "phase11_metrics_path": str(metrics_path.resolve()),
            "phase11_run_id": run_id,
            "representation": "market_rff",
            "label_shuffle_seed": label_seed,
        }
    )
    config_path.write_text(json.dumps(generated, indent=2), encoding="utf-8")
    command = [
        config.python_executable,
        "-m",
        "freqtrade",
        "backtesting",
        "--config",
        str(config_path),
        "--strategy",
        "Phase4RFFStrategy",
        "--strategy-path",
        str(config.strategy_directory),
        "--freqaimodel",
        "Phase11ShuffledLabelRegressor",
        "--freqaimodel-path",
        str(config.model_directory),
        "--timerange",
        config.timerange,
        "--timeframe",
        config.timeframe,
        "--pairs",
        config.pair,
        "--datadir",
        str(config.data_directory),
        "--cache",
        "none",
        "--export",
        "trades",
        "--export-directory",
        str(backtest_directory),
        "--no-color",
    ]
    print(
        f"PHASE11 START feature_seed={config.seed} label_seed={label_seed} "
        f"P/N={ratio:.5f} P={feature_count}",
        flush=True,
    )
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    started = datetime.now(tz=UTC)
    try:
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
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
    log_path.write_text(
        f"COMMAND: {subprocess.list2cmdline(command)}\n\n{stdout}\n\nSTDERR:\n{stderr}",
        encoding="utf-8",
    )
    result = _finalize_case(
        config,
        ratio,
        identifier,
        config_path,
        log_path,
        metrics_path,
        backtest_directory,
        market_data,
        returncode,
        timed_out,
        wall_seconds,
        stderr,
    )
    result = _enrich_case(result, metrics_path, label_seed, maximum_absolute_correlation)
    print(
        f"PHASE11 DONE feature_seed={config.seed} label_seed={label_seed} "
        f"P/N={ratio:.5f} success={result['success']} seconds={wall_seconds:.1f}",
        flush=True,
    )
    return result


def _run_substudy(
    config: Phase11Config,
    feature_seed: int,
    label_seed: int,
    ratios: tuple[float, ...],
    run_id: str,
) -> dict[str, Any]:
    output_directory = (
        config.output_directory / "runs" / f"feature-{feature_seed}-labels-{label_seed}"
    )
    phase4 = _phase4_config(config, feature_seed, ratios, output_directory)
    phase4.validate()
    phase3 = Phase3Config(
        data_directory=config.data_directory,
        python_executable=config.python_executable,
        timerange=config.timerange,
        train_periods_days=(config.train_period_days,),
        backtest_period_days=config.backtest_period_days,
        minimum_windows_per_period=config.minimum_training_windows,
    )
    data_audit = audit_data_coverage(phase3)
    market_data = _load_evaluation_market_data(phase4) if data_audit["passed"] else pd.DataFrame()
    results = []
    checkpoint_path = output_directory / "checkpoint.json"
    if data_audit["passed"]:
        for ratio in ratios:
            result = (
                _recover_case(
                    phase4,
                    ratio,
                    label_seed,
                    market_data,
                    config.maximum_absolute_train_label_correlation,
                )
                if config.resume
                else None
            )
            if result is None:
                result = _run_case(
                    phase4,
                    ratio,
                    label_seed,
                    run_id,
                    market_data,
                    config.maximum_absolute_train_label_correlation,
                )
            results.append(result)
            checkpoint_path.write_text(
                json.dumps(_json_safe(results), indent=2, allow_nan=False),
                encoding="utf-8",
            )
    gate = evaluate_phase4_gate(phase4, data_audit, results)
    gate["checks"]["shuffled_label_integrity_passed"] = bool(results) and all(
        row["integrity"].get("shuffled_label_diagnostics_match")
        and row["integrity"].get("shuffled_train_label_correlation_bounded")
        for row in results
    )
    gate["passed"] = all(gate["checks"].values())
    output_directory.mkdir(parents=True, exist_ok=True)
    summary_path = output_directory / "summary.json"
    summary = {
        "phase": 11,
        "scope": "one paired feature/permutation seed rolling FreqAI substudy",
        "run_id": run_id,
        "feature_seed": feature_seed,
        "label_shuffle_seed": label_seed,
        "ratios": list(ratios),
        "data_audit": data_audit,
        "results": results,
        "gate": gate,
        "artifacts": {"summary": str(summary_path), "checkpoint": str(checkpoint_path)},
    }
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    return summary


def flatten_substudies(substudies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for substudy in substudies:
        for result in substudy["results"]:
            row = _flatten_result(result)
            row["feature_seed"] = substudy["feature_seed"]
            row["label_shuffle_seed"] = substudy["label_shuffle_seed"]
            for key, value in result["label_shuffle"].items():
                if not isinstance(value, dict):
                    row[f"shuffle_{key}"] = value
            rows.append(row)
    return rows


def aggregate_robustness(
    config: Phase11Config,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    aggregates = []
    for ratio in config.robustness_ratios:
        matched = [row for row in rows if row["target_pn_ratio"] == ratio]
        if len(matched) != len(config.feature_seeds):
            continue
        for metric in AGGREGATE_METRICS:
            aggregates.append(
                {
                    "target_pn_ratio": ratio,
                    "actual_pn_ratio": matched[0]["actual_pn_ratio"],
                    "feature_count": matched[0]["feature_count"],
                    "metric": metric,
                    **summarize_values([float(row[metric]) for row in matched]),
                }
            )
    return aggregates


def compare_with_unshuffled(
    config: Phase11Config,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    reference = pd.read_csv(config.phase10_map)
    reference = reference.loc[reference["representation"] == "market_rff"]
    lookup = {(int(row.seed), float(row.target_pn_ratio)): row for row in reference.itertuples()}
    comparisons = []
    for row in rows:
        baseline = lookup[(int(row["feature_seed"]), float(row["target_pn_ratio"]))]
        comparisons.append(
            {
                "feature_seed": row["feature_seed"],
                "label_shuffle_seed": row["label_shuffle_seed"],
                "target_pn_ratio": row["target_pn_ratio"],
                "feature_count": row["feature_count"],
                "shuffled_mse": row["oos_mse"],
                "unshuffled_mse": baseline.oos_mse,
                "shuffled_to_unshuffled_mse": float(row["oos_mse"]) / baseline.oos_mse,
                "shuffled_mse_beats_zero": float(row["oos_mse"]) < float(row["zero_mse"]),
                "shuffled_oos_r2": row["oos_r2"],
                "shuffled_oos_ic": row["oos_information_coefficient"],
            }
        )
    robust = [row for row in comparisons if row["target_pn_ratio"] in config.robustness_ratios]
    ratios = [float(row["shuffled_to_unshuffled_mse"]) for row in robust]
    systematic = []
    for ratio in config.robustness_ratios:
        matched = [row for row in robust if row["target_pn_ratio"] == ratio]
        if len(matched) == len(config.feature_seeds) and all(
            bool(row["shuffled_mse_beats_zero"]) for row in matched
        ):
            systematic.append(ratio)
    return {
        "primary_metric": "chronological OOS MSE; trading is excluded from inference",
        "matched_case_count": len(comparisons),
        "robust_matched_case_count": len(robust),
        "shuffled_mse_win_count_vs_unshuffled": sum(
            value < 1 for value in (row["shuffled_to_unshuffled_mse"] for row in robust)
        ),
        "geometric_mean_shuffled_to_unshuffled_mse": math.exp(
            statistics.fmean(math.log(value) for value in ratios)
        ),
        "median_shuffled_to_unshuffled_mse": statistics.median(ratios),
        "shuffled_mse_beats_zero_case_count": sum(
            float(row["oos_mse"]) < float(row["zero_mse"]) for row in rows
        ),
        "shuffled_positive_r2_case_count": sum(float(row["oos_r2"]) > 0 for row in rows),
        "shuffled_positive_net_return_case_count": sum(
            float(row["trading_profit_total"]) > 0 for row in rows
        ),
        "shuffled_positive_sharpe_case_count": sum(
            float(row["trading_sharpe"]) > 0 for row in rows
        ),
        "shuffled_profit_factor_above_one_case_count": sum(
            float(row["trading_profit_factor"]) > 1 for row in rows
        ),
        "ratios_beating_zero_in_all_seeds": systematic,
        "systematic_null_failure_detected": bool(systematic),
        "maximum_absolute_oos_ic": max(
            abs(float(row["oos_information_coefficient"])) for row in rows
        ),
        "comparisons": comparisons,
    }


def _permutation_hashes_match_across_p(substudies: list[dict[str, Any]]) -> bool:
    for substudy in substudies:
        by_window: dict[str, list[str]] = {}
        expected_repetitions = len(substudy["results"])
        for result in substudy["results"]:
            for window, fingerprint in result["label_shuffle"]["permutation_hashes"].items():
                by_window.setdefault(window, []).append(fingerprint)
        if not by_window or any(
            len(values) != expected_repetitions or len(set(values)) != 1
            for values in by_window.values()
        ):
            return False
    return True


def evaluate_phase11_gate(
    config: Phase11Config,
    substudies: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_case_count = (
        sum(len(ratios) for _, _, ratios in _task_grid(config))
        if config.smoke_test
        else PHASE11_EXPECTED_CASE_COUNT
    )
    expected_cells = {
        (feature_seed, label_seed, ratio)
        for feature_seed, label_seed, ratios in _task_grid(config)
        for ratio in ratios
    }
    observed_cells = {
        (row["feature_seed"], row["label_shuffle_seed"], row["target_pn_ratio"]) for row in rows
    }
    expected_aggregates = len(config.robustness_ratios) * len(AGGREGATE_METRICS)
    phase10 = json.loads(config.phase10_summary.read_text(encoding="utf-8"))
    reference = pd.read_csv(config.phase10_map)
    reference = reference.loc[reference["representation"] == "market_rff"]
    reference_lookup = {
        (int(row.seed), float(row.target_pn_ratio)): row for row in reference.itertuples()
    }
    references_match = all(
        (row["feature_seed"], row["target_pn_ratio"]) in reference_lookup
        and int(row["oos_observation_count"])
        == int(
            reference_lookup[(row["feature_seed"], row["target_pn_ratio"])].oos_observation_count
        )
        and all(
            math.isclose(
                float(row[metric]),
                float(
                    getattr(
                        reference_lookup[(row["feature_seed"], row["target_pn_ratio"])],
                        metric,
                    )
                ),
                rel_tol=1e-12,
                abs_tol=1e-20,
            )
            for metric in (
                "zero_mse",
                "oos_actual_mean",
                "oos_actual_standard_deviation",
            )
        )
        for row in rows
    )
    checks = {
        "phase10_reference_gate_passed": phase10.get("gate", {}).get("passed") is True,
        "every_rolling_substudy_passed": bool(substudies)
        and all(item["gate"]["passed"] for item in substudies),
        "complete_predeclared_case_count": len(rows) == expected_case_count,
        "complete_predeclared_cell_map": observed_cells == expected_cells,
        "complete_robustness_aggregate": len(aggregates) == expected_aggregates,
        "feature_counts_match_measured_n": all(
            int(row["feature_count"])
            == feature_count_for_ratio(float(row["target_pn_ratio"]), config.effective_n)
            for row in rows
        ),
        "all_metrics_are_finite": bool(rows)
        and all(
            math.isfinite(float(row[key]))
            for row in rows
            for key in (
                "oos_mse",
                "oos_mae",
                "oos_r2",
                "oos_information_coefficient",
                "shuffle_maximum_absolute_information_coefficient",
                "trading_sharpe",
            )
        ),
        "every_label_multiset_preserved": all(
            bool(row["shuffle_all_multisets_preserved"]) for row in rows
        ),
        "every_label_permutation_is_a_derangement": all(
            bool(row["shuffle_all_permutations_non_identity"])
            and int(row["shuffle_maximum_fixed_point_count"]) == 0
            for row in rows
        ),
        "train_label_correlation_is_bounded": all(
            float(row["shuffle_maximum_absolute_information_coefficient"])
            <= config.maximum_absolute_train_label_correlation
            for row in rows
        ),
        "same_permutation_reused_across_p": _permutation_hashes_match_across_p(substudies),
        "phase10_oos_coverage_moments_and_zero_baseline_reproduced": references_match,
        "same_oos_observations_for_every_case": len(
            {int(row["oos_observation_count"]) for row in rows}
        )
        == 1,
        "same_zero_baseline_for_every_case": np.allclose(
            [float(row["zero_mse"]) for row in rows],
            float(rows[0]["zero_mse"]),
            rtol=0,
            atol=1e-20,
        ),
        "trading_excluded_from_inference": True,
        "holdout_was_not_used": _parse_timerange(config.timerange)[1]
        <= datetime.strptime(config.holdout_start, "%Y%m%d").replace(tzinfo=UTC),
        "cuda_float64_used": config.dtype == "float64"
        and all(float(row["train_peak_vram_mib"]) > 0 for row in rows),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "expected_case_count": expected_case_count,
        "observed_case_count": len(rows),
    }


def _write_plot(config: Phase11Config, path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return False
    reference = sorted(
        (row for row in rows if row["feature_seed"] == config.feature_seeds[0]),
        key=lambda row: row["actual_pn_ratio"],
    )
    phase10 = pd.read_csv(config.phase10_map)
    phase10 = phase10.loc[
        (phase10["representation"] == "market_rff") & (phase10["seed"] == config.feature_seeds[0])
    ].sort_values("actual_pn_ratio")
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("OOS MSE / zero", "OOS IC", "Training MSE", "Net Sharpe"),
    )
    x = [row["actual_pn_ratio"] for row in reference]
    figure.add_trace(
        go.Scatter(
            x=x,
            y=[float(row["oos_mse"]) / float(row["zero_mse"]) for row in reference],
            mode="lines+markers",
            name="shuffled labels",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=phase10["actual_pn_ratio"],
            y=phase10["oos_mse"] / phase10["zero_mse"],
            mode="lines+markers",
            name="original labels",
        ),
        row=1,
        col=1,
    )
    panels = (
        (1, 2, "oos_information_coefficient"),
        (2, 1, "train_train_mse_mean"),
        (2, 2, "trading_sharpe"),
    )
    for panel_row, panel_column, metric in panels:
        figure.add_trace(
            go.Scatter(
                x=x,
                y=[row[metric] for row in reference],
                mode="lines+markers",
                name=f"shuffled {metric}",
                showlegend=False,
            ),
            row=panel_row,
            col=panel_column,
        )
    figure.update_xaxes(type="log", title_text="P/N")
    figure.update_yaxes(type="log", row=1, col=1)
    figure.update_yaxes(type="log", row=2, col=1)
    figure.update_layout(
        title="Phase 11 - Shuffled-label negative control",
        template="plotly_white",
        height=900,
        width=1400,
    )
    figure.write_html(path, include_plotlyjs="cdn")
    return True


def run_phase11(config: Phase11Config) -> dict[str, Any]:
    config.validate()
    config.output_directory.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    checkpoint_path = config.output_directory / "checkpoint.json"
    substudies = []
    completed = []
    for feature_seed, label_seed, ratios in _task_grid(config):
        print(
            f"PHASE11 SUBSTUDY feature_seed={feature_seed} label_seed={label_seed} "
            f"points={len(ratios)}",
            flush=True,
        )
        substudy = _run_substudy(config, feature_seed, label_seed, ratios, run_id)
        substudies.append(substudy)
        completed.append(
            {
                "feature_seed": feature_seed,
                "label_shuffle_seed": label_seed,
                "ratios": list(ratios),
                "passed": substudy["gate"]["passed"],
                "summary": substudy["artifacts"]["summary"],
            }
        )
        checkpoint_path.write_text(
            json.dumps(_json_safe({"run_id": run_id, "completed": completed}), indent=2),
            encoding="utf-8",
        )
    rows = flatten_substudies(substudies)
    aggregates = aggregate_robustness(config, rows)
    curve_assessments = []
    for feature_seed in config.feature_seeds:
        matched = [
            row
            for row in rows
            if row["feature_seed"] == feature_seed
            and row["target_pn_ratio"] in config.robustness_ratios
        ]
        curve_assessments.append({"feature_seed": feature_seed, **assess_flat_curve(matched)})
    comparison = compare_with_unshuffled(config, rows)
    gate = evaluate_phase11_gate(config, substudies, rows, aggregates)
    map_path = config.output_directory / "shuffled_label_map.csv"
    aggregate_path = config.output_directory / "robustness_aggregates.csv"
    comparison_path = config.output_directory / "matched_comparisons.csv"
    summary_path = config.output_directory / "summary.json"
    plot_path = config.output_directory / "shuffled_label_control.html"
    _write_csv(map_path, rows)
    _write_csv(aggregate_path, aggregates)
    _write_csv(comparison_path, comparison["comparisons"])
    plot_written = _write_plot(config, plot_path, rows)
    config_payload = asdict(config)
    for key in (
        "data_directory",
        "output_directory",
        "phase10_summary",
        "phase10_map",
        "strategy_directory",
        "model_directory",
        "models_directory",
    ):
        config_payload[key] = str(config_payload[key])
    summary = {
        "phase": 11,
        "objective": "Verify that predictability disappears after shuffling training labels",
        "scope": "development-only negative control; 2026 holdout sealed",
        "run_id": run_id,
        "design": {
            "representation": "market_rff",
            "original_features_and_dates_preserved": True,
            "training_labels_permuted_within_each_rolling_window": True,
            "permutation_is_timestamp_keyed": True,
            "permutation_is_a_derangement": True,
            "same_permutation_reused_across_p": True,
            "oos_labels_shuffled": False,
            "trading_used_for_inference": False,
            "holdout_used": False,
        },
        "config": config_payload,
        "compute": {
            "case_count": len(rows),
            "rolling_model_fit_count": sum(int(row["train_training_window_count"]) for row in rows),
            "summed_freqtrade_wall_seconds": sum(float(row["wall_seconds"]) for row in rows),
            "summed_cuda_training_seconds": sum(
                float(row["train_training_seconds_total"]) for row in rows
            ),
            "maximum_peak_vram_mib": max(float(row["train_peak_vram_mib"]) for row in rows),
        },
        "curve_assessments": curve_assessments,
        "robustness_aggregates": aggregates,
        "negative_control": comparison,
        "gate": gate,
        "artifacts": {
            "summary": str(summary_path),
            "shuffled_label_map": str(map_path),
            "robustness_aggregates": str(aggregate_path),
            "matched_comparisons": str(comparison_path),
            "interactive_plot": str(plot_path) if plot_written else None,
            "checkpoint": str(checkpoint_path),
        },
    }
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    return summary
