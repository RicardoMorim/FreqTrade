"""Phase 20: frozen effective-dimension audit of prior RFF experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PHASE20_PROTOCOL_VERSION = 1
PHASE20_EXPECTED_PHASE6_ROWS = 30
PHASE20_EXPECTED_PHASE10_ROWS = 64
PHASE20_EXPECTED_PHASE15_ROWS = 78
PHASE20_EXPECTED_TOTAL_ROWS = 173
PHASE20_EXPECTED_PHASE10_CONTRASTS = 11
PHASE20_HOLDOUT_START = "20260101"
PHASE20_EFFECTIVE_RANK_DEFINITION = (
    "spectral participation ratio (sum(lambda))^2 / sum(lambda^2), computed on "
    "retained eigenvalues of the centered training Gram matrix"
)


@dataclass(frozen=True)
class Phase20Config:
    phase6_summary: Path = Path("user_data/research_results/double_descent/phase6/summary.json")
    phase6_kernel_result: Path = Path(
        "user_data/research_results/double_descent/phase6/kernel_result.json"
    )
    phase6_seed_rows: Path = Path(
        "user_data/research_results/double_descent/phase6/prediction_convergence_by_seed.csv"
    )
    phase10_summary: Path = Path(
        "user_data/research_results/double_descent/phase10/summary.json"
    )
    phase10_rows: Path = Path(
        "user_data/research_results/double_descent/phase10/representation_map.csv"
    )
    phase15_summary: Path = Path(
        "user_data/research_results/double_descent/phase15/summary.json"
    )
    phase15_preparation: Path = Path(
        "user_data/research_results/double_descent/phase15/preparation.json"
    )
    phase15_rows: Path = Path(
        "user_data/research_results/double_descent/phase15/rff_results.csv"
    )
    output_directory: Path = Path("user_data/research_results/double_descent/phase20")

    def source_paths(self) -> tuple[Path, ...]:
        return (
            self.phase6_summary,
            self.phase6_kernel_result,
            self.phase6_seed_rows,
            self.phase10_summary,
            self.phase10_rows,
            self.phase15_summary,
            self.phase15_preparation,
            self.phase15_rows,
        )

    def load_and_validate(self) -> dict[str, Any]:
        missing = [str(path) for path in self.source_paths() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Phase 20 source artifacts are missing: {missing}")
        phase6 = _read_json(self.phase6_summary)
        phase10 = _read_json(self.phase10_summary)
        phase15 = _read_json(self.phase15_summary)
        phase15_preparation = _read_json(self.phase15_preparation)
        checks = {
            "phase6_complete_gate_passed": phase6.get("phase") == 6
            and phase6.get("gate", {}).get("passed") is True,
            "phase10_complete_gate_passed": phase10.get("phase") == 10
            and phase10.get("gate", {}).get("passed") is True,
            "phase15_complete_gate_passed": phase15.get("phase") == 15
            and phase15.get("gate", {}).get("passed") is True,
            "phase15_preparation_passed": phase15_preparation.get("phase") == 15
            and phase15_preparation.get("gate", {}).get("passed") is True,
            "all_sources_kept_holdout_sealed": all(
                (
                    phase6.get("gate", {}).get("checks", {}).get("holdout_was_not_used") is True,
                    phase10.get("design", {}).get("holdout_used") is False,
                    phase15.get("design", {}).get("holdout_used") is False,
                    phase15_preparation.get("design", {}).get("holdout_used") is False,
                )
            ),
            "all_rff_sources_are_float64": phase6.get("config", {}).get("dtype") == "float64"
            and phase10.get("config", {}).get("dtype") == "float64"
            and phase15.get("config", {}).get("dtype") == "float64",
            "ridgeless_sources_preserved": phase6.get("config", {}).get("ridge") == 0
            and phase10.get("config", {}).get("ridge") == 0
            and phase15.get("config", {}).get("ridge") == 0,
            "sealed_holdout_boundary_preserved": phase10.get("config", {}).get("holdout_start")
            == PHASE20_HOLDOUT_START
            and phase15.get("config", {}).get("holdout_start") == PHASE20_HOLDOUT_START,
            "kernel_and_representation_gammas_are_labelled_separately": phase6.get(
                "config", {}
            ).get("gamma")
            == 0.2
            and phase10.get("config", {}).get("gamma") == 0.5
            and phase15.get("config", {}).get("gamma") == 0.5,
        }
        if not all(checks.values()):
            failed = sorted(name for name, passed in checks.items() if not passed)
            raise ValueError(f"Phase 20 frozen source protocol mismatch: {failed}")
        return {
            "phase6": phase6,
            "phase10": phase10,
            "phase15": phase15,
            "phase15_preparation": phase15_preparation,
            "checks": checks,
        }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"expected a finite number, got {value!r}")
    return result


def _as_int(value: Any) -> int:
    result = _as_float(value)
    rounded = round(result)
    if not math.isclose(result, rounded, abs_tol=1e-9):
        raise ValueError(f"expected an integer, got {value!r}")
    return int(rounded)


def dimension_metrics(
    *,
    feature_count: int | None,
    effective_n: int,
    effective_rank: float,
    algebraic_rank: float | None,
) -> dict[str, float | int | None]:
    """Normalize nominal, algebraic, and spectral dimensions without conflating them."""
    if effective_n < 2 or effective_rank <= 0 or not math.isfinite(effective_rank):
        raise ValueError("effective N and spectral effective rank must be positive")
    maximum_centered_rank = effective_n - 1
    available_rank = (
        maximum_centered_rank
        if feature_count is None
        else min(int(feature_count), maximum_centered_rank)
    )
    if effective_rank > available_rank + 1e-6:
        raise ValueError("spectral effective rank exceeds the available centered rank")
    if algebraic_rank is not None:
        if algebraic_rank <= 0 or algebraic_rank > available_rank + 1e-6:
            raise ValueError("algebraic rank is outside the available centered rank")
        if effective_rank > algebraic_rank + 1e-6:
            raise ValueError("spectral effective rank exceeds algebraic rank")
    return {
        "maximum_centered_rank": maximum_centered_rank,
        "available_rank": available_rank,
        "effective_rank_fraction_of_n": effective_rank / effective_n,
        "effective_rank_fraction_of_available_rank": effective_rank / available_rank,
        "nominal_to_effective_dimension": (
            feature_count / effective_rank if feature_count is not None else None
        ),
        "algebraic_to_effective_dimension": (
            algebraic_rank / effective_rank if algebraic_rank is not None else None
        ),
        "algebraic_rank_fraction_of_available_rank": (
            algebraic_rank / available_rank if algebraic_rank is not None else None
        ),
    }


def _row(
    *,
    source_phase: int,
    study: str,
    asset: str,
    asset_label: str,
    representation: str,
    seed: int | None,
    gamma: float,
    feature_count: int | None,
    effective_n: int,
    actual_pn_ratio: float | None,
    effective_rank: float,
    algebraic_rank: float | None,
    condition_number: float,
    training_window_count: int,
) -> dict[str, Any]:
    metrics = dimension_metrics(
        feature_count=feature_count,
        effective_n=effective_n,
        effective_rank=effective_rank,
        algebraic_rank=algebraic_rank,
    )
    return {
        "source_phase": source_phase,
        "study": study,
        "asset": asset,
        "asset_label": asset_label,
        "representation": representation,
        "seed": seed,
        "gamma": gamma,
        "feature_count": feature_count,
        "nominal_complexity": "kernel_limit" if feature_count is None else "finite_features",
        "effective_n": effective_n,
        "actual_pn_ratio": actual_pn_ratio,
        "effective_rank": effective_rank,
        "algebraic_rank": algebraic_rank,
        "condition_number": condition_number,
        "training_window_count": training_window_count,
        "dtype": "float64",
        **metrics,
    }


def collect_effective_dimension_rows(
    config: Phase20Config, sources: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    phase6_rows = _read_csv(config.phase6_seed_rows)
    phase10_rows = _read_csv(config.phase10_rows)
    phase15_rows = _read_csv(config.phase15_rows)

    phase6_n = _as_int(sources["phase6"]["config"]["effective_n"])
    phase6_gamma = _as_float(sources["phase6"]["config"]["gamma"])
    for source in phase6_rows:
        rows.append(
            _row(
                source_phase=6,
                study="kernel_convergence_1h",
                asset="btc",
                asset_label="BTC",
                representation="market_rff",
                seed=_as_int(source["seed"]),
                gamma=phase6_gamma,
                feature_count=_as_int(source["feature_count"]),
                effective_n=phase6_n,
                actual_pn_ratio=_as_float(source["pn_ratio"]),
                effective_rank=_as_float(source["train_effective_rank_mean"]),
                algebraic_rank=None,
                condition_number=_as_float(source["train_condition_number_maximum"]),
                training_window_count=13,
            )
        )
    kernel = _read_json(config.phase6_kernel_result)
    training = kernel["training"]
    rows.append(
        _row(
            source_phase=6,
            study="kernel_convergence_1h",
            asset="btc",
            asset_label="BTC",
            representation="exact_rbf_kernel",
            seed=None,
            gamma=phase6_gamma,
            feature_count=None,
            effective_n=phase6_n,
            actual_pn_ratio=None,
            effective_rank=_as_float(training["effective_rank_mean"]),
            algebraic_rank=_as_float(training["rank_median"]),
            condition_number=_as_float(training["condition_number_maximum"]),
            training_window_count=_as_int(training["training_window_count"]),
        )
    )

    phase10_n = _as_int(sources["phase10"]["config"]["effective_n"])
    phase10_gamma = _as_float(sources["phase10"]["config"]["gamma"])
    for source in phase10_rows:
        if source.get("success", "").lower() != "true":
            raise ValueError("Phase 10 contains an unsuccessful source row")
        feature_count = _as_int(source["feature_count"])
        ratio = _as_float(source["actual_pn_ratio"])
        if not math.isclose(feature_count / ratio, phase10_n, rel_tol=0, abs_tol=0.51):
            raise ValueError("Phase 10 P/N does not recover frozen effective N")
        rows.append(
            _row(
                source_phase=10,
                study="representation_controls_1h",
                asset="btc",
                asset_label="BTC",
                representation=source["representation"],
                seed=_as_int(source["seed"]),
                gamma=phase10_gamma,
                feature_count=feature_count,
                effective_n=phase10_n,
                actual_pn_ratio=ratio,
                effective_rank=_as_float(source["train_effective_rank_mean"]),
                algebraic_rank=_as_float(source["train_rank_median"]),
                condition_number=_as_float(source["train_condition_number_maximum"]),
                training_window_count=_as_int(source["train_training_window_count"]),
            )
        )

    phase15_gamma = _as_float(sources["phase15"]["config"]["gamma"])
    effective_n_map = sources["phase15_preparation"]["effective_n_by_asset"]
    for source in phase15_rows:
        if source.get("success", "").lower() != "true":
            raise ValueError("Phase 15 contains an unsuccessful source row")
        asset = source["asset"]
        study = source["study"]
        effective_n = _as_int(effective_n_map[asset][study])
        feature_count = _as_int(source["feature_count"])
        ratio = _as_float(source["actual_pn_ratio"])
        if not math.isclose(feature_count / ratio, effective_n, rel_tol=0, abs_tol=0.51):
            raise ValueError("Phase 15 P/N does not recover frozen effective N")
        rows.append(
            _row(
                source_phase=15,
                study=study,
                asset=asset,
                asset_label=source["asset_label"],
                representation=source["representation"],
                seed=_as_int(source["seed"]),
                gamma=phase15_gamma,
                feature_count=feature_count,
                effective_n=effective_n,
                actual_pn_ratio=ratio,
                effective_rank=_as_float(source["train_effective_rank_mean"]),
                algebraic_rank=_as_float(source["train_rank_median"]),
                condition_number=_as_float(source["train_condition_number_maximum"]),
                training_window_count=_as_int(source["train_training_window_count"]),
            )
        )
    return rows, {
        "phase6_finite_rff": len(phase6_rows),
        "phase6_exact_kernel": 1,
        "phase10": len(phase10_rows),
        "phase15": len(phase15_rows),
    }


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    finite = [float(value) for value in values]
    if not finite or any(not math.isfinite(value) for value in finite):
        raise ValueError("cannot summarize empty or non-finite values")
    mean = statistics.fmean(finite)
    standard_deviation = statistics.stdev(finite) if len(finite) > 1 else 0.0
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(len(finite), 1.96)
    half_width = critical * standard_deviation / math.sqrt(len(finite)) if len(finite) > 1 else 0
    return {
        "count": len(finite),
        "mean": mean,
        "median": statistics.median(finite),
        "standard_deviation": standard_deviation,
        "minimum": min(finite),
        "maximum": max(finite),
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def aggregate_effective_dimensions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "source_phase",
        "study",
        "asset",
        "asset_label",
        "representation",
        "gamma",
        "feature_count",
        "nominal_complexity",
        "effective_n",
        "actual_pn_ratio",
    )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    aggregates: list[dict[str, Any]] = []
    for key, group in grouped.items():
        aggregate = dict(zip(keys, key, strict=True))
        aggregate["seed_count"] = len(group)
        for metric in (
            "effective_rank",
            "effective_rank_fraction_of_n",
            "effective_rank_fraction_of_available_rank",
            "nominal_to_effective_dimension",
            "condition_number",
        ):
            values = [row[metric] for row in group if row[metric] is not None]
            if values:
                for statistic, value in _summary(values).items():
                    aggregate[f"{metric}_{statistic}"] = value
        ranks = [row["algebraic_rank"] for row in group if row["algebraic_rank"] is not None]
        aggregate["algebraic_rank_median"] = statistics.median(ranks) if ranks else None
        aggregates.append(aggregate)
    return sorted(
        aggregates,
        key=lambda row: (
            row["source_phase"],
            row["study"],
            row["asset"],
            row["representation"],
            math.inf if row["feature_count"] is None else row["feature_count"],
        ),
    )


def build_curve_summaries(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in aggregates:
        if row["feature_count"] is None or row["representation"] == "market_linear":
            continue
        key = (
            row["source_phase"],
            row["study"],
            row["asset"],
            row["asset_label"],
            row["representation"],
            row["gamma"],
            row["effective_n"],
        )
        grouped[key].append(row)
    summaries: list[dict[str, Any]] = []
    for key, curve in grouped.items():
        curve.sort(key=lambda row: row["feature_count"])
        first, last = curve[0], curve[-1]
        nominal_growth = last["feature_count"] / first["feature_count"]
        effective_growth = (
            last["effective_rank_median"] / first["effective_rank_median"]
        )
        high = [row["effective_rank_median"] for row in curve if row["actual_pn_ratio"] >= 5]
        high_median = statistics.median(high) if high else None
        summaries.append(
            {
                "source_phase": key[0],
                "study": key[1],
                "asset": key[2],
                "asset_label": key[3],
                "representation": key[4],
                "gamma": key[5],
                "effective_n": key[6],
                "point_count": len(curve),
                "smallest_feature_count": first["feature_count"],
                "smallest_actual_pn_ratio": first["actual_pn_ratio"],
                "smallest_effective_rank": first["effective_rank_median"],
                "largest_feature_count": last["feature_count"],
                "largest_actual_pn_ratio": last["actual_pn_ratio"],
                "largest_effective_rank": last["effective_rank_median"],
                "largest_effective_rank_fraction_of_n": last[
                    "effective_rank_fraction_of_n_median"
                ],
                "largest_nominal_to_effective_dimension": last[
                    "nominal_to_effective_dimension_median"
                ],
                "nominal_growth": nominal_growth,
                "effective_rank_growth": effective_growth,
                "log_effective_rank_elasticity": (
                    math.log(effective_growth) / math.log(nominal_growth)
                    if nominal_growth > 1 and effective_growth > 0
                    else None
                ),
                "high_dimensional_point_count": len(high),
                "high_dimensional_effective_rank_relative_range": (
                    (max(high) - min(high)) / high_median if high_median else None
                ),
            }
        )
    return sorted(
        summaries,
        key=lambda row: (
            row["source_phase"], row["study"], row["asset"], row["representation"]
        ),
    )


def build_phase10_contrasts(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        row
        for row in aggregates
        if row["source_phase"] == 10
        and row["representation"] in {"market_rff", "pure_noise", "market_plus_noise"}
    ]
    by_feature: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_feature[row["feature_count"]][row["representation"]] = row
    contrasts = []
    for feature_count, representations in sorted(by_feature.items()):
        if set(representations) != {"market_rff", "pure_noise", "market_plus_noise"}:
            raise ValueError("Phase 10 representation controls are not matched by P")
        market = representations["market_rff"]
        noise = representations["pure_noise"]
        mixed = representations["market_plus_noise"]
        contrasts.append(
            {
                "feature_count": feature_count,
                "actual_pn_ratio": market["actual_pn_ratio"],
                "market_rff_effective_rank": market["effective_rank_median"],
                "pure_noise_effective_rank": noise["effective_rank_median"],
                "market_plus_noise_effective_rank": mixed["effective_rank_median"],
                "market_rff_to_noise_effective_rank_ratio": market["effective_rank_median"]
                / noise["effective_rank_median"],
                "market_plus_noise_to_noise_effective_rank_ratio": mixed[
                    "effective_rank_median"
                ]
                / noise["effective_rank_median"],
                "market_rff_effective_rank_fraction_of_n": market[
                    "effective_rank_fraction_of_n_median"
                ],
                "pure_noise_effective_rank_fraction_of_n": noise[
                    "effective_rank_fraction_of_n_median"
                ],
                "market_plus_noise_effective_rank_fraction_of_n": mixed[
                    "effective_rank_fraction_of_n_median"
                ],
            }
        )
    return contrasts


def _curve(
    curves: list[dict[str, Any]], *, phase: int, study: str, asset: str, representation: str
) -> dict[str, Any]:
    return next(
        row
        for row in curves
        if row["source_phase"] == phase
        and row["study"] == study
        and row["asset"] == asset
        and row["representation"] == representation
    )


def assess_effective_dimension(
    aggregates: list[dict[str, Any]], curves: list[dict[str, Any]]
) -> dict[str, Any]:
    market = _curve(
        curves,
        phase=10,
        study="representation_controls_1h",
        asset="btc",
        representation="market_rff",
    )
    noise = _curve(
        curves,
        phase=10,
        study="representation_controls_1h",
        asset="btc",
        representation="pure_noise",
    )
    mixed = _curve(
        curves,
        phase=10,
        study="representation_controls_1h",
        asset="btc",
        representation="market_plus_noise",
    )
    primary_15m = [
        row
        for row in curves
        if row["source_phase"] == 15 and row["study"] == "native_15m"
    ]
    phase6 = _curve(
        curves,
        phase=6,
        study="kernel_convergence_1h",
        asset="btc",
        representation="market_rff",
    )
    kernel = next(row for row in aggregates if row["representation"] == "exact_rbf_kernel")
    one_million_relative_gap = abs(
        phase6["largest_effective_rank"] - kernel["effective_rank_median"]
    ) / kernel["effective_rank_median"]
    return {
        "phase10_same_n_representation_control": {
            "pn_ratio": market["largest_actual_pn_ratio"],
            "feature_count": market["largest_feature_count"],
            "market_rff_effective_rank": market["largest_effective_rank"],
            "market_rff_effective_rank_fraction_of_n": market[
                "largest_effective_rank_fraction_of_n"
            ],
            "market_rff_nominal_to_effective_dimension": market[
                "largest_nominal_to_effective_dimension"
            ],
            "pure_noise_effective_rank": noise["largest_effective_rank"],
            "pure_noise_effective_rank_fraction_of_n": noise[
                "largest_effective_rank_fraction_of_n"
            ],
            "market_plus_noise_effective_rank": mixed["largest_effective_rank"],
            "market_plus_noise_effective_rank_fraction_of_n": mixed[
                "largest_effective_rank_fraction_of_n"
            ],
            "interpretation": (
                "At matched nominal P and N, market RFF occupy a small spectral subspace while "
                "timestamp-keyed noise nearly fills the centered sample space. The distinction "
                "therefore depends on representation structure, not P alone."
            ),
        },
        "phase15_cross_asset_15m": {
            row["asset"]: {
                "feature_count": row["largest_feature_count"],
                "effective_n": row["effective_n"],
                "effective_rank": row["largest_effective_rank"],
                "effective_rank_fraction_of_n": row[
                    "largest_effective_rank_fraction_of_n"
                ],
                "nominal_to_effective_dimension": row[
                    "largest_nominal_to_effective_dimension"
                ],
            }
            for row in primary_15m
        },
        "phase6_kernel_convergence": {
            "gamma": phase6["gamma"],
            "one_million_effective_rank": phase6["largest_effective_rank"],
            "exact_kernel_effective_rank": kernel["effective_rank_median"],
            "relative_effective_rank_gap": one_million_relative_gap,
            "one_million_nominal_to_effective_dimension": phase6[
                "largest_nominal_to_effective_dimension"
            ],
            "interpretation": (
                "The million-feature gamma=0.2 representation and its exact RBF kernel have "
                "nearly the same spectral participation ratio. This is a within-gamma numerical "
                "anchor and is not substituted for the later gamma=0.5 experiments."
            ),
        },
        "scientific_conclusion": (
            "Nominal predictor count substantially overstates the statistical dimension of "
            "market RFF. Large P/N describes interpolation capacity but is not a faithful measure "
            "of independent market directions; noise controls show that the same P can have a "
            "radically different effective dimension. This dimensional result is not evidence "
            "of predictability or profitability."
        ),
    }


def _write_plot(path: Path, aggregates: list[dict[str, Any]], curves: list[dict[str, Any]]) -> None:
    from plotly import graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Same P, different statistical dimension",
            "Market RFF remain low-dimensional across assets",
            "Nominal P grows far faster than effective rank",
            "Nominal predictors per effective dimension at largest P",
        ),
        vertical_spacing=0.13,
        horizontal_spacing=0.11,
    )
    palette = {"market_rff": "#2563eb", "pure_noise": "#e76f51", "market_plus_noise": "#2a9d8f"}
    for representation, dash in (
        ("market_rff", "solid"),
        ("pure_noise", "dash"),
        ("market_plus_noise", "dot"),
    ):
        selected = [
            row
            for row in aggregates
            if row["source_phase"] == 10 and row["representation"] == representation
        ]
        figure.add_trace(
            go.Scatter(
                x=[row["actual_pn_ratio"] for row in selected],
                y=[row["effective_rank_fraction_of_n_median"] for row in selected],
                mode="lines+markers",
                name=representation.replace("_", " "),
                line={"color": palette[representation], "dash": dash},
                hovertemplate="P/N=%{x:.3g}<br>d_eff/N=%{y:.3%}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    symbols = {"btc": "circle", "eth": "diamond", "gold": "square"}
    for asset in ("btc", "eth", "gold"):
        for study, dash in (("native_15m", "solid"), ("matched_1h_control", "dash")):
            selected = [
                row
                for row in aggregates
                if row["source_phase"] == 15
                and row["asset"] == asset
                and row["study"] == study
            ]
            figure.add_trace(
                go.Scatter(
                    x=[row["actual_pn_ratio"] for row in selected],
                    y=[row["effective_rank_fraction_of_n_median"] for row in selected],
                    mode="lines+markers",
                    name=f"{asset.upper()} {study}",
                    legendgroup=f"phase15-{asset}",
                    line={"dash": dash},
                    marker={"symbol": symbols[asset]},
                    hovertemplate="P/N=%{x:.3g}<br>d_eff/N=%{y:.3%}<extra></extra>",
                ),
                row=1,
                col=2,
            )
    selected_curves = [
        (15, "native_15m", "btc", "BTC 15m gamma=0.5", "#2563eb"),
        (15, "native_15m", "eth", "ETH 15m gamma=0.5", "#7c3aed"),
        (15, "native_15m", "gold", "Gold proxy 15m gamma=0.5", "#ca8a04"),
        (6, "kernel_convergence_1h", "btc", "BTC 1h gamma=0.2", "#2a9d8f"),
    ]
    for phase, study, asset, label, color in selected_curves:
        selected = [
            row
            for row in aggregates
            if row["source_phase"] == phase
            and row["study"] == study
            and row["asset"] == asset
            and row["representation"] == "market_rff"
        ]
        figure.add_trace(
            go.Scatter(
                x=[row["feature_count"] for row in selected],
                y=[row["effective_rank_median"] for row in selected],
                mode="lines+markers",
                name=label,
                line={"color": color},
                hovertemplate="P=%{x:,}<br>d_eff=%{y:.2f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    endpoint_curves = [
        row
        for row in curves
        if row["representation"] == "market_rff"
        and (row["source_phase"] == 6 or row["source_phase"] == 15)
    ]
    labels = [
        (
            "BTC 1h gamma=.2"
            if row["source_phase"] == 6
            else f"{row['asset_label']} {row['study'].replace('_', ' ')}"
        )
        for row in endpoint_curves
    ]
    figure.add_trace(
        go.Bar(
            x=labels,
            y=[row["largest_nominal_to_effective_dimension"] for row in endpoint_curves],
            name="P / d_eff",
            marker={"color": "#64748b"},
            text=[
                f"{row['largest_nominal_to_effective_dimension']:,.0f}x"
                for row in endpoint_curves
            ],
            textposition="outside",
            hovertemplate="%{x}<br>P/d_eff=%{y:,.1f}x<extra></extra>",
        ),
        row=2,
        col=2,
    )
    figure.update_xaxes(type="log", title_text="P/N", row=1, col=1)
    figure.update_xaxes(type="log", title_text="P/N", row=1, col=2)
    figure.update_xaxes(type="log", title_text="Nominal predictors P", row=2, col=1)
    figure.update_yaxes(type="log", title_text="Effective rank", row=2, col=1)
    figure.update_yaxes(type="log", title_text="P / effective rank", row=2, col=2)
    figure.update_yaxes(title_text="Effective rank / N", tickformat=".1%", row=1, col=1)
    figure.update_yaxes(title_text="Effective rank / N", tickformat=".1%", row=1, col=2)
    figure.update_layout(
        title={
            "text": (
                "Phase 20 — hundreds of thousands of predictors collapse to tens of "
                "effective spectral dimensions"
            ),
            "x": 0.5,
        },
        template="plotly_white",
        height=940,
        hovermode="closest",
        legend={"orientation": "h", "y": -0.17},
        margin={"b": 170},
    )
    path.write_text(figure.to_html(include_plotlyjs="cdn"), encoding="utf-8")


def run_phase20(config: Phase20Config) -> dict[str, Any]:
    sources = config.load_and_validate()
    rows, source_counts = collect_effective_dimension_rows(config, sources)
    aggregates = aggregate_effective_dimensions(rows)
    curves = build_curve_summaries(aggregates)
    contrasts = build_phase10_contrasts(aggregates)
    assessment = assess_effective_dimension(aggregates, curves)

    finite_dimension_rows = [row for row in rows if row["feature_count"] is not None]
    checks = {
        **sources["checks"],
        "complete_phase6_finite_rff_rows": source_counts["phase6_finite_rff"]
        == PHASE20_EXPECTED_PHASE6_ROWS,
        "one_exact_kernel_anchor": source_counts["phase6_exact_kernel"] == 1,
        "complete_phase10_rows": source_counts["phase10"] == PHASE20_EXPECTED_PHASE10_ROWS,
        "complete_phase15_rows": source_counts["phase15"] == PHASE20_EXPECTED_PHASE15_ROWS,
        "complete_total_dimension_rows": len(rows) == PHASE20_EXPECTED_TOTAL_ROWS,
        "all_effective_ranks_positive_and_bounded": all(
            row["effective_rank"] > 0
            and row["effective_rank"] <= row["available_rank"] + 1e-6
            for row in rows
        ),
        "all_finite_nominal_metrics_are_finite": all(
            math.isfinite(row["actual_pn_ratio"])
            and math.isfinite(row["nominal_to_effective_dimension"])
            for row in finite_dimension_rows
        ),
        "complete_phase10_matched_contrasts": len(contrasts)
        == PHASE20_EXPECTED_PHASE10_CONTRASTS,
        "million_feature_point_present": any(
            row["source_phase"] == 6 and row["feature_count"] == 1_000_000 for row in rows
        ),
        "phase15_covers_btc_eth_and_gold": {
            row["asset"] for row in rows if row["source_phase"] == 15
        }
        == {"btc", "eth", "gold"},
        "prediction_and_trading_excluded_from_phase20_gate": True,
        "no_new_model_fit_or_holdout_access": True,
    }
    config.output_directory.mkdir(parents=True, exist_ok=True)
    row_path = config.output_directory / "effective_dimension_rows.csv"
    aggregate_path = config.output_directory / "effective_dimension_aggregates.csv"
    curve_path = config.output_directory / "curve_summaries.csv"
    contrast_path = config.output_directory / "representation_contrasts.csv"
    plot_path = config.output_directory / "effective_dimension.html"
    summary_path = config.output_directory / "summary.json"
    _write_csv(row_path, rows)
    _write_csv(aggregate_path, aggregates)
    _write_csv(curve_path, curves)
    _write_csv(contrast_path, contrasts)
    _write_plot(plot_path, aggregates, curves)
    summary = {
        "phase": 20,
        "stage": "complete" if all(checks.values()) else "complete_gate_failed",
        "protocol_version": PHASE20_PROTOCOL_VERSION,
        "objective": "Compare nominal predictor count P with effective statistical dimension",
        "scope": "frozen descriptive post-processing; no fitting and no holdout access",
        "design": {
            "effective_rank_definition": PHASE20_EFFECTIVE_RANK_DEFINITION,
            "centered_rank_ceiling": "min(P, N - 1)",
            "source_phases": [6, 10, 15],
            "phase6_role": "within-gamma=0.2 finite-RFF to exact-kernel anchor",
            "phase10_role": "same-N market, pure-noise, and market-plus-noise controls",
            "phase15_role": "BTC, ETH, and tokenized-gold replication at 15m and matched 1h target",
            "gamma_families_are_not_pooled": True,
            "prediction_or_trading_selected_analysis": False,
            "trading_used_for_inference": False,
            "holdout_used": False,
        },
        "source_counts": source_counts,
        "source_fingerprints_sha256": {
            str(path): _sha256(path) for path in config.source_paths()
        },
        "gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "expected_row_count": PHASE20_EXPECTED_TOTAL_ROWS,
            "observed_row_count": len(rows),
        },
        "assessment": assessment,
        "curve_summaries": curves,
        "limitations": {
            "effective_rank_is_spectral_not_causal": (
                "The participation ratio summarizes eigenvalue concentration; it does not count "
                "causal market factors or establish predictive information."
            ),
            "aggregated_window_diagnostics": (
                "Source rows contain the mean effective rank across rolling windows, so Phase 20 "
                "cannot reconstruct the full within-window distribution."
            ),
            "gamma_families": (
                "The exact kernel anchor uses gamma=0.2 while the later representation and "
                "cross-asset experiments use frozen gamma=0.5. Only within-family comparisons "
                "are interpreted as convergence."
            ),
            "gold_proxy": "Gold is represented by PAXG and has shorter coverage than BTC and ETH.",
            "phase19": (
                "Phase 19 failed one strict CPU maximum-absolute tolerance at P/N=1; Phase 20 "
                "therefore uses the completed float64 source diagnostics descriptively and does "
                "not reinterpret numerical stability as alpha."
            ),
        },
        "artifacts": {
            "summary": str(summary_path),
            "effective_dimension_rows": str(row_path),
            "effective_dimension_aggregates": str(aggregate_path),
            "curve_summaries": str(curve_path),
            "representation_contrasts": str(contrast_path),
            "interactive_plot": str(plot_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
