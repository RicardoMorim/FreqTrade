"""Synthetic controls for the double-descent research programme.

Phase 1 deliberately excludes market data and FreqAI. Its purpose is to prove that the feature
generator, minimum-norm solver, diagnostics, and automated gate can recover a known interpolation
peak before the same machinery is allowed near financial data.
"""

from __future__ import annotations

import csv
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]

DEFAULT_RATIOS = (
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.98,
    1.00,
    1.02,
    1.05,
    1.10,
    1.25,
    1.50,
    2.00,
    3.00,
    5.00,
    8.00,
)


@dataclass(frozen=True)
class Phase1Config:
    """Frozen configuration for the synthetic validation phase."""

    n_train: int = 128
    n_test: int = 1024
    input_dimension: int = 8
    linear_teacher_dimension: int = 2048
    ratios: tuple[float, ...] = DEFAULT_RATIOS
    repetitions: int = 8
    base_seed: int = 20260810
    noise_standard_deviation: float = 0.5
    gamma: float = 0.2
    rcond: float = 1e-12
    interpolation_tolerance: float = 1e-10
    threshold_band: tuple[float, float] = (0.90, 1.10)
    far_overparameterized_ratio: float = 3.0
    minimum_peak_recovery: float = 0.50
    far_vs_underparameterized_tolerance: float = 1.10

    def validate(self) -> None:
        if self.n_train < 16 or self.n_test < 16:
            raise ValueError("n_train and n_test must both be at least 16")
        if self.input_dimension < 4:
            raise ValueError("input_dimension must be at least 4")
        if self.repetitions < 1:
            raise ValueError("repetitions must be positive")
        if self.noise_standard_deviation <= 0:
            raise ValueError("noise_standard_deviation must be positive")
        if self.gamma <= 0:
            raise ValueError("gamma must be positive")
        if not self.ratios or any(ratio <= 0 for ratio in self.ratios):
            raise ValueError("ratios must contain only positive values")
        if tuple(sorted(set(self.ratios))) != self.ratios:
            raise ValueError("ratios must be sorted and unique")
        max_features = _feature_count(self.ratios[-1], self.n_train)
        if max_features > self.linear_teacher_dimension:
            raise ValueError("linear_teacher_dimension must cover the largest feature count")


@dataclass(frozen=True)
class SyntheticDataset:
    """One fixed train/test draw and its nested feature bank."""

    train_features: FloatArray
    test_features: FloatArray
    train_target: FloatArray
    test_target: FloatArray
    train_inputs: FloatArray | None = None
    test_inputs: FloatArray | None = None


def _feature_count(requested_ratio: float, n_train: int) -> int:
    """Reserve one model parameter for the unpenalized intercept."""
    return max(1, round(requested_ratio * n_train) - 1)


def random_fourier_features(
    inputs: FloatArray,
    feature_count: int,
    gamma: float,
    seed: int,
) -> FloatArray:
    """Create deterministic, prefix-stable RFFs for an RBF kernel."""
    if feature_count < 1:
        raise ValueError("feature_count must be positive")
    if gamma <= 0:
        raise ValueError("gamma must be positive")

    seed_sequence = np.random.SeedSequence(seed)
    projection_seed, phase_seed = seed_sequence.spawn(2)
    projection_rng = np.random.default_rng(projection_seed)
    phase_rng = np.random.default_rng(phase_seed)
    weights = projection_rng.normal(
        loc=0.0,
        scale=math.sqrt(2.0 * gamma),
        size=(feature_count, inputs.shape[1]),
    )
    phases = phase_rng.uniform(0.0, 2.0 * math.pi, size=feature_count)
    return np.sqrt(2.0) * np.cos(inputs @ weights.T + phases)


def rbf_kernel(left: FloatArray, right: FloatArray, gamma: float) -> FloatArray:
    """Compute the RBF kernel without constructing a three-dimensional tensor."""
    left_norm = np.sum(left * left, axis=1)[:, None]
    right_norm = np.sum(right * right, axis=1)[None, :]
    squared_distance = np.maximum(left_norm + right_norm - 2.0 * left @ right.T, 0.0)
    return np.exp(-gamma * squared_distance)


def _linear_dataset(
    config: Phase1Config, repetition_seed: int, max_features: int
) -> SyntheticDataset:
    seed_sequence = np.random.SeedSequence(repetition_seed)
    feature_seed, coefficient_seed, noise_seed = seed_sequence.spawn(3)
    feature_rng = np.random.default_rng(feature_seed)
    coefficient_rng = np.random.default_rng(coefficient_seed)
    noise_rng = np.random.default_rng(noise_seed)

    train_full = feature_rng.normal(size=(config.n_train, config.linear_teacher_dimension))
    test_full = feature_rng.normal(size=(config.n_test, config.linear_teacher_dimension))
    coefficients = coefficient_rng.normal(size=config.linear_teacher_dimension)
    coefficients /= np.linalg.norm(coefficients)
    train_target = train_full @ coefficients + noise_rng.normal(
        scale=config.noise_standard_deviation, size=config.n_train
    )
    test_target = test_full @ coefficients + noise_rng.normal(
        scale=config.noise_standard_deviation, size=config.n_test
    )
    return SyntheticDataset(
        train_features=train_full[:, :max_features],
        test_features=test_full[:, :max_features],
        train_target=train_target,
        test_target=test_target,
    )


def _rff_dataset(
    config: Phase1Config, repetition_seed: int, max_features: int
) -> SyntheticDataset:
    seed_sequence = np.random.SeedSequence(repetition_seed)
    input_seed, noise_seed, rff_seed = seed_sequence.spawn(3)
    input_rng = np.random.default_rng(input_seed)
    noise_rng = np.random.default_rng(noise_seed)
    train_inputs = input_rng.normal(size=(config.n_train, config.input_dimension))
    test_inputs = input_rng.normal(size=(config.n_test, config.input_dimension))

    def teacher(inputs: FloatArray) -> FloatArray:
        return (
            np.sin(inputs[:, 0])
            + 0.5 * np.cos(1.3 * inputs[:, 1])
            + 0.3 * inputs[:, 2] * inputs[:, 3]
        )

    train_target = teacher(train_inputs) + noise_rng.normal(
        scale=config.noise_standard_deviation, size=config.n_train
    )
    test_target = teacher(test_inputs) + noise_rng.normal(
        scale=config.noise_standard_deviation, size=config.n_test
    )
    stable_rff_seed = int(rff_seed.generate_state(1, dtype=np.uint32)[0])
    all_inputs = np.vstack((train_inputs, test_inputs))
    feature_bank = random_fourier_features(
        all_inputs,
        feature_count=max_features,
        gamma=config.gamma,
        seed=stable_rff_seed,
    )
    return SyntheticDataset(
        train_features=feature_bank[: config.n_train],
        test_features=feature_bank[config.n_train :],
        train_target=train_target,
        test_target=test_target,
        train_inputs=train_inputs,
        test_inputs=test_inputs,
    )


def _regression_metrics(target: FloatArray, prediction: FloatArray) -> dict[str, float]:
    residual = prediction - target
    target_variance = float(np.sum((target - np.mean(target)) ** 2))
    correlation = float(np.corrcoef(prediction, target)[0, 1])
    if not np.isfinite(correlation):
        correlation = 0.0
    return {
        "mse": float(np.mean(residual**2)),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1.0 - np.sum(residual**2) / target_variance),
        "information_coefficient": correlation,
        "directional_accuracy": float(np.mean(np.sign(prediction) == np.sign(target))),
    }


def fit_minimum_norm(
    train_features: FloatArray,
    train_target: FloatArray,
    test_features: FloatArray,
    rcond: float,
) -> dict[str, Any]:
    """Fit a ridgeless minimum-norm regressor with an unpenalized intercept."""
    feature_mean = np.mean(train_features, axis=0)
    target_mean = float(np.mean(train_target))
    centered_train = train_features - feature_mean
    centered_target = train_target - target_mean
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        centered_train, centered_target, rcond=rcond
    )
    train_prediction = target_mean + centered_train @ coefficients
    test_prediction = target_mean + (test_features - feature_mean) @ coefficients

    retained = singular_values[:rank]
    condition_number = float(retained[0] / retained[-1]) if retained.size else math.inf
    squared_singular_values = retained**2
    effective_rank = (
        float(np.sum(squared_singular_values) ** 2 / np.sum(squared_singular_values**2))
        if retained.size
        else 0.0
    )
    return {
        "coefficients": coefficients,
        "train_prediction": train_prediction,
        "test_prediction": test_prediction,
        "rank": int(rank),
        "condition_number": condition_number,
        "effective_rank": effective_rank,
        "coefficient_norm": float(np.linalg.norm(coefficients)),
    }


def _kernel_limit_metrics(dataset: SyntheticDataset, config: Phase1Config) -> dict[str, float]:
    if dataset.train_inputs is None or dataset.test_inputs is None:
        raise ValueError("kernel limit requires the original RFF inputs")
    train_kernel = rbf_kernel(dataset.train_inputs, dataset.train_inputs, config.gamma)
    test_kernel = rbf_kernel(dataset.test_inputs, dataset.train_inputs, config.gamma)
    train_column_mean = np.mean(train_kernel, axis=0, keepdims=True)
    train_grand_mean = float(np.mean(train_kernel))
    centered_train_kernel = (
        train_kernel
        - np.mean(train_kernel, axis=1, keepdims=True)
        - train_column_mean
        + train_grand_mean
    )
    centered_test_kernel = (
        test_kernel
        - np.mean(test_kernel, axis=1, keepdims=True)
        - train_column_mean
        + train_grand_mean
    )
    target_mean = float(np.mean(dataset.train_target))
    alpha = np.linalg.lstsq(
        centered_train_kernel,
        dataset.train_target - target_mean,
        rcond=config.rcond,
    )[0]
    prediction = target_mean + centered_test_kernel @ alpha
    metrics = _regression_metrics(dataset.test_target, prediction)
    return {f"kernel_limit_{key}": value for key, value in metrics.items()}


def _kernel_approximation_error(
    dataset: SyntheticDataset, feature_count: int, gamma: float
) -> float:
    if dataset.train_inputs is None:
        return math.nan
    exact_kernel = rbf_kernel(dataset.train_inputs, dataset.train_inputs, gamma)
    features = dataset.train_features[:, :feature_count]
    approximate_kernel = features @ features.T / feature_count
    return float(
        np.linalg.norm(approximate_kernel - exact_kernel, ord="fro")
        / np.linalg.norm(exact_kernel, ord="fro")
    )


def _run_dataset(
    dataset_name: str,
    config: Phase1Config,
    repetition: int,
    repetition_seed: int,
    max_features: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if dataset_name == "linear_gaussian":
        dataset = _linear_dataset(config, repetition_seed, max_features)
        kernel_metrics: dict[str, float] = {}
    elif dataset_name == "nonlinear_rff":
        dataset = _rff_dataset(config, repetition_seed, max_features)
        kernel_metrics = _kernel_limit_metrics(dataset, config)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    rows: list[dict[str, Any]] = []
    for requested_ratio in config.ratios:
        feature_count = _feature_count(requested_ratio, config.n_train)
        started_at = time.perf_counter()
        fitted = fit_minimum_norm(
            dataset.train_features[:, :feature_count],
            dataset.train_target,
            dataset.test_features[:, :feature_count],
            config.rcond,
        )
        fit_seconds = time.perf_counter() - started_at
        train_metrics = _regression_metrics(dataset.train_target, fitted["train_prediction"])
        test_metrics = _regression_metrics(dataset.test_target, fitted["test_prediction"])
        kernel_error = _kernel_approximation_error(dataset, feature_count, config.gamma)
        parameter_count = feature_count + 1
        rows.append(
            {
                "dataset": dataset_name,
                "repetition": repetition,
                "seed": repetition_seed,
                "requested_pn_ratio": requested_ratio,
                "feature_count": feature_count,
                "parameter_count": parameter_count,
                "actual_pn_ratio": parameter_count / config.n_train,
                "train_mse": train_metrics["mse"],
                "oos_mse": test_metrics["mse"],
                "oos_mae": test_metrics["mae"],
                "oos_r2": test_metrics["r2"],
                "oos_information_coefficient": test_metrics["information_coefficient"],
                "oos_directional_accuracy": test_metrics["directional_accuracy"],
                "rank": fitted["rank"],
                "effective_rank": fitted["effective_rank"],
                "condition_number": fitted["condition_number"],
                "coefficient_norm": fitted["coefficient_norm"],
                "interpolates": train_metrics["mse"] <= config.interpolation_tolerance,
                "kernel_relative_frobenius_error": kernel_error,
                "fit_seconds": fit_seconds,
            }
        )
    return rows, kernel_metrics


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    datasets = sorted({str(row["dataset"]) for row in rows})
    ratios = sorted({float(row["requested_pn_ratio"]) for row in rows})
    metrics = (
        "train_mse",
        "oos_mse",
        "oos_mae",
        "oos_r2",
        "oos_information_coefficient",
        "oos_directional_accuracy",
        "rank",
        "effective_rank",
        "condition_number",
        "coefficient_norm",
        "kernel_relative_frobenius_error",
        "fit_seconds",
    )
    for dataset in datasets:
        for ratio in ratios:
            selected = [
                row
                for row in rows
                if row["dataset"] == dataset and row["requested_pn_ratio"] == ratio
            ]
            aggregate_row: dict[str, Any] = {
                "dataset": dataset,
                "requested_pn_ratio": ratio,
                "actual_pn_ratio": float(np.mean([row["actual_pn_ratio"] for row in selected])),
                "feature_count": int(selected[0]["feature_count"]),
                "parameter_count": int(selected[0]["parameter_count"]),
                "interpolation_rate": float(np.mean([row["interpolates"] for row in selected])),
            }
            for metric in metrics:
                values = np.asarray([row[metric] for row in selected], dtype=np.float64)
                finite_values = values[np.isfinite(values)]
                if finite_values.size:
                    aggregate_row[f"{metric}_mean"] = float(np.mean(finite_values))
                    aggregate_row[f"{metric}_median"] = float(np.median(finite_values))
                    aggregate_row[f"{metric}_std"] = float(np.std(finite_values, ddof=0))
                else:
                    aggregate_row[f"{metric}_mean"] = math.nan
                    aggregate_row[f"{metric}_median"] = math.nan
                    aggregate_row[f"{metric}_std"] = math.nan
            result.append(aggregate_row)
    return result


def evaluate_gate(
    aggregate_rows: list[dict[str, Any]], config: Phase1Config
) -> dict[str, Any]:
    """Apply the predeclared, non-visual Phase 1 success criteria."""
    dataset_results: dict[str, Any] = {}
    for dataset in sorted({str(row["dataset"]) for row in aggregate_rows}):
        rows = [row for row in aggregate_rows if row["dataset"] == dataset]
        threshold_rows = [
            row
            for row in rows
            if config.threshold_band[0] <= row["actual_pn_ratio"] <= config.threshold_band[1]
        ]
        under_rows = [row for row in rows if row["actual_pn_ratio"] < config.threshold_band[0]]
        far_rows = [
            row
            for row in rows
            if row["actual_pn_ratio"] >= config.far_overparameterized_ratio
        ]
        interpolating_rows = [row for row in rows if row["interpolation_rate"] >= 0.80]
        if not threshold_rows or not under_rows or not far_rows:
            raise ValueError(
                "ratios must cover under, threshold, and far-overparameterized regimes"
            )
        peak_row = max(rows, key=lambda row: row["oos_mse_mean"])
        farthest_row = max(far_rows, key=lambda row: row["actual_pn_ratio"])
        best_under_row = min(under_rows, key=lambda row: row["oos_mse_mean"])
        first_interpolation_ratio = (
            min(row["actual_pn_ratio"] for row in interpolating_rows)
            if interpolating_rows
            else math.inf
        )
        recovery_fraction = 1.0 - farthest_row["oos_mse_mean"] / peak_row["oos_mse_mean"]
        checks = {
            "interpolation_near_one": (
                config.threshold_band[0]
                <= first_interpolation_ratio
                <= config.threshold_band[1]
            ),
            "oos_peak_near_interpolation": (
                config.threshold_band[0]
                <= peak_row["actual_pn_ratio"]
                <= config.threshold_band[1]
            ),
            "far_regime_recovers_from_peak": (
                recovery_fraction >= config.minimum_peak_recovery
            ),
            "far_regime_matches_best_underparameterized": (
                farthest_row["oos_mse_mean"]
                <= best_under_row["oos_mse_mean"]
                * config.far_vs_underparameterized_tolerance
            ),
        }
        if dataset == "nonlinear_rff":
            first_kernel_error = rows[0]["kernel_relative_frobenius_error_mean"]
            last_kernel_error = rows[-1]["kernel_relative_frobenius_error_mean"]
            checks["rff_converges_toward_kernel"] = last_kernel_error < first_kernel_error

        dataset_results[dataset] = {
            "passed": all(checks.values()),
            "checks": checks,
            "first_interpolation_ratio": first_interpolation_ratio,
            "peak_ratio": peak_row["actual_pn_ratio"],
            "peak_oos_mse": peak_row["oos_mse_mean"],
            "far_ratio": farthest_row["actual_pn_ratio"],
            "far_oos_mse": farthest_row["oos_mse_mean"],
            "best_under_ratio": best_under_row["actual_pn_ratio"],
            "best_under_oos_mse": best_under_row["oos_mse_mean"],
            "peak_recovery_fraction": recovery_fraction,
        }
    return {
        "passed": all(result["passed"] for result in dataset_results.values()),
        "datasets": dataset_results,
    }


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _safe_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return _safe_json_value(value.item())
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _write_plot(path: Path, aggregate_rows: list[dict[str, Any]]) -> bool:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return False

    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("OOS MSE", "Training MSE", "Condition number", "RFF kernel error"),
    )
    colours = {"linear_gaussian": "#2563eb", "nonlinear_rff": "#dc2626"}
    for dataset in sorted({str(row["dataset"]) for row in aggregate_rows}):
        rows = [row for row in aggregate_rows if row["dataset"] == dataset]
        x_values = [row["actual_pn_ratio"] for row in rows]
        colour = colours[dataset]
        for row_number, column_number, metric in (
            (1, 1, "oos_mse_mean"),
            (1, 2, "train_mse_mean"),
            (2, 1, "condition_number_mean"),
        ):
            figure.add_trace(
                go.Scatter(
                    x=x_values,
                    y=[row[metric] for row in rows],
                    mode="lines+markers",
                    name=dataset,
                    legendgroup=dataset,
                    showlegend=(row_number, column_number) == (1, 1),
                    line={"color": colour},
                ),
                row=row_number,
                col=column_number,
            )
        if dataset == "nonlinear_rff":
            figure.add_trace(
                go.Scatter(
                    x=x_values,
                    y=[row["kernel_relative_frobenius_error_mean"] for row in rows],
                    mode="lines+markers",
                    name="RFF kernel approximation",
                    line={"color": "#059669"},
                    showlegend=False,
                ),
                row=2,
                col=2,
            )
    for row_number, column_number in ((1, 1), (1, 2), (2, 1), (2, 2)):
        figure.update_xaxes(type="log", title_text="P / N", row=row_number, col=column_number)
        figure.update_yaxes(type="log", row=row_number, col=column_number)
        figure.add_vline(
            x=1.0,
            line_dash="dash",
            line_color="#6b7280",
            row=row_number,
            col=column_number,
        )
    figure.update_layout(
        title="Phase 1 — Synthetic double descent validation",
        template="plotly_white",
        height=850,
        width=1200,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path, include_plotlyjs="cdn")
    return True


def run_phase1(config: Phase1Config, output_directory: Path | None = None) -> dict[str, Any]:
    """Run both synthetic controls, aggregate repetitions, and evaluate the gate."""
    config.validate()
    started_at = time.perf_counter()
    max_features = _feature_count(config.ratios[-1], config.n_train)
    detailed_rows: list[dict[str, Any]] = []
    kernel_limit_rows: list[dict[str, Any]] = []
    for dataset_index, dataset_name in enumerate(("linear_gaussian", "nonlinear_rff")):
        for repetition in range(config.repetitions):
            repetition_seed = config.base_seed + dataset_index * 100_000 + repetition
            rows, kernel_metrics = _run_dataset(
                dataset_name,
                config,
                repetition,
                repetition_seed,
                max_features,
            )
            detailed_rows.extend(rows)
            if kernel_metrics:
                kernel_limit_rows.append(
                    {
                        "dataset": dataset_name,
                        "repetition": repetition,
                        "seed": repetition_seed,
                        **kernel_metrics,
                    }
                )

    aggregate_rows = _aggregate(detailed_rows)
    gate = evaluate_gate(aggregate_rows, config)
    summary: dict[str, Any] = {
        "phase": 1,
        "objective": "Validate double-descent detection on controlled synthetic data",
        "config": asdict(config),
        "gate": gate,
        "kernel_limit": kernel_limit_rows,
        "runtime_seconds": time.perf_counter() - started_at,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": _package_version("scipy"),
            "plotly": _package_version("plotly"),
        },
        "artifacts": {},
    }

    if output_directory is not None:
        output_directory.mkdir(parents=True, exist_ok=True)
        detailed_path = output_directory / "metrics_detailed.csv"
        aggregate_path = output_directory / "metrics_aggregate.csv"
        summary_path = output_directory / "summary.json"
        plot_path = output_directory / "double_descent.html"
        _write_csv(detailed_path, detailed_rows)
        _write_csv(aggregate_path, aggregate_rows)
        plot_written = _write_plot(plot_path, aggregate_rows)
        summary["artifacts"] = {
            "detailed_metrics": str(detailed_path),
            "aggregate_metrics": str(aggregate_path),
            "interactive_plot": str(plot_path) if plot_written else None,
            "summary": str(summary_path),
        }
        summary_path.write_text(
            json.dumps(_safe_json_value(summary), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return summary
