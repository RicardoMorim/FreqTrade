"""Isolated CPU benchmark for large nested Random Fourier Feature models."""

from __future__ import annotations

import argparse
import base64
import csv
import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import numpy as np
import psutil
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
SolverName = Literal["primal_svd", "streamed_dual"]


@dataclass(frozen=True)
class BenchmarkCase:
    feature_count: int
    solver: SolverName


DEFAULT_CASES = (
    BenchmarkCase(64, "primal_svd"),
    BenchmarkCase(128, "primal_svd"),
    BenchmarkCase(512, "primal_svd"),
    BenchmarkCase(4_096, "primal_svd"),
    BenchmarkCase(4_096, "streamed_dual"),
    BenchmarkCase(10_000, "streamed_dual"),
    BenchmarkCase(100_000, "streamed_dual"),
    BenchmarkCase(250_000, "streamed_dual"),
    BenchmarkCase(500_000, "streamed_dual"),
    BenchmarkCase(1_000_000, "streamed_dual"),
)


@dataclass(frozen=True)
class Phase2Config:
    n_train: int = 128
    n_inference: int = 256
    input_dimension: int = 8
    cases: tuple[BenchmarkCase, ...] = DEFAULT_CASES
    chunk_size: int = 8_192
    base_seed: int = 20260810
    noise_standard_deviation: float = 0.5
    gamma: float = 0.2
    rcond: float = 1e-12
    dtype: str = "float64"
    memory_sample_interval_seconds: float = 0.01
    maximum_ram_fraction: float = 0.50
    overlap_relative_tolerance: float = 1e-6

    def validate(self) -> None:
        if self.n_train < 16 or self.n_inference < 1:
            raise ValueError("n_train must be at least 16 and n_inference must be positive")
        if self.input_dimension < 4:
            raise ValueError("input_dimension must be at least 4")
        if not self.cases:
            raise ValueError("at least one benchmark case is required")
        if any(case.feature_count < 1 for case in self.cases):
            raise ValueError("feature counts must be positive")
        if any(case.solver not in {"primal_svd", "streamed_dual"} for case in self.cases):
            raise ValueError("solver must be primal_svd or streamed_dual")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        if not 0 < self.maximum_ram_fraction <= 1:
            raise ValueError("maximum_ram_fraction must be in (0, 1]")


@dataclass(frozen=True)
class BenchmarkDataset:
    train_inputs: FloatArray
    inference_inputs: FloatArray
    train_target: FloatArray
    inference_target: FloatArray
    rff_seed: int


def _dtype(name: str) -> np.dtype[Any]:
    return np.dtype(name)


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _make_dataset(config: Phase2Config) -> BenchmarkDataset:
    seed_sequence = np.random.SeedSequence(config.base_seed)
    input_seed, noise_seed, rff_seed = seed_sequence.spawn(3)
    input_rng = np.random.default_rng(input_seed)
    noise_rng = np.random.default_rng(noise_seed)
    dtype = _dtype(config.dtype)
    train_inputs = input_rng.normal(size=(config.n_train, config.input_dimension)).astype(dtype)
    inference_inputs = input_rng.normal(
        size=(config.n_inference, config.input_dimension)
    ).astype(dtype)

    def teacher(inputs: NDArray[Any]) -> NDArray[Any]:
        return (
            np.sin(inputs[:, 0])
            + 0.5 * np.cos(1.3 * inputs[:, 1])
            + 0.3 * inputs[:, 2] * inputs[:, 3]
        )

    train_target = (
        teacher(train_inputs)
        + noise_rng.normal(scale=config.noise_standard_deviation, size=config.n_train)
    ).astype(dtype)
    inference_target = (
        teacher(inference_inputs)
        + noise_rng.normal(scale=config.noise_standard_deviation, size=config.n_inference)
    ).astype(dtype)
    return BenchmarkDataset(
        train_inputs=train_inputs,
        inference_inputs=inference_inputs,
        train_target=train_target,
        inference_target=inference_target,
        rff_seed=int(rff_seed.generate_state(1, dtype=np.uint32)[0]),
    )


def iter_rff_parameter_chunks(
    input_dimension: int,
    feature_count: int,
    chunk_size: int,
    gamma: float,
    seed: int,
    dtype: np.dtype[Any],
) -> Iterator[tuple[NDArray[Any], NDArray[Any]]]:
    """Yield prefix-stable projection parameters without allocating all P parameters."""
    seed_sequence = np.random.SeedSequence(seed)
    projection_seed, phase_seed = seed_sequence.spawn(2)
    projection_rng = np.random.default_rng(projection_seed)
    phase_rng = np.random.default_rng(phase_seed)
    remaining = feature_count
    while remaining:
        current_size = min(chunk_size, remaining)
        weights = projection_rng.normal(
            loc=0.0,
            scale=math.sqrt(2.0 * gamma),
            size=(current_size, input_dimension),
        ).astype(dtype)
        phases = phase_rng.uniform(0.0, 2.0 * math.pi, size=current_size).astype(dtype)
        yield weights, phases
        remaining -= current_size


def _rff_chunk(
    inputs: NDArray[Any],
    weights: NDArray[Any],
    phases: NDArray[Any],
    feature_count: int,
) -> NDArray[Any]:
    scale = math.sqrt(2.0 / feature_count)
    return scale * np.cos(inputs @ weights.T + phases)


def _spectral_diagnostics_from_singular_values(
    singular_values: NDArray[Any], rank: int
) -> tuple[float, float]:
    retained = singular_values[:rank]
    if not retained.size:
        return math.inf, 0.0
    condition_number = float(retained[0] / retained[-1])
    squared = retained.astype(np.float64) ** 2
    effective_rank = float(np.sum(squared) ** 2 / np.sum(squared**2))
    return condition_number, effective_rank


def solve_primal(
    dataset: BenchmarkDataset,
    case: BenchmarkCase,
    config: Phase2Config,
) -> dict[str, Any]:
    """Materialize Z and solve minimum-norm least squares by direct SVD."""
    dtype = _dtype(config.dtype)
    generation_started = time.perf_counter()
    chunks = []
    all_inputs = np.vstack((dataset.train_inputs, dataset.inference_inputs))
    for weights, phases in iter_rff_parameter_chunks(
        config.input_dimension,
        case.feature_count,
        config.chunk_size,
        config.gamma,
        dataset.rff_seed,
        dtype,
    ):
        chunks.append(_rff_chunk(all_inputs, weights, phases, case.feature_count))
    all_features = np.hstack(chunks)
    generation_seconds = time.perf_counter() - generation_started

    train_features = all_features[: config.n_train]
    inference_features = all_features[config.n_train :]
    feature_mean = np.mean(train_features, axis=0)
    target_mean = float(np.mean(dataset.train_target))
    centered_train = train_features - feature_mean
    centered_target = dataset.train_target - target_mean
    solver_started = time.perf_counter()
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        centered_train, centered_target, rcond=config.rcond
    )
    solver_seconds = time.perf_counter() - solver_started
    train_prediction = target_mean + centered_train @ coefficients
    inference_started = time.perf_counter()
    inference_prediction = target_mean + (inference_features - feature_mean) @ coefficients
    inference_seconds = time.perf_counter() - inference_started
    condition_number, effective_rank = _spectral_diagnostics_from_singular_values(
        singular_values, int(rank)
    )
    return {
        "train_prediction": train_prediction,
        "inference_prediction": inference_prediction,
        "rank": int(rank),
        "condition_number": condition_number,
        "effective_rank": effective_rank,
        "coefficient_norm": float(np.linalg.norm(coefficients)),
        "rff_generation_seconds": generation_seconds,
        "training_solver_seconds": solver_seconds,
        "training_total_seconds": generation_seconds + solver_seconds,
        "inference_seconds": inference_seconds,
        "inference_generation_seconds": 0.0,
    }


def solve_streamed_dual(
    dataset: BenchmarkDataset,
    case: BenchmarkCase,
    config: Phase2Config,
) -> dict[str, Any]:
    """Fit in sample space while streaming feature chunks and bounding memory by chunk size."""
    dtype = _dtype(config.dtype)
    target_mean = float(np.mean(dataset.train_target))
    centered_target = dataset.train_target - target_mean
    gram = np.zeros((config.n_train, config.n_train), dtype=dtype)
    generation_seconds = 0.0
    algebra_seconds = 0.0
    training_started = time.perf_counter()
    for weights, phases in iter_rff_parameter_chunks(
        config.input_dimension,
        case.feature_count,
        config.chunk_size,
        config.gamma,
        dataset.rff_seed,
        dtype,
    ):
        generation_started = time.perf_counter()
        train_chunk = _rff_chunk(
            dataset.train_inputs, weights, phases, case.feature_count
        )
        generation_seconds += time.perf_counter() - generation_started
        algebra_started = time.perf_counter()
        centered_chunk = train_chunk - np.mean(train_chunk, axis=0)
        gram += centered_chunk @ centered_chunk.T
        algebra_seconds += time.perf_counter() - algebra_started
    solve_started = time.perf_counter()
    alpha, _, rank, _ = np.linalg.lstsq(gram, centered_target, rcond=config.rcond)
    algebra_seconds += time.perf_counter() - solve_started
    training_total_seconds = time.perf_counter() - training_started
    train_prediction = target_mean + gram @ alpha

    inference_started = time.perf_counter()
    inference_generation_seconds = 0.0
    inference_prediction = np.full(config.n_inference, target_mean, dtype=dtype)
    coefficient_squared_norm = 0.0
    for weights, phases in iter_rff_parameter_chunks(
        config.input_dimension,
        case.feature_count,
        config.chunk_size,
        config.gamma,
        dataset.rff_seed,
        dtype,
    ):
        generation_started = time.perf_counter()
        train_chunk = _rff_chunk(
            dataset.train_inputs, weights, phases, case.feature_count
        )
        inference_chunk = _rff_chunk(
            dataset.inference_inputs, weights, phases, case.feature_count
        )
        inference_generation_seconds += time.perf_counter() - generation_started
        feature_mean = np.mean(train_chunk, axis=0)
        centered_train_chunk = train_chunk - feature_mean
        centered_inference_chunk = inference_chunk - feature_mean
        coefficient_chunk = centered_train_chunk.T @ alpha
        inference_prediction += centered_inference_chunk @ coefficient_chunk
        coefficient_squared_norm += float(coefficient_chunk @ coefficient_chunk)
    inference_seconds = time.perf_counter() - inference_started

    eigenvalues = np.linalg.eigvalsh(gram).astype(np.float64)
    positive = eigenvalues[eigenvalues > np.max(eigenvalues) * config.rcond]
    singular_values = np.sqrt(positive[::-1])
    condition_number, effective_rank = _spectral_diagnostics_from_singular_values(
        singular_values, singular_values.size
    )
    return {
        "train_prediction": train_prediction,
        "inference_prediction": inference_prediction,
        "rank": int(rank),
        "condition_number": condition_number,
        "effective_rank": effective_rank,
        "coefficient_norm": math.sqrt(coefficient_squared_norm),
        "rff_generation_seconds": generation_seconds,
        "training_solver_seconds": algebra_seconds,
        "training_total_seconds": training_total_seconds,
        "inference_seconds": inference_seconds,
        "inference_generation_seconds": inference_generation_seconds,
    }


def benchmark_case(case: BenchmarkCase, config: Phase2Config) -> dict[str, Any]:
    """Run one benchmark case inside a fresh process."""
    config.validate()
    dataset = _make_dataset(config)
    computation_started = time.perf_counter()
    if case.solver == "primal_svd":
        result = solve_primal(dataset, case, config)
    elif case.solver == "streamed_dual":
        result = solve_streamed_dual(dataset, case, config)
    else:
        raise ValueError(f"unknown solver: {case.solver}")
    total_compute_seconds = time.perf_counter() - computation_started
    train_residual = result["train_prediction"] - dataset.train_target
    inference_residual = result["inference_prediction"] - dataset.inference_target
    inference_prediction = result.pop("inference_prediction")
    result.pop("train_prediction")
    dtype_bytes = _dtype(config.dtype).itemsize
    materialized_bytes = (
        (config.n_train + config.n_inference) * case.feature_count * dtype_bytes
    )
    probe_size = min(16, inference_prediction.size)
    return {
        "feature_count": case.feature_count,
        "pn_ratio": case.feature_count / config.n_train,
        "solver": case.solver,
        "backend": "numpy_cpu",
        "dtype": config.dtype,
        "n_train": config.n_train,
        "n_inference": config.n_inference,
        "input_dimension": config.input_dimension,
        "chunk_size": min(config.chunk_size, case.feature_count),
        **result,
        "train_mse": float(np.mean(train_residual**2)),
        "oos_mse": float(np.mean(inference_residual**2)),
        "prediction_finite": bool(np.all(np.isfinite(inference_prediction))),
        "prediction_probe": [float(value) for value in inference_prediction[:probe_size]],
        "prediction_checksum": float(np.sum(inference_prediction, dtype=np.float64)),
        "total_compute_seconds": total_compute_seconds,
        "materialized_design_gib": materialized_bytes / 2**30,
        "parameter_vector_mib": case.feature_count * dtype_bytes / 2**20,
        "features_per_training_second": (
            case.feature_count / result["training_total_seconds"]
            if result["training_total_seconds"] > 0
            else math.inf
        ),
    }


def _encode_worker_payload(case: BenchmarkCase, config: Phase2Config) -> str:
    payload = {"case": asdict(case), "config": asdict(config)}
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()


def _decode_worker_payload(payload: str) -> tuple[BenchmarkCase, Phase2Config]:
    decoded = json.loads(base64.urlsafe_b64decode(payload.encode()).decode())
    case = BenchmarkCase(**decoded["case"])
    config_values = decoded["config"]
    config_values["cases"] = tuple(BenchmarkCase(**item) for item in config_values["cases"])
    return case, Phase2Config(**config_values)


def _process_tree_rss(process: psutil.Process) -> int:
    try:
        processes = [process, *process.children(recursive=True)]
        return sum(item.memory_info().rss for item in processes if item.is_running())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0


def run_isolated_case(case: BenchmarkCase, config: Phase2Config) -> dict[str, Any]:
    """Execute one case in a subprocess and sample its process-tree RSS."""
    payload = _encode_worker_payload(case, config)
    command = [
        sys.executable,
        "-m",
        "research.double_descent.benchmark",
        "--worker-payload",
        payload,
    ]
    wall_started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=Path.cwd(),
        env=os.environ.copy(),
    )
    monitored = psutil.Process(process.pid)
    peak_rss = _process_tree_rss(monitored)
    while process.poll() is None:
        peak_rss = max(peak_rss, _process_tree_rss(monitored))
        time.sleep(config.memory_sample_interval_seconds)
    peak_rss = max(peak_rss, _process_tree_rss(monitored))
    stdout, stderr = process.communicate()
    wall_seconds = time.perf_counter() - wall_started
    if process.returncode != 0:
        return {
            "feature_count": case.feature_count,
            "solver": case.solver,
            "success": False,
            "returncode": process.returncode,
            "stderr": stderr.strip(),
            "wall_seconds": wall_seconds,
            "peak_rss_mib": peak_rss / 2**20,
        }
    result = json.loads(stdout)
    result.update(
        {
            "success": True,
            "returncode": 0,
            "stderr": stderr.strip(),
            "wall_seconds": wall_seconds,
            "peak_rss_mib": peak_rss / 2**20,
            "vram_measured": False,
            "peak_vram_mib": None,
        }
    )
    return result


def hardware_inventory() -> dict[str, Any]:
    """Describe hardware and distinguish present GPUs from usable Python backends."""
    virtual_memory = psutil.virtual_memory()
    gpu_rows: list[dict[str, Any]] = []
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        completed = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            for line in completed.stdout.splitlines():
                name, memory_total, driver = (item.strip() for item in line.split(",", 2))
                gpu_rows.append(
                    {
                        "name": name,
                        "memory_total_mib": float(memory_total),
                        "driver_version": driver,
                    }
                )
    torch_version = _package_version("torch")
    torch_cuda_available = False
    if torch_version:
        try:
            import torch

            torch_cuda_available = bool(torch.cuda.is_available())
        except (ImportError, OSError):
            torch_cuda_available = False
    cupy_present = importlib.util.find_spec("cupy") is not None
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_total_gib": virtual_memory.total / 2**30,
        "ram_available_at_start_gib": virtual_memory.available / 2**30,
        "numpy": np.__version__,
        "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "gpus": gpu_rows,
        "torch": torch_version,
        "torch_cuda_available": torch_cuda_available,
        "cupy_present": cupy_present,
        "cuda_python_backend_available": cupy_present or torch_cuda_available,
        "vram_measurement": "not_applicable_cpu_backend",
    }


def evaluate_benchmark_gate(
    results: list[dict[str, Any]], config: Phase2Config, hardware: dict[str, Any]
) -> dict[str, Any]:
    successful = [row for row in results if row.get("success")]
    target_maximum = max(case.feature_count for case in config.cases)
    completed_maximum = max((row["feature_count"] for row in successful), default=0)
    all_finite = all(row.get("prediction_finite", False) for row in successful)
    max_peak_rss = max((row["peak_rss_mib"] for row in successful), default=math.inf)
    ram_limit_mib = hardware["ram_total_gib"] * 1024 * config.maximum_ram_fraction

    solver_pairs: dict[int, dict[str, dict[str, Any]]] = {}
    for row in successful:
        solver_pairs.setdefault(row["feature_count"], {})[row["solver"]] = row
    overlaps = [pair for pair in solver_pairs.values() if len(pair) == 2]
    overlap_checks = []
    for pair in overlaps:
        primal = np.asarray(pair["primal_svd"]["prediction_probe"])
        dual = np.asarray(pair["streamed_dual"]["prediction_probe"])
        overlap_checks.append(
            bool(
                np.allclose(
                    primal,
                    dual,
                    rtol=config.overlap_relative_tolerance,
                    atol=config.overlap_relative_tolerance,
                )
            )
        )

    largest_rows = [row for row in successful if row["feature_count"] == completed_maximum]
    largest_streamed = next(
        (row for row in largest_rows if row["solver"] == "streamed_dual"), None
    )
    streaming_bounded = bool(
        largest_streamed
        and largest_streamed["peak_rss_mib"]
        < largest_streamed["materialized_design_gib"] * 1024
    )
    checks = {
        "all_cases_succeeded": len(successful) == len(results) == len(config.cases),
        "all_predictions_finite": all_finite,
        "target_maximum_completed": completed_maximum == target_maximum,
        "peak_ram_within_budget": max_peak_rss <= ram_limit_mib,
        "primal_dual_overlap_matches": bool(overlap_checks) and all(overlap_checks),
        "streaming_memory_is_bounded": streaming_bounded,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "target_maximum_features": target_maximum,
        "completed_maximum_features": completed_maximum,
        "maximum_peak_rss_mib": max_peak_rss,
        "ram_budget_mib": ram_limit_mib,
        "overlap_case_count": len(overlap_checks),
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
    scalar_rows = []
    for row in rows:
        scalar_rows.append(
            {
                key: json.dumps(value) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
        )
    fieldnames = sorted({key for row in scalar_rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scalar_rows)


def _write_plot(path: Path, results: list[dict[str, Any]]) -> bool:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return False
    successful = [row for row in results if row.get("success")]
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Wall time", "Peak process RAM", "Training throughput"),
    )
    colours = {"primal_svd": "#2563eb", "streamed_dual": "#dc2626"}
    for solver in ("primal_svd", "streamed_dual"):
        rows = sorted(
            (row for row in successful if row["solver"] == solver),
            key=lambda row: row["feature_count"],
        )
        if not rows:
            continue
        for column, metric in (
            (1, "wall_seconds"),
            (2, "peak_rss_mib"),
            (3, "features_per_training_second"),
        ):
            figure.add_trace(
                go.Scatter(
                    x=[row["feature_count"] for row in rows],
                    y=[row[metric] for row in rows],
                    mode="lines+markers",
                    name=solver,
                    legendgroup=solver,
                    showlegend=column == 1,
                    line={"color": colours[solver]},
                ),
                row=1,
                col=column,
            )
    for column in (1, 2, 3):
        figure.update_xaxes(type="log", title_text="Random features P", row=1, col=column)
        figure.update_yaxes(type="log", row=1, col=column)
    figure.update_layout(
        title="Phase 2 — RFF computational benchmark",
        template="plotly_white",
        height=500,
        width=1400,
    )
    figure.write_html(path, include_plotlyjs="cdn")
    return True


def run_phase2(config: Phase2Config, output_directory: Path | None = None) -> dict[str, Any]:
    """Run all benchmark cases in isolated subprocesses and persist the decision artifacts."""
    config.validate()
    hardware = hardware_inventory()
    started_at = time.perf_counter()
    results = [run_isolated_case(case, config) for case in config.cases]
    gate = evaluate_benchmark_gate(results, config, hardware)
    successful = [row for row in results if row.get("success")]
    largest = max(successful, key=lambda row: row["feature_count"]) if successful else None
    longest_wall_seconds = max((row["wall_seconds"] for row in successful), default=math.inf)
    if longest_wall_seconds <= 60:
        session_classification = "short"
    elif longest_wall_seconds <= 600:
        session_classification = "standard"
    else:
        session_classification = "long"
    summary: dict[str, Any] = {
        "phase": 2,
        "objective": (
            "Measure RFF generation, fit, inference, and peak memory on "
            "available hardware"
        ),
        "config": asdict(config),
        "hardware": hardware,
        "results": results,
        "gate": gate,
        "runtime_seconds": time.perf_counter() - started_at,
        "recommendation": {
            "session_classification_at_maximum": session_classification,
            "maximum_tested_features": largest["feature_count"] if largest else 0,
            "preferred_large_p_solver": "streamed_dual",
            "gpu_benchmark_required_later": bool(hardware["gpus"]),
            "gpu_blocker": (
                None
                if hardware["cuda_python_backend_available"]
                else "GPU present but no CUDA-enabled Python array backend is installed"
            ),
        },
        "artifacts": {},
    }
    if output_directory is not None:
        output_directory.mkdir(parents=True, exist_ok=True)
        csv_path = output_directory / "benchmark.csv"
        summary_path = output_directory / "summary.json"
        plot_path = output_directory / "benchmark.html"
        _write_csv(csv_path, results)
        plot_written = _write_plot(plot_path, results)
        summary["artifacts"] = {
            "benchmark": str(csv_path),
            "summary": str(summary_path),
            "interactive_plot": str(plot_path) if plot_written else None,
        }
        summary_path.write_text(
            json.dumps(_safe_json_value(summary), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return summary


def _worker_main(payload: str) -> int:
    case, config = _decode_worker_payload(payload)
    print(json.dumps(_safe_json_value(benchmark_case(case, config)), sort_keys=True))
    return 0


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-payload")
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    if not arguments.worker_payload:
        raise SystemExit("benchmark module is an internal worker; use the phase 2 runner")
    return _worker_main(arguments.worker_payload)


if __name__ == "__main__":
    raise SystemExit(main())
