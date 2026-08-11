"""Persistent CUDA worker for streamed nested Random Fourier Feature regression."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def iter_nested_rff_parameters(
    input_dimension: int,
    feature_count: int,
    chunk_size: int,
    gamma: float,
    seed: int,
    dtype: np.dtype[Any],
) -> Iterator[tuple[NDArray[Any], NDArray[Any]]]:
    """Yield chunk-size-independent prefixes of one deterministic RFF sequence."""
    if input_dimension < 1 or feature_count < 1 or chunk_size < 1:
        raise ValueError("RFF dimensions and chunk size must be positive")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
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


@dataclass
class WorkerState:
    train_inputs: Any
    alpha: Any
    target_mean: Any
    feature_count: int
    chunk_size: int
    gamma: float
    seed: int
    numpy_dtype: np.dtype[Any]
    torch_dtype: Any


class CudaRFFWorker:
    """Fit RFF ridge models in sample space and retain one model for inference."""

    def __init__(self) -> None:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the worker interpreter")
        torch.use_deterministic_algorithms(True)
        self.torch = torch
        self.device = torch.device("cuda:0")
        self.state: WorkerState | None = None

    def _rff_chunk(self, inputs: Any, weights: NDArray[Any], phases: NDArray[Any]) -> Any:
        torch = self.torch
        weight_tensor = torch.as_tensor(weights, dtype=inputs.dtype, device=self.device)
        phase_tensor = torch.as_tensor(phases, dtype=inputs.dtype, device=self.device)
        return math.sqrt(2.0) * torch.cos(inputs @ weight_tensor.T + phase_tensor)

    def fit(self, request: dict[str, Any]) -> dict[str, Any]:
        torch = self.torch
        feature_count = int(request["feature_count"])
        chunk_size = min(int(request["chunk_size"]), feature_count)
        gamma = float(request["gamma"])
        seed = int(request["seed"])
        ridge = float(request["ridge"])
        rcond = float(request["rcond"])
        dtype_name = str(request["dtype"])
        if dtype_name not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        if ridge < 0 or rcond <= 0:
            raise ValueError("ridge must be non-negative and rcond must be positive")
        numpy_dtype = np.dtype(dtype_name)
        torch_dtype = torch.float64 if dtype_name == "float64" else torch.float32
        inputs_numpy = np.load(request["inputs_path"], allow_pickle=False).astype(
            numpy_dtype, copy=False
        )
        target_numpy = np.load(request["target_path"], allow_pickle=False).astype(
            numpy_dtype, copy=False
        )
        if inputs_numpy.ndim != 2 or target_numpy.ndim != 1:
            raise ValueError("training inputs must be 2D and target must be 1D")
        if inputs_numpy.shape[0] != target_numpy.shape[0]:
            raise ValueError("training inputs and target lengths differ")
        if not np.isfinite(inputs_numpy).all() or not np.isfinite(target_numpy).all():
            raise ValueError("training arrays contain non-finite values")

        self.state = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        train_inputs = torch.as_tensor(inputs_numpy, dtype=torch_dtype, device=self.device)
        train_target = torch.as_tensor(target_numpy, dtype=torch_dtype, device=self.device)
        target_mean = torch.mean(train_target)
        centered_target = train_target - target_mean
        sample_count = train_inputs.shape[0]
        gram = torch.zeros((sample_count, sample_count), dtype=torch_dtype, device=self.device)
        feature_generation_seconds = 0.0
        gram_seconds = 0.0
        for weights, phases in iter_nested_rff_parameters(
            inputs_numpy.shape[1],
            feature_count,
            chunk_size,
            gamma,
            seed,
            numpy_dtype,
        ):
            generation_started = time.perf_counter()
            chunk = self._rff_chunk(train_inputs, weights, phases)
            torch.cuda.synchronize(self.device)
            feature_generation_seconds += time.perf_counter() - generation_started
            gram_started = time.perf_counter()
            centered_chunk = chunk - torch.mean(chunk, dim=0)
            gram.addmm_(centered_chunk, centered_chunk.T, alpha=1.0 / feature_count)
            torch.cuda.synchronize(self.device)
            gram_seconds += time.perf_counter() - gram_started

        solver_started = time.perf_counter()
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        largest_eigenvalue = eigenvalues[-1]
        numerical_floor = torch.finfo(torch_dtype).eps * sample_count
        relative_eigenvalue_threshold = max(rcond**2, numerical_floor)
        retained = eigenvalues > largest_eigenvalue * relative_eigenvalue_threshold
        retained_eigenvalues = eigenvalues[retained]
        retained_eigenvectors = eigenvectors[:, retained]
        denominators = retained_eigenvalues if ridge == 0 else retained_eigenvalues + ridge
        projected_target = retained_eigenvectors.T @ centered_target
        spectral_coefficients = projected_target / denominators
        alpha = retained_eigenvectors @ spectral_coefficients
        train_prediction = target_mean + gram @ alpha
        torch.cuda.synchronize(self.device)
        solver_seconds = time.perf_counter() - solver_started
        train_residual = train_prediction - train_target
        normalized_coefficient_squared_norm = alpha @ gram @ alpha
        raw_coefficient_squared_norm = normalized_coefficient_squared_norm / feature_count
        positive_eigenvalues = retained_eigenvalues.to(torch.float64)
        condition_number = torch.sqrt(positive_eigenvalues[-1] / positive_eigenvalues[0])
        effective_rank = positive_eigenvalues.sum() ** 2 / (positive_eigenvalues.square().sum())
        ridge_denominators = positive_eigenvalues + ridge
        ridge_effective_degrees_of_freedom = torch.sum(positive_eigenvalues / ridge_denominators)
        ridge_system_condition_number = ridge_denominators[-1] / ridge_denominators[0]
        np.save(
            request["train_prediction_path"],
            train_prediction.detach().cpu().numpy(),
            allow_pickle=False,
        )
        self.state = WorkerState(
            train_inputs=train_inputs,
            alpha=alpha,
            target_mean=target_mean,
            feature_count=feature_count,
            chunk_size=chunk_size,
            gamma=gamma,
            seed=seed,
            numpy_dtype=numpy_dtype,
            torch_dtype=torch_dtype,
        )
        device = torch.cuda.get_device_properties(0)
        return {
            "n_train": int(sample_count),
            "input_dimension": int(inputs_numpy.shape[1]),
            "feature_count": feature_count,
            "rank": int(retained_eigenvalues.numel()),
            "effective_rank": float(effective_rank.item()),
            "condition_number": float(condition_number.item()),
            "coefficient_norm": math.sqrt(max(float(raw_coefficient_squared_norm.item()), 0.0)),
            "normalized_feature_coefficient_norm": math.sqrt(
                max(float(normalized_coefficient_squared_norm.item()), 0.0)
            ),
            "ridge_effective_degrees_of_freedom": float(ridge_effective_degrees_of_freedom.item()),
            "ridge_system_condition_number": float(ridge_system_condition_number.item()),
            "scaled_train_mse": float(torch.mean(train_residual.square()).item()),
            "scaled_train_max_abs_error": float(torch.max(torch.abs(train_residual)).item()),
            "relative_eigenvalue_threshold": relative_eigenvalue_threshold,
            "feature_generation_seconds": feature_generation_seconds,
            "gram_seconds": gram_seconds,
            "solver_seconds": solver_seconds,
            "training_total_seconds": time.perf_counter() - started,
            "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_vram_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "cuda_device": device.name,
            "cuda_compute_capability": ".".join(
                str(value) for value in torch.cuda.get_device_capability(0)
            ),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "dtype": dtype_name,
            "ridge": ridge,
        }

    def predict(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.state is None:
            raise RuntimeError("predict requested before fit")
        torch = self.torch
        state = self.state
        inputs_numpy = np.load(request["inputs_path"], allow_pickle=False).astype(
            state.numpy_dtype, copy=False
        )
        if inputs_numpy.ndim != 2 or inputs_numpy.shape[1] != state.train_inputs.shape[1]:
            raise ValueError("prediction input dimensions do not match training")
        if not np.isfinite(inputs_numpy).all():
            raise ValueError("prediction inputs contain non-finite values")
        started = time.perf_counter()
        inputs = torch.as_tensor(inputs_numpy, dtype=state.torch_dtype, device=self.device)
        prediction = torch.full(
            (inputs.shape[0],),
            float(state.target_mean.item()),
            dtype=state.torch_dtype,
            device=self.device,
        )
        for weights, phases in iter_nested_rff_parameters(
            state.train_inputs.shape[1],
            state.feature_count,
            state.chunk_size,
            state.gamma,
            state.seed,
            state.numpy_dtype,
        ):
            train_chunk = self._rff_chunk(state.train_inputs, weights, phases)
            inference_chunk = self._rff_chunk(inputs, weights, phases)
            feature_mean = torch.mean(train_chunk, dim=0)
            coefficient_chunk = (train_chunk - feature_mean).T @ state.alpha
            coefficient_chunk /= state.feature_count
            prediction.add_((inference_chunk - feature_mean) @ coefficient_chunk)
        torch.cuda.synchronize(self.device)
        prediction_numpy = prediction.detach().cpu().numpy()
        np.save(request["output_path"], prediction_numpy, allow_pickle=False)
        return {
            "prediction_count": int(prediction_numpy.size),
            "prediction_finite": bool(np.isfinite(prediction_numpy).all()),
            "inference_seconds": time.perf_counter() - started,
        }


def serve() -> int:
    worker = CudaRFFWorker()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            command = request.pop("command")
            if command == "fit":
                result = worker.fit(request)
            elif command == "predict":
                result = worker.predict(request)
            elif command == "shutdown":
                print(json.dumps({"ok": True, "result": {"shutdown": True}}), flush=True)
                return 0
            else:
                raise ValueError(f"unknown command: {command}")
            print(json.dumps({"ok": True, "result": result}), flush=True)
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                    }
                ),
                flush=True,
            )
    return 0


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_arguments()
    if not arguments.serve:
        raise SystemExit("this module is an internal worker; pass --serve")
    raise SystemExit(serve())
