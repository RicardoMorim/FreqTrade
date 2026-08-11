"""Persistent CUDA worker for the exact centered RBF-kernel limit."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class KernelWorkerState:
    train_inputs: Any
    alpha: Any
    target_mean: Any
    train_column_mean: Any
    train_grand_mean: Any
    gamma: float
    numpy_dtype: np.dtype[Any]
    torch_dtype: Any


class CudaRBFKernelWorker:
    """Fit the exact centered RBF kernel in sample space and retain it for inference."""

    def __init__(self) -> None:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the worker interpreter")
        torch.use_deterministic_algorithms(True)
        self.torch = torch
        self.device = torch.device("cuda:0")
        self.state: KernelWorkerState | None = None

    def _rbf_kernel(self, left: Any, right: Any, gamma: float) -> Any:
        left_norm = (left * left).sum(dim=1, keepdim=True)
        right_norm = (right * right).sum(dim=1).unsqueeze(0)
        squared_distance = (left_norm + right_norm - 2.0 * (left @ right.T)).clamp_min_(0.0)
        return self.torch.exp(-gamma * squared_distance)

    def fit(self, request: dict[str, Any]) -> dict[str, Any]:
        torch = self.torch
        gamma = float(request["gamma"])
        ridge = float(request["ridge"])
        rcond = float(request["rcond"])
        dtype_name = str(request["dtype"])
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if ridge < 0 or rcond <= 0:
            raise ValueError("ridge must be non-negative and rcond must be positive")
        if dtype_name not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
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

        kernel_started = time.perf_counter()
        train_kernel = self._rbf_kernel(train_inputs, train_inputs, gamma)
        torch.cuda.synchronize(self.device)
        kernel_seconds = time.perf_counter() - kernel_started
        centering_started = time.perf_counter()
        train_column_mean = train_kernel.mean(dim=0)
        train_grand_mean = train_kernel.mean()
        centered_kernel = (
            train_kernel
            - train_kernel.mean(dim=1, keepdim=True)
            - train_column_mean.unsqueeze(0)
            + train_grand_mean
        )
        torch.cuda.synchronize(self.device)
        centering_seconds = time.perf_counter() - centering_started

        solver_started = time.perf_counter()
        eigenvalues, eigenvectors = torch.linalg.eigh(centered_kernel)
        largest_eigenvalue = eigenvalues[-1]
        numerical_floor = torch.finfo(torch_dtype).eps * train_inputs.shape[0]
        relative_eigenvalue_threshold = max(rcond**2, numerical_floor)
        retained = eigenvalues > largest_eigenvalue * relative_eigenvalue_threshold
        retained_eigenvalues = eigenvalues[retained]
        retained_eigenvectors = eigenvectors[:, retained]
        denominators = retained_eigenvalues if ridge == 0 else retained_eigenvalues + ridge
        alpha = retained_eigenvectors @ ((retained_eigenvectors.T @ centered_target) / denominators)
        train_prediction = target_mean + centered_kernel @ alpha
        torch.cuda.synchronize(self.device)
        solver_seconds = time.perf_counter() - solver_started
        train_residual = train_prediction - train_target
        coefficient_squared_norm = alpha @ centered_kernel @ alpha
        positive_eigenvalues = retained_eigenvalues.to(torch.float64)
        condition_number = torch.sqrt(positive_eigenvalues[-1] / positive_eigenvalues[0])
        kernel_condition_number = positive_eigenvalues[-1] / positive_eigenvalues[0]
        effective_rank = positive_eigenvalues.sum() ** 2 / positive_eigenvalues.square().sum()
        np.save(
            request["train_prediction_path"],
            train_prediction.detach().cpu().numpy(),
            allow_pickle=False,
        )
        self.state = KernelWorkerState(
            train_inputs=train_inputs,
            alpha=alpha,
            target_mean=target_mean,
            train_column_mean=train_column_mean,
            train_grand_mean=train_grand_mean,
            gamma=gamma,
            numpy_dtype=numpy_dtype,
            torch_dtype=torch_dtype,
        )
        device = torch.cuda.get_device_properties(0)
        return {
            "model": "exact_centered_rbf_kernel",
            "kernel_limit": True,
            "n_train": int(train_inputs.shape[0]),
            "input_dimension": int(train_inputs.shape[1]),
            "rank": int(retained_eigenvalues.numel()),
            "effective_rank": float(effective_rank.item()),
            "condition_number": float(condition_number.item()),
            "kernel_matrix_condition_number": float(kernel_condition_number.item()),
            "coefficient_norm": math.sqrt(max(float(coefficient_squared_norm.item()), 0.0)),
            "scaled_train_mse": float(torch.mean(train_residual.square()).item()),
            "scaled_train_max_abs_error": float(torch.max(torch.abs(train_residual)).item()),
            "relative_eigenvalue_threshold": relative_eigenvalue_threshold,
            "kernel_seconds": kernel_seconds,
            "centering_seconds": centering_seconds,
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
            "gamma": gamma,
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
        cross_kernel = self._rbf_kernel(inputs, state.train_inputs, state.gamma)
        centered_cross_kernel = (
            cross_kernel
            - cross_kernel.mean(dim=1, keepdim=True)
            - state.train_column_mean.unsqueeze(0)
            + state.train_grand_mean
        )
        prediction = state.target_mean + centered_cross_kernel @ state.alpha
        torch.cuda.synchronize(self.device)
        prediction_numpy = prediction.detach().cpu().numpy()
        np.save(request["output_path"], prediction_numpy, allow_pickle=False)
        return {
            "prediction_count": int(prediction_numpy.size),
            "prediction_finite": bool(np.isfinite(prediction_numpy).all()),
            "inference_seconds": time.perf_counter() - started,
        }


def serve() -> int:
    worker = CudaRBFKernelWorker()
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
