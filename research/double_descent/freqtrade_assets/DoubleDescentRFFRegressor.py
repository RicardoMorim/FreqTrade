from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen


logger = logging.getLogger(__name__)


def _to_numpy_2d(value: Any) -> np.ndarray:
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D features, got shape={array.shape}")
    return np.ascontiguousarray(array)


def _to_numpy_1d(value: Any) -> np.ndarray:
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    return np.ascontiguousarray(np.asarray(value, dtype=np.float32).reshape(-1))


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DoubleDescentRFFRegressor requested CUDA but CUDA is unavailable")
    return torch.device(name)


def _dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}") from exc


def _solve(kernel: torch.Tensor, y: torch.Tensor, ridge: float, rtol: float) -> torch.Tensor:
    if ridge > 0:
        eye = torch.eye(kernel.shape[0], dtype=kernel.dtype, device=kernel.device)
        return torch.linalg.solve(kernel + ridge * eye, y)
    return torch.linalg.pinv(kernel, hermitian=True, rtol=rtol) @ y


class _RFFEstimator:
    def __init__(self, params: dict[str, Any]) -> None:
        self.p = int(params.get("n_random_features", 1024))
        self.gamma = float(params.get("gamma", 0.5))
        self.ridge = float(params.get("ridge_lambda", 1e-6))
        self.seed = int(params.get("seed", 42))
        self.chunk_size = int(params.get("chunk_size", 4096))
        self.device_name = str(params.get("device", "auto"))
        self.feature_dtype_name = str(params.get("feature_dtype", "float32"))
        self.acc_dtype_name = str(params.get("accumulator_dtype", "float32"))
        self.pinv_rtol = float(params.get("pinv_rtol", 1e-6))
        self.compute_diagnostics = bool(params.get("compute_diagnostics", False))
        if self.p <= 0 or self.chunk_size <= 0 or self.gamma <= 0 or self.ridge < 0:
            raise ValueError("Invalid RFF model parameters")
        self.train_x: np.ndarray | None = None
        self.alpha: np.ndarray | None = None
        self.diagnostics: dict[str, Any] = {}

    def _weights(self, device: torch.device, input_dim: int):
        gen = torch.Generator(device=device)
        gen.manual_seed(self.seed)
        dtype = _dtype(self.feature_dtype_name)
        produced = 0
        omega_scale = math.sqrt(2.0 * self.gamma)
        while produced < self.p:
            width = min(self.chunk_size, self.p - produced)
            omega = torch.randn(
                (width, input_dim), generator=gen, dtype=dtype, device=device
            ).mul_(omega_scale)
            bias = torch.rand((width,), generator=gen, dtype=dtype, device=device).mul_(
                2.0 * math.pi
            )
            yield omega, bias
            produced += width

    def _gram(self, x: np.ndarray) -> torch.Tensor:
        device = _device(self.device_name)
        feature_dtype = _dtype(self.feature_dtype_name)
        acc_dtype = _dtype(self.acc_dtype_name)
        x_t = torch.as_tensor(x, dtype=feature_dtype, device=device)
        gram = torch.zeros((len(x), len(x)), dtype=acc_dtype, device=device)
        scale = math.sqrt(2.0 / self.p)
        for omega, bias in self._weights(device, x.shape[1]):
            z = torch.cos(x_t @ omega.T + bias).mul_(scale)
            gram.addmm_(z.to(acc_dtype), z.to(acc_dtype).T)
        return gram

    def _cross(self, x: np.ndarray, train_x: np.ndarray) -> torch.Tensor:
        device = _device(self.device_name)
        feature_dtype = _dtype(self.feature_dtype_name)
        acc_dtype = _dtype(self.acc_dtype_name)
        x_t = torch.as_tensor(x, dtype=feature_dtype, device=device)
        train_t = torch.as_tensor(train_x, dtype=feature_dtype, device=device)
        kernel = torch.zeros((len(x), len(train_x)), dtype=acc_dtype, device=device)
        scale = math.sqrt(2.0 / self.p)
        for omega, bias in self._weights(device, x.shape[1]):
            z_x = torch.cos(x_t @ omega.T + bias).mul_(scale)
            z_train = torch.cos(train_t @ omega.T + bias).mul_(scale)
            kernel.addmm_(z_x.to(acc_dtype), z_train.to(acc_dtype).T)
        return kernel

    def fit(self, x: Any, y: Any) -> "_RFFEstimator":
        x_np = _to_numpy_2d(x)
        y_np = _to_numpy_1d(y)
        gram = self._gram(x_np)
        y_t = torch.as_tensor(y_np, dtype=gram.dtype, device=gram.device)
        alpha = _solve(gram, y_t, self.ridge, self.pinv_rtol)
        train_pred = gram @ alpha
        self.train_x = x_np
        self.alpha = alpha.detach().cpu().numpy().astype(np.float32, copy=False)
        self.diagnostics = {
            "mode": "rff",
            "n_samples": int(len(x_np)),
            "n_input_features": int(x_np.shape[1]),
            "n_random_features": self.p,
            "p_over_n": self.p / len(x_np),
            "ridge_lambda": self.ridge,
            "gamma": self.gamma,
            "seed": self.seed,
            "device": str(gram.device),
            "train_mse": float(torch.mean((train_pred - y_t) ** 2).detach().cpu()),
        }
        if self.compute_diagnostics:
            eigvals = torch.linalg.eigvalsh(gram).clamp_min(0)
            total = eigvals.sum()
            largest = eigvals.max()
            if total > 0 and largest > 0:
                probs = eigvals / total
                probs = probs[probs > 0]
                self.diagnostics["effective_rank_entropy"] = float(
                    torch.exp(-(probs * torch.log(probs)).sum()).detach().cpu()
                )
                self.diagnostics["effective_rank_stable"] = float((total / largest).detach().cpu())
        return self

    def predict(self, x: Any) -> np.ndarray:
        if self.train_x is None or self.alpha is None:
            raise RuntimeError("Estimator is not fitted")
        x_np = _to_numpy_2d(x)
        kernel = self._cross(x_np, self.train_x)
        alpha = torch.as_tensor(self.alpha, dtype=kernel.dtype, device=kernel.device)
        return (kernel @ alpha).detach().cpu().numpy().reshape(-1)


class _RBFKernelEstimator:
    def __init__(self, params: dict[str, Any]) -> None:
        self.gamma = float(params.get("gamma", 0.5))
        self.ridge = float(params.get("ridge_lambda", 1e-6))
        self.device_name = str(params.get("device", "auto"))
        self.acc_dtype_name = str(params.get("accumulator_dtype", "float32"))
        self.pinv_rtol = float(params.get("pinv_rtol", 1e-6))
        self.train_x: np.ndarray | None = None
        self.alpha: np.ndarray | None = None
        self.diagnostics: dict[str, Any] = {}

    def _kernel(self, left: np.ndarray, right: np.ndarray) -> torch.Tensor:
        device = _device(self.device_name)
        dtype = _dtype(self.acc_dtype_name)
        left_t = torch.as_tensor(left, dtype=dtype, device=device)
        right_t = torch.as_tensor(right, dtype=dtype, device=device)
        left_sq = (left_t * left_t).sum(dim=1, keepdim=True)
        right_sq = (right_t * right_t).sum(dim=1).unsqueeze(0)
        distance = (left_sq + right_sq - 2.0 * left_t @ right_t.T).clamp_min_(0)
        return torch.exp(-self.gamma * distance)

    def fit(self, x: Any, y: Any) -> "_RBFKernelEstimator":
        x_np = _to_numpy_2d(x)
        y_np = _to_numpy_1d(y)
        kernel = self._kernel(x_np, x_np)
        y_t = torch.as_tensor(y_np, dtype=kernel.dtype, device=kernel.device)
        alpha = _solve(kernel, y_t, self.ridge, self.pinv_rtol)
        train_pred = kernel @ alpha
        self.train_x = x_np
        self.alpha = alpha.detach().cpu().numpy().astype(np.float32, copy=False)
        self.diagnostics = {
            "mode": "rbf_kernel_limit",
            "n_samples": int(len(x_np)),
            "n_input_features": int(x_np.shape[1]),
            "n_random_features": None,
            "p_over_n": "infinity",
            "ridge_lambda": self.ridge,
            "gamma": self.gamma,
            "device": str(kernel.device),
            "train_mse": float(torch.mean((train_pred - y_t) ** 2).detach().cpu()),
        }
        return self

    def predict(self, x: Any) -> np.ndarray:
        if self.train_x is None or self.alpha is None:
            raise RuntimeError("Estimator is not fitted")
        x_np = _to_numpy_2d(x)
        kernel = self._kernel(x_np, self.train_x)
        alpha = torch.as_tensor(self.alpha, dtype=kernel.dtype, device=kernel.device)
        return (kernel @ alpha).detach().cpu().numpy().reshape(-1)


class DoubleDescentRFFRegressor(BaseRegressionModel):
    """FreqAI model used by the double-descent research experiment.

    `mode=rff` sweeps finite Random Fourier Feature width P.
    `mode=rbf_kernel_limit` evaluates the P -> infinity RBF-kernel limit.
    """

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        x = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]
        if getattr(y, "shape", (0, 0))[1] != 1:
            raise ValueError("DoubleDescentRFFRegressor currently supports exactly one target")

        params = dict(self.model_training_parameters)
        mode = str(params.get("mode", "rff"))
        if mode == "rff":
            model = _RFFEstimator(params).fit(x, y)
        elif mode == "rbf_kernel_limit":
            model = _RBFKernelEstimator(params).fit(x, y)
        else:
            raise ValueError(f"Unknown double-descent mode: {mode}")

        diagnostics_path = Path(dk.data_path) / "double_descent_diagnostics.jsonl"
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        record = dict(model.diagnostics)
        record["pair"] = getattr(dk, "pair", None)
        with diagnostics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        logger.info("Double-descent diagnostics: %s", record)
        return model
