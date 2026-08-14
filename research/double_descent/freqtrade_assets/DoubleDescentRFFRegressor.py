from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pandas import DataFrame

from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen


logger = logging.getLogger(__name__)


def _to_numpy_2d(value: Any) -> np.ndarray:
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D features, got shape={array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Features contain NaN or infinite values")
    return np.ascontiguousarray(array)


def _to_numpy_1d(value: Any) -> np.ndarray:
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if not np.isfinite(array).all():
        raise ValueError("Target contains NaN or infinite values")
    return np.ascontiguousarray(array)


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


def _numpy_dtype(dtype: torch.dtype) -> np.dtype:
    return np.dtype(np.float64 if dtype == torch.float64 else np.float32)


def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return (matrix + matrix.T).mul_(0.5)


def _solve_psd(matrix: torch.Tensor, y: torch.Tensor, ridge: float, rtol: float) -> torch.Tensor:
    matrix = _symmetrize(matrix)
    if ridge > 0:
        regularized = matrix.clone()
        regularized.diagonal().add_(ridge)
        chol, info = torch.linalg.cholesky_ex(regularized)
        if int(info.max().detach().cpu()) == 0:
            return torch.cholesky_solve(y[:, None], chol).squeeze(1)
        return torch.linalg.solve(regularized, y)
    return torch.linalg.pinv(matrix, hermitian=True, rtol=rtol) @ y


def _window_metrics(actual: np.ndarray, predicted: np.ndarray, accepted: np.ndarray) -> dict[str, Any]:
    mask = accepted & np.isfinite(actual) & np.isfinite(predicted)
    y = actual[mask].astype(np.float64, copy=False)
    p = predicted[mask].astype(np.float64, copy=False)
    n = int(len(y))
    if n == 0:
        return {"n_oos": 0}

    err = p - y
    sse = float(np.dot(err, err))
    sae = float(np.abs(err).sum())
    sum_y = float(y.sum())
    sum_y2 = float(np.dot(y, y))
    sum_p = float(p.sum())
    sum_p2 = float(np.dot(p, p))
    sum_py = float(np.dot(p, y))
    direction_correct = int(np.sum(np.signbit(p) == np.signbit(y)))

    sst = sum_y2 - (sum_y * sum_y / n)
    r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")
    cov = sum_py - (sum_p * sum_y / n)
    var_p = sum_p2 - (sum_p * sum_p / n)
    var_y = sst
    ic = float(cov / math.sqrt(var_p * var_y)) if var_p > 0 and var_y > 0 else float("nan")

    return {
        "n_oos": n,
        "sse": sse,
        "sae": sae,
        "sum_y": sum_y,
        "sum_y2": sum_y2,
        "sum_pred": sum_p,
        "sum_pred2": sum_p2,
        "sum_pred_y": sum_py,
        "direction_correct": direction_correct,
        "mse": sse / n,
        "mae": sae / n,
        "r2": r2,
        "ic": ic,
        "directional_accuracy": direction_correct / n,
    }


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
        self.matmul_precision = str(params.get("matmul_precision", "highest"))
        if self.p <= 0 or self.chunk_size <= 0 or self.gamma <= 0 or self.ridge < 0:
            raise ValueError("Invalid RFF model parameters")
        if self.matmul_precision not in {"highest", "high", "medium"}:
            raise ValueError("matmul_precision must be highest, high, or medium")
        self.train_x: np.ndarray | None = None
        self.alpha: np.ndarray | None = None
        self.coef: np.ndarray | None = None
        self.solver_space: str | None = None
        self.diagnostics: dict[str, Any] = {}

    def _weights(self, device: torch.device, input_dim: int):
        gen = torch.Generator(device=device)
        gen.manual_seed(self.seed)
        dtype = _dtype(self.feature_dtype_name)
        produced = 0
        omega_scale = math.sqrt(2.0 * self.gamma)
        while produced < self.p:
            width = min(self.chunk_size, self.p - produced)
            omega_full = torch.randn(
                (self.chunk_size, input_dim), generator=gen, dtype=dtype, device=device
            ).mul_(omega_scale)
            bias_full = torch.rand(
                (self.chunk_size,), generator=gen, dtype=dtype, device=device
            ).mul_(2.0 * math.pi)
            yield omega_full[:width], bias_full[:width]
            produced += width

    def _feature_chunks(self, x_t: torch.Tensor):
        scale = math.sqrt(2.0 / self.p)
        for omega, bias in self._weights(x_t.device, x_t.shape[1]):
            yield torch.cos(x_t @ omega.T + bias).mul_(scale)

    def _explicit_features(self, x: np.ndarray) -> torch.Tensor:
        device = _device(self.device_name)
        feature_dtype = _dtype(self.feature_dtype_name)
        x_t = torch.as_tensor(x, dtype=feature_dtype, device=device)
        return torch.cat(list(self._feature_chunks(x_t)), dim=1)

    def _gram(self, x: np.ndarray) -> torch.Tensor:
        device = _device(self.device_name)
        feature_dtype = _dtype(self.feature_dtype_name)
        acc_dtype = _dtype(self.acc_dtype_name)
        x_t = torch.as_tensor(x, dtype=feature_dtype, device=device)
        gram = torch.zeros((len(x), len(x)), dtype=acc_dtype, device=device)
        for z in self._feature_chunks(x_t):
            z_acc = z if z.dtype == acc_dtype else z.to(acc_dtype)
            gram.addmm_(z_acc, z_acc.T)
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
            z_x_acc = z_x if z_x.dtype == acc_dtype else z_x.to(acc_dtype)
            z_train_acc = z_train if z_train.dtype == acc_dtype else z_train.to(acc_dtype)
            kernel.addmm_(z_x_acc, z_train_acc.T)
        return kernel

    def fit(self, x: Any, y: Any) -> "_RFFEstimator":
        x_np = _to_numpy_2d(x)
        y_np = _to_numpy_1d(y)
        if len(x_np) != len(y_np):
            raise ValueError("X and y contain different numbers of rows")

        torch.set_float32_matmul_precision(self.matmul_precision)
        device = _device(self.device_name)
        acc_dtype = _dtype(self.acc_dtype_name)
        y_t = torch.as_tensor(y_np, dtype=acc_dtype, device=device)
        n = len(x_np)

        if self.p <= n:
            z = self._explicit_features(x_np)
            z_acc = z if z.dtype == acc_dtype else z.to(acc_dtype)
            if self.ridge > 0:
                normal = z_acc.T @ z_acc
                coef = _solve_psd(normal, z_acc.T @ y_t, self.ridge, self.pinv_rtol)
            else:
                coef = torch.linalg.pinv(z_acc, rtol=self.pinv_rtol) @ y_t
            train_pred = z_acc @ coef
            self.coef = coef.detach().cpu().numpy().astype(_numpy_dtype(acc_dtype), copy=False)
            self.solver_space = "primal"
            diagnostic_matrix = z_acc.T @ z_acc if self.compute_diagnostics else None
        else:
            gram = self._gram(x_np)
            alpha = _solve_psd(gram, y_t, self.ridge, self.pinv_rtol)
            train_pred = gram @ alpha
            self.train_x = x_np
            self.alpha = alpha.detach().cpu().numpy().astype(_numpy_dtype(acc_dtype), copy=False)
            self.solver_space = "dual"
            diagnostic_matrix = gram if self.compute_diagnostics else None

        self.diagnostics = {
            "mode": "rff",
            "solver_space": self.solver_space,
            "n_samples": int(n),
            "n_input_features": int(x_np.shape[1]),
            "n_random_features": self.p,
            "p_over_n": self.p / n,
            "ridge_lambda": self.ridge,
            "gamma": self.gamma,
            "seed": self.seed,
            "device": str(device),
            "feature_dtype": self.feature_dtype_name,
            "accumulator_dtype": self.acc_dtype_name,
            "matmul_precision": self.matmul_precision,
            "train_mse": float(torch.mean((train_pred - y_t) ** 2).detach().cpu()),
        }
        if diagnostic_matrix is not None:
            eigvals = torch.linalg.eigvalsh(_symmetrize(diagnostic_matrix)).clamp_min(0)
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
        if self.solver_space is None:
            raise RuntimeError("Estimator is not fitted")
        x_np = _to_numpy_2d(x)
        acc_dtype = _dtype(self.acc_dtype_name)

        if self.solver_space == "primal":
            if self.coef is None:
                raise RuntimeError("Primal coefficient state missing")
            z = self._explicit_features(x_np)
            z_acc = z if z.dtype == acc_dtype else z.to(acc_dtype)
            coef = torch.as_tensor(self.coef, dtype=acc_dtype, device=z.device)
            pred = z_acc @ coef
        else:
            if self.train_x is None or self.alpha is None:
                raise RuntimeError("Dual estimator state missing")
            kernel = self._cross(x_np, self.train_x)
            alpha = torch.as_tensor(self.alpha, dtype=kernel.dtype, device=kernel.device)
            pred = kernel @ alpha

        return pred.detach().cpu().numpy().reshape(-1)


class _RBFKernelEstimator:
    def __init__(self, params: dict[str, Any]) -> None:
        self.gamma = float(params.get("gamma", 0.5))
        self.ridge = float(params.get("ridge_lambda", 1e-6))
        self.device_name = str(params.get("device", "auto"))
        self.acc_dtype_name = str(params.get("accumulator_dtype", "float32"))
        self.pinv_rtol = float(params.get("pinv_rtol", 1e-6))
        self.matmul_precision = str(params.get("matmul_precision", "highest"))
        self.train_x: np.ndarray | None = None
        self.alpha: np.ndarray | None = None
        self.diagnostics: dict[str, Any] = {}

    def _kernel(self, left: np.ndarray, right: np.ndarray) -> torch.Tensor:
        torch.set_float32_matmul_precision(self.matmul_precision)
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
        alpha = _solve_psd(kernel, y_t, self.ridge, self.pinv_rtol)
        train_pred = kernel @ alpha
        self.train_x = x_np
        self.alpha = alpha.detach().cpu().numpy().astype(_numpy_dtype(kernel.dtype), copy=False)
        self.diagnostics = {
            "mode": "rbf_kernel_limit",
            "solver_space": "dual",
            "n_samples": int(len(x_np)),
            "n_input_features": int(x_np.shape[1]),
            "n_random_features": None,
            "p_over_n": "infinity",
            "ridge_lambda": self.ridge,
            "gamma": self.gamma,
            "device": str(kernel.device),
            "accumulator_dtype": self.acc_dtype_name,
            "matmul_precision": self.matmul_precision,
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

    def predict(
        self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> tuple[DataFrame, np.ndarray]:
        pred_df, do_predict = super().predict(unfiltered_df, dk, **kwargs)
        if len(dk.label_list) != 1:
            return pred_df, do_predict

        label = dk.label_list[0]
        if label not in unfiltered_df.columns or label not in pred_df.columns:
            return pred_df, do_predict

        actual = np.asarray(unfiltered_df[label], dtype=np.float64).reshape(-1)
        predicted = np.asarray(pred_df[label], dtype=np.float64).reshape(-1)
        accepted = np.asarray(do_predict).reshape(-1) == 1
        if not (len(actual) == len(predicted) == len(accepted)):
            logger.warning(
                "Skipping OOS metrics due to length mismatch actual=%s pred=%s mask=%s",
                len(actual),
                len(predicted),
                len(accepted),
            )
            return pred_df, do_predict

        record = _window_metrics(actual, predicted, accepted)
        record.update(
            {
                "pair": getattr(dk, "pair", None),
                "label": label,
                "start": str(unfiltered_df["date"].iloc[0]) if len(unfiltered_df) else None,
                "stop": str(unfiltered_df["date"].iloc[-1]) if len(unfiltered_df) else None,
            }
        )
        model_diag = getattr(self.model, "diagnostics", {})
        for key in (
            "mode",
            "solver_space",
            "n_samples",
            "n_input_features",
            "n_random_features",
            "p_over_n",
            "ridge_lambda",
            "gamma",
            "seed",
            "device",
            "feature_dtype",
            "accumulator_dtype",
            "matmul_precision",
        ):
            if key in model_diag:
                record[key] = model_diag[key]

        metrics_path = Path(dk.data_path) / "double_descent_oos.jsonl"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        logger.info("Double-descent OOS metrics: %s", record)
        return pred_df, do_predict
