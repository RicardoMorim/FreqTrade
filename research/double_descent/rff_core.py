from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover - exercised only in minimal installs
    raise ImportError(
        "double_descent requires PyTorch. Install a CUDA-enabled PyTorch build for GPU runs."
    ) from exc


DeviceName = Literal["auto", "cpu", "cuda"]
ModeName = Literal["rff", "rbf_kernel_limit"]
MatmulPrecision = Literal["highest", "high", "medium"]


@dataclass(slots=True)
class RFFConfig:
    """Configuration for the finite-width Random Fourier Feature regressor."""

    n_random_features: int = 1024
    gamma: float = 0.5
    ridge_lambda: float = 1e-6
    seed: int = 42
    chunk_size: int = 4096
    device: DeviceName = "auto"
    feature_dtype: str = "float32"
    accumulator_dtype: str = "float32"
    pinv_rtol: float = 1e-6
    matmul_precision: MatmulPrecision = "highest"

    def validate(self) -> None:
        if self.n_random_features <= 0:
            raise ValueError("n_random_features must be > 0")
        if self.gamma <= 0:
            raise ValueError("gamma must be > 0")
        if self.ridge_lambda < 0:
            raise ValueError("ridge_lambda must be >= 0")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.pinv_rtol <= 0:
            raise ValueError("pinv_rtol must be > 0")
        if self.matmul_precision not in {"highest", "high", "medium"}:
            raise ValueError("matmul_precision must be highest, high, or medium")


@dataclass(slots=True)
class KernelLimitConfig:
    """Configuration for the infinite-width RBF-kernel limit of the RFF model."""

    gamma: float = 0.5
    ridge_lambda: float = 1e-6
    device: DeviceName = "auto"
    accumulator_dtype: str = "float32"
    pinv_rtol: float = 1e-6
    matmul_precision: MatmulPrecision = "highest"

    def validate(self) -> None:
        if self.gamma <= 0:
            raise ValueError("gamma must be > 0")
        if self.ridge_lambda < 0:
            raise ValueError("ridge_lambda must be >= 0")
        if self.pinv_rtol <= 0:
            raise ValueError("pinv_rtol must be > 0")
        if self.matmul_precision not in {"highest", "high", "medium"}:
            raise ValueError("matmul_precision must be highest, high, or medium")


def _resolve_device(name: DeviceName) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return torch.device(name)


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def _numpy_dtype(dtype: torch.dtype) -> np.dtype:
    if dtype == torch.float64:
        return np.dtype(np.float64)
    return np.dtype(np.float32)


def _to_2d_float_array(x: Any) -> np.ndarray:
    if hasattr(x, "to_numpy"):
        x = x.to_numpy()
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape={arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("Feature matrix contains NaN or infinite values")
    return np.ascontiguousarray(arr)


def _to_1d_float_array(y: Any) -> np.ndarray:
    if hasattr(y, "to_numpy"):
        y = y.to_numpy()
    arr = np.asarray(y, dtype=np.float32).reshape(-1)
    if not np.isfinite(arr).all():
        raise ValueError("Target contains NaN or infinite values")
    return np.ascontiguousarray(arr)


def _configure_matmul(precision: MatmulPrecision) -> None:
    torch.set_float32_matmul_precision(precision)


def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return (matrix + matrix.T).mul_(0.5)


def _solve_psd_system(
    matrix: torch.Tensor,
    y: torch.Tensor,
    ridge_lambda: float,
    pinv_rtol: float,
) -> torch.Tensor:
    """Solve a symmetric PSD system using Cholesky for ridge, pinv for ridgeless."""

    matrix = _symmetrize(matrix)
    if ridge_lambda > 0:
        regularized = matrix.clone()
        regularized.diagonal().add_(ridge_lambda)
        chol, info = torch.linalg.cholesky_ex(regularized)
        if int(info.max().detach().cpu()) == 0:
            return torch.cholesky_solve(y[:, None], chol).squeeze(1)
        return torch.linalg.solve(regularized, y)

    return torch.linalg.pinv(matrix, hermitian=True, rtol=pinv_rtol) @ y


def _kernel_diagnostics(kernel: torch.Tensor) -> dict[str, float]:
    """Return effective-rank diagnostics from a symmetric positive semidefinite matrix."""

    eigvals = torch.linalg.eigvalsh(_symmetrize(kernel)).clamp_min(0)
    total = eigvals.sum()
    if total <= 0:
        return {
            "effective_rank_entropy": 0.0,
            "effective_rank_stable": 0.0,
            "condition_number_nonzero": math.inf,
        }

    probs = eigvals / total
    positive_probs = probs[probs > 0]
    entropy = -(positive_probs * torch.log(positive_probs)).sum()
    effective_rank_entropy = torch.exp(entropy)
    largest = eigvals.max()
    effective_rank_stable = total / largest if largest > 0 else torch.tensor(0.0)

    threshold = largest * 1e-12
    nonzero = eigvals[eigvals > threshold]
    condition = largest / nonzero.min() if nonzero.numel() else torch.tensor(float("inf"))

    return {
        "effective_rank_entropy": float(effective_rank_entropy.detach().cpu()),
        "effective_rank_stable": float(effective_rank_stable.detach().cpu()),
        "condition_number_nonzero": float(condition.detach().cpu()),
    }


class StreamingRFFRegressor:
    """Minimum-norm / ridge regression on Random Fourier Features.

    For P <= N the implementation uses an explicit N x P feature matrix and a
    primal solve. This avoids constructing/solving an unnecessarily large N x N
    system for small models.

    For P > N, feature chunks are streamed into the N x N dual Gram matrix, so
    memory scales with N^2 rather than N x P.

    Random features are generated in fixed-size RNG blocks. The final block is
    generated at full chunk_size and sliced. This is intentional: with a fixed
    seed/chunk_size/gamma, the feature set for smaller P is an exact prefix of the
    feature set for larger P, even when P is not a multiple of chunk_size.
    """

    def __init__(self, config: RFFConfig, *, compute_diagnostics: bool = False) -> None:
        config.validate()
        self.config = config
        self.compute_diagnostics = compute_diagnostics
        self.train_x_: np.ndarray | None = None
        self.alpha_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.solver_space_: str | None = None
        self.diagnostics_: dict[str, Any] = {}

    def _feature_chunks(self, x: torch.Tensor):
        cfg = self.config
        p = cfg.n_random_features
        d = x.shape[1]
        feature_dtype = _resolve_dtype(cfg.feature_dtype)
        generator = torch.Generator(device=x.device)
        generator.manual_seed(cfg.seed)
        scale = math.sqrt(2.0 / p)
        omega_scale = math.sqrt(2.0 * cfg.gamma)

        produced = 0
        while produced < p:
            width = min(cfg.chunk_size, p - produced)
            omega_full = torch.randn(
                (cfg.chunk_size, d),
                generator=generator,
                device=x.device,
                dtype=feature_dtype,
            )
            omega_full.mul_(omega_scale)
            bias_full = torch.rand(
                (cfg.chunk_size,),
                generator=generator,
                device=x.device,
                dtype=feature_dtype,
            )
            bias_full.mul_(2.0 * math.pi)

            omega = omega_full[:width]
            bias = bias_full[:width]
            z = torch.cos(x @ omega.T + bias)
            z.mul_(scale)
            yield z
            produced += width

    def _explicit_features(self, x: np.ndarray) -> torch.Tensor:
        device = _resolve_device(self.config.device)
        feature_dtype = _resolve_dtype(self.config.feature_dtype)
        x_t = torch.as_tensor(x, dtype=feature_dtype, device=device)
        chunks = list(self._feature_chunks(x_t))
        return torch.cat(chunks, dim=1)

    def _dual_gram(self, x: np.ndarray) -> torch.Tensor:
        device = _resolve_device(self.config.device)
        acc_dtype = _resolve_dtype(self.config.accumulator_dtype)
        feature_dtype = _resolve_dtype(self.config.feature_dtype)
        x_t = torch.as_tensor(x, dtype=feature_dtype, device=device)
        gram = torch.zeros((len(x), len(x)), dtype=acc_dtype, device=device)
        for z in self._feature_chunks(x_t):
            z_acc = z if z.dtype == acc_dtype else z.to(acc_dtype)
            gram.addmm_(z_acc, z_acc.T)
        return gram

    def _cross_kernel(self, left: np.ndarray, right: np.ndarray) -> torch.Tensor:
        device = _resolve_device(self.config.device)
        acc_dtype = _resolve_dtype(self.config.accumulator_dtype)
        feature_dtype = _resolve_dtype(self.config.feature_dtype)
        left_t = torch.as_tensor(left, dtype=feature_dtype, device=device)
        right_t = torch.as_tensor(right, dtype=feature_dtype, device=device)
        out = torch.zeros((left.shape[0], right.shape[0]), dtype=acc_dtype, device=device)

        cfg = self.config
        p = cfg.n_random_features
        d = left.shape[1]
        generator = torch.Generator(device=device)
        generator.manual_seed(cfg.seed)
        scale = math.sqrt(2.0 / p)
        omega_scale = math.sqrt(2.0 * cfg.gamma)

        produced = 0
        while produced < p:
            width = min(cfg.chunk_size, p - produced)
            omega_full = torch.randn(
                (cfg.chunk_size, d),
                generator=generator,
                device=device,
                dtype=feature_dtype,
            ).mul_(omega_scale)
            bias_full = torch.rand(
                (cfg.chunk_size,),
                generator=generator,
                device=device,
                dtype=feature_dtype,
            ).mul_(2.0 * math.pi)
            omega = omega_full[:width]
            bias = bias_full[:width]

            z_left = torch.cos(left_t @ omega.T + bias).mul_(scale)
            z_right = torch.cos(right_t @ omega.T + bias).mul_(scale)
            z_left_acc = z_left if z_left.dtype == acc_dtype else z_left.to(acc_dtype)
            z_right_acc = z_right if z_right.dtype == acc_dtype else z_right.to(acc_dtype)
            out.addmm_(z_left_acc, z_right_acc.T)
            produced += width

        return out

    def fit(self, x: Any, y: Any) -> "StreamingRFFRegressor":
        x_np = _to_2d_float_array(x)
        y_np = _to_1d_float_array(y)
        if x_np.shape[0] != y_np.shape[0]:
            raise ValueError("X and y contain different numbers of rows")

        _configure_matmul(self.config.matmul_precision)
        device = _resolve_device(self.config.device)
        acc_dtype = _resolve_dtype(self.config.accumulator_dtype)
        y_t = torch.as_tensor(y_np, dtype=acc_dtype, device=device)
        n = x_np.shape[0]
        p = self.config.n_random_features

        self.train_x_ = None
        self.alpha_ = None
        self.coef_ = None

        if p <= n:
            z = self._explicit_features(x_np)
            z_acc = z if z.dtype == acc_dtype else z.to(acc_dtype)
            if self.config.ridge_lambda > 0:
                normal = z_acc.T @ z_acc
                rhs = z_acc.T @ y_t
                coef = _solve_psd_system(
                    normal,
                    rhs,
                    ridge_lambda=self.config.ridge_lambda,
                    pinv_rtol=self.config.pinv_rtol,
                )
            else:
                coef = torch.linalg.pinv(z_acc, rtol=self.config.pinv_rtol) @ y_t

            fitted = z_acc @ coef
            self.coef_ = coef.detach().cpu().numpy().astype(_numpy_dtype(acc_dtype), copy=False)
            self.solver_space_ = "primal"
            diagnostic_matrix = z_acc.T @ z_acc if self.compute_diagnostics else None
        else:
            gram = self._dual_gram(x_np)
            alpha = _solve_psd_system(
                gram,
                y_t,
                ridge_lambda=self.config.ridge_lambda,
                pinv_rtol=self.config.pinv_rtol,
            )
            fitted = gram @ alpha
            self.train_x_ = x_np
            self.alpha_ = alpha.detach().cpu().numpy().astype(_numpy_dtype(acc_dtype), copy=False)
            self.solver_space_ = "dual"
            diagnostic_matrix = gram if self.compute_diagnostics else None

        train_mse = torch.mean((fitted - y_t) ** 2)
        diagnostics: dict[str, Any] = {
            "mode": "rff",
            "solver_space": self.solver_space_,
            "n_samples": int(n),
            "n_input_features": int(x_np.shape[1]),
            "n_random_features": int(p),
            "p_over_n": float(p / n),
            "train_mse": float(train_mse.detach().cpu()),
            "device": str(device),
            "ridge_lambda": float(self.config.ridge_lambda),
            "gamma": float(self.config.gamma),
            "seed": int(self.config.seed),
            "feature_dtype": self.config.feature_dtype,
            "accumulator_dtype": self.config.accumulator_dtype,
            "matmul_precision": self.config.matmul_precision,
        }
        if diagnostic_matrix is not None:
            diagnostics.update(_kernel_diagnostics(diagnostic_matrix))
        self.diagnostics_ = diagnostics
        return self

    def predict(self, x: Any) -> np.ndarray:
        if self.solver_space_ is None:
            raise RuntimeError("Model must be fitted before predict()")
        x_np = _to_2d_float_array(x)
        acc_dtype = _resolve_dtype(self.config.accumulator_dtype)

        if self.solver_space_ == "primal":
            if self.coef_ is None:
                raise RuntimeError("Primal coefficient state is missing")
            z = self._explicit_features(x_np)
            z_acc = z if z.dtype == acc_dtype else z.to(acc_dtype)
            coef = torch.as_tensor(self.coef_, dtype=acc_dtype, device=z.device)
            pred = z_acc @ coef
        else:
            if self.train_x_ is None or self.alpha_ is None:
                raise RuntimeError("Dual estimator state is missing")
            kernel = self._cross_kernel(x_np, self.train_x_)
            alpha = torch.as_tensor(self.alpha_, dtype=kernel.dtype, device=kernel.device)
            pred = kernel @ alpha

        return pred.detach().cpu().numpy().reshape(-1)

    def save_diagnostics(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.diagnostics_, indent=2), encoding="utf-8")


class RBFKernelLimitRegressor:
    """Exact RBF-kernel regression: the P -> infinity limit of the RFF model."""

    def __init__(self, config: KernelLimitConfig, *, compute_diagnostics: bool = False) -> None:
        config.validate()
        self.config = config
        self.compute_diagnostics = compute_diagnostics
        self.train_x_: np.ndarray | None = None
        self.alpha_: np.ndarray | None = None
        self.diagnostics_: dict[str, Any] = {}

    def _rbf_kernel(self, left: np.ndarray, right: np.ndarray) -> torch.Tensor:
        _configure_matmul(self.config.matmul_precision)
        device = _resolve_device(self.config.device)
        dtype = _resolve_dtype(self.config.accumulator_dtype)
        left_t = torch.as_tensor(left, dtype=dtype, device=device)
        right_t = torch.as_tensor(right, dtype=dtype, device=device)
        left_sq = (left_t * left_t).sum(dim=1, keepdim=True)
        right_sq = (right_t * right_t).sum(dim=1).unsqueeze(0)
        distances = (left_sq + right_sq - 2.0 * left_t @ right_t.T).clamp_min_(0)
        return torch.exp(-self.config.gamma * distances)

    def fit(self, x: Any, y: Any) -> "RBFKernelLimitRegressor":
        x_np = _to_2d_float_array(x)
        y_np = _to_1d_float_array(y)
        if x_np.shape[0] != y_np.shape[0]:
            raise ValueError("X and y contain different numbers of rows")

        kernel = self._rbf_kernel(x_np, x_np)
        y_t = torch.as_tensor(y_np, dtype=kernel.dtype, device=kernel.device)
        alpha = _solve_psd_system(
            kernel,
            y_t,
            ridge_lambda=self.config.ridge_lambda,
            pinv_rtol=self.config.pinv_rtol,
        )

        self.train_x_ = x_np
        self.alpha_ = alpha.detach().cpu().numpy().astype(_numpy_dtype(kernel.dtype), copy=False)
        fitted = kernel @ alpha
        train_mse = torch.mean((fitted - y_t) ** 2)
        diagnostics: dict[str, Any] = {
            "mode": "rbf_kernel_limit",
            "solver_space": "dual",
            "n_samples": int(x_np.shape[0]),
            "n_input_features": int(x_np.shape[1]),
            "n_random_features": None,
            "p_over_n": math.inf,
            "train_mse": float(train_mse.detach().cpu()),
            "device": str(kernel.device),
            "ridge_lambda": float(self.config.ridge_lambda),
            "gamma": float(self.config.gamma),
            "accumulator_dtype": self.config.accumulator_dtype,
            "matmul_precision": self.config.matmul_precision,
        }
        if self.compute_diagnostics:
            diagnostics.update(_kernel_diagnostics(kernel))
        self.diagnostics_ = diagnostics
        return self

    def predict(self, x: Any) -> np.ndarray:
        if self.train_x_ is None or self.alpha_ is None:
            raise RuntimeError("Model must be fitted before predict()")
        x_np = _to_2d_float_array(x)
        kernel = self._rbf_kernel(x_np, self.train_x_)
        alpha = torch.as_tensor(self.alpha_, dtype=kernel.dtype, device=kernel.device)
        pred = kernel @ alpha
        return pred.detach().cpu().numpy().reshape(-1)


def config_as_dict(config: RFFConfig | KernelLimitConfig) -> dict[str, Any]:
    return asdict(config)
