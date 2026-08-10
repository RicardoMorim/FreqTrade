from __future__ import annotations

import math

import numpy as np
import torch

from rff_core import KernelLimitConfig, RBFKernelLimitRegressor, RFFConfig, StreamingRFFRegressor


def test_rff_is_deterministic_across_fit_and_predict() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(30, 4)).astype(np.float32)
    y = rng.normal(size=30).astype(np.float32)
    test = rng.normal(size=(8, 4)).astype(np.float32)
    config = RFFConfig(
        n_random_features=128,
        gamma=0.25,
        ridge_lambda=1e-3,
        seed=7,
        chunk_size=17,
        device="cpu",
    )
    a = StreamingRFFRegressor(config).fit(x, y).predict(test)
    b = StreamingRFFRegressor(config).fit(x, y).predict(test)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_random_feature_prefix_is_nested_across_p() -> None:
    rng = np.random.default_rng(10)
    x = rng.normal(size=(12, 5)).astype(np.float32)

    small = StreamingRFFRegressor(
        RFFConfig(n_random_features=70, seed=17, chunk_size=64, device="cpu")
    )
    large = StreamingRFFRegressor(
        RFFConfig(n_random_features=130, seed=17, chunk_size=64, device="cpu")
    )

    z_small = small._explicit_features(x) / math.sqrt(2.0 / 70)
    z_large = large._explicit_features(x)[:, :70] / math.sqrt(2.0 / 130)
    torch.testing.assert_close(z_small, z_large, rtol=2e-7, atol=1e-7)


def test_hybrid_solver_uses_primal_below_interpolation_and_dual_above() -> None:
    rng = np.random.default_rng(20)
    x = rng.normal(size=(40, 4)).astype(np.float32)
    y = rng.normal(size=40).astype(np.float32)

    primal = StreamingRFFRegressor(
        RFFConfig(n_random_features=20, ridge_lambda=1e-3, seed=5, chunk_size=16, device="cpu")
    ).fit(x, y)
    dual = StreamingRFFRegressor(
        RFFConfig(n_random_features=80, ridge_lambda=1e-3, seed=5, chunk_size=16, device="cpu")
    ).fit(x, y)

    assert primal.diagnostics_["solver_space"] == "primal"
    assert dual.diagnostics_["solver_space"] == "dual"


def test_float64_accumulator_preserves_float64_coefficients() -> None:
    rng = np.random.default_rng(21)
    x = rng.normal(size=(30, 3)).astype(np.float32)
    y = rng.normal(size=30).astype(np.float32)
    model = StreamingRFFRegressor(
        RFFConfig(
            n_random_features=15,
            ridge_lambda=1e-4,
            accumulator_dtype="float64",
            seed=2,
            chunk_size=8,
            device="cpu",
        )
    ).fit(x, y)
    assert model.coef_ is not None
    assert model.coef_.dtype == np.float64


def test_rff_approaches_kernel_limit() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(25, 3)).astype(np.float32)
    y = (0.3 * x[:, 0] - 0.1 * x[:, 1] + rng.normal(scale=0.05, size=25)).astype(np.float32)
    test = rng.normal(size=(6, 3)).astype(np.float32)

    finite = StreamingRFFRegressor(
        RFFConfig(
            n_random_features=12000,
            gamma=0.2,
            ridge_lambda=1e-2,
            seed=11,
            chunk_size=1000,
            device="cpu",
        )
    ).fit(x, y)
    limit = RBFKernelLimitRegressor(
        KernelLimitConfig(gamma=0.2, ridge_lambda=1e-2, device="cpu")
    ).fit(x, y)

    finite_pred = finite.predict(test)
    limit_pred = limit.predict(test)
    assert np.corrcoef(finite_pred, limit_pred)[0, 1] > 0.95


def test_diagnostics_report_p_over_n() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(size=(20, 2)).astype(np.float32)
    y = rng.normal(size=20).astype(np.float32)
    model = StreamingRFFRegressor(
        RFFConfig(n_random_features=100, device="cpu", chunk_size=25),
        compute_diagnostics=True,
    ).fit(x, y)
    assert model.diagnostics_["p_over_n"] == 5.0
    assert model.diagnostics_["effective_rank_entropy"] > 0
