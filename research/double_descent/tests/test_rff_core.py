from __future__ import annotations

import numpy as np

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
