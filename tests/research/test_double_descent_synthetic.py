from pathlib import Path

import numpy as np

from research.double_descent.synthetic import (
    Phase1Config,
    fit_minimum_norm,
    random_fourier_features,
    run_phase1,
)


def test_random_fourier_features_are_nested() -> None:
    inputs = np.random.default_rng(7).normal(size=(12, 4))

    small = random_fourier_features(inputs, feature_count=16, gamma=0.2, seed=42)
    large = random_fourier_features(inputs, feature_count=64, gamma=0.2, seed=42)

    np.testing.assert_array_equal(small, large[:, :16])


def test_minimum_norm_interpolates_overparameterized_training_data() -> None:
    rng = np.random.default_rng(11)
    train_features = rng.normal(size=(24, 48))
    test_features = rng.normal(size=(10, 48))
    train_target = rng.normal(size=24)

    result = fit_minimum_norm(train_features, train_target, test_features, rcond=1e-12)

    assert np.mean((result["train_prediction"] - train_target) ** 2) < 1e-20
    assert result["rank"] == 23  # Train centering reserves one degree for the intercept.


def test_phase1_writes_reproducible_artifacts(tmp_path: Path) -> None:
    config = Phase1Config(
        n_train=32,
        n_test=64,
        linear_teacher_dimension=256,
        ratios=(0.5, 0.9, 1.0, 1.1, 2.0, 4.0),
        repetitions=2,
        base_seed=1234,
    )

    summary = run_phase1(config, tmp_path)

    assert summary["phase"] == 1
    assert set(summary["gate"]["datasets"]) == {"linear_gaussian", "nonlinear_rff"}
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "metrics_detailed.csv").is_file()
    assert (tmp_path / "metrics_aggregate.csv").is_file()
    assert len(summary["kernel_limit"]) == config.repetitions
