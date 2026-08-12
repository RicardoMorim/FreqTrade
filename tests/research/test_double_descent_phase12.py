from pathlib import Path

import numpy as np
import pandas as pd

from research.double_descent.freqai.Phase12BaselineRegressor import (
    BASELINES,
    Phase12BaselineRegressor,
    _BaselinePredictor,
)
from research.double_descent.phase12 import (
    PHASE12_BASELINES,
    PHASE12_EXPECTED_CASE_COUNT,
    PHASE12_PREDICTION_BASELINES,
    PHASE12_RIDGE_ALPHA,
    Phase12Config,
)


def test_phase12_design_is_frozen() -> None:
    assert PHASE12_BASELINES == BASELINES
    assert PHASE12_EXPECTED_CASE_COUNT == 8
    assert PHASE12_RIDGE_ALPHA == 1.0
    assert "buy_and_hold" not in PHASE12_PREDICTION_BASELINES
    assert set(PHASE12_PREDICTION_BASELINES) == set(PHASE12_BASELINES) - {"buy_and_hold"}


def test_full_run_rejects_post_hoc_baseline_subset(tmp_path: Path) -> None:
    config = Phase12Config(
        data_directory=tmp_path,
        baselines=("zero_return",),
    )

    try:
        config._validate_design()
    except ValueError as exc:
        assert "every frozen baseline" in str(exc)
    else:
        raise AssertionError("full Phase 12 accepted an incomplete baseline set")


def test_linear_predictor_uses_fitted_feature_center() -> None:
    model = _BaselinePredictor(
        baseline="market_ols",
        coefficient=np.array([2.0, -1.0]),
        intercept_scaled=0.5,
        feature_mean=np.array([10.0, 20.0]),
    )
    features = pd.DataFrame([[11.0, 23.0], [9.0, 17.0]])

    np.testing.assert_allclose(model.predict_scaled(features), [-0.5, 1.5])


def test_momentum_definitions_are_fixed() -> None:
    raw = pd.DataFrame(
        {
            "%-return_1h": [0.01, -0.02],
            "%-return_24h": [0.24, -0.12],
            "%-return_volatility_24h": [2.0, -1.0],
        }
    )
    model = _BaselinePredictor(
        baseline="volatility_adjusted_momentum_24h",
        score_standard_deviation=2.0,
        target_standard_deviation=0.01,
    )

    np.testing.assert_allclose(
        Phase12BaselineRegressor._raw_momentum_prediction("momentum_1h", raw, model),
        [0.01, -0.02],
    )
    np.testing.assert_allclose(
        Phase12BaselineRegressor._raw_momentum_prediction("momentum_24h", raw, model),
        [0.01, -0.005],
    )
    np.testing.assert_allclose(
        Phase12BaselineRegressor._raw_momentum_prediction(
            "volatility_adjusted_momentum_24h", raw, model
        ),
        [0.01, -0.005],
    )


def test_smoke_design_allows_one_known_baseline(tmp_path: Path) -> None:
    config = Phase12Config(
        data_directory=tmp_path,
        baselines=("market_ols",),
        smoke_test=True,
    )

    config._validate_design()
