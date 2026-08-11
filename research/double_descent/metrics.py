"""Prediction metrics shared by real-data double-descent phases."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def prediction_metrics(actual: ArrayLike, prediction: ArrayLike) -> dict[str, Any]:
    """Calculate return-prediction metrics after a joint finite-value filter."""
    actual_array = np.asarray(actual, dtype=np.float64).reshape(-1)
    prediction_array = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if actual_array.shape != prediction_array.shape:
        raise ValueError("actual and prediction arrays must have identical shapes")
    finite = np.isfinite(actual_array) & np.isfinite(prediction_array)
    actual_array = actual_array[finite]
    prediction_array = prediction_array[finite]
    if actual_array.size == 0:
        raise ValueError("no finite prediction pairs remain")
    residual = prediction_array - actual_array
    squared_error = residual**2
    centered_denominator = float(np.sum((actual_array - np.mean(actual_array)) ** 2))
    zero_denominator = float(np.sum(actual_array**2))
    actual_ranks = pd.Series(actual_array).rank(method="average").to_numpy()
    prediction_ranks = pd.Series(prediction_array).rank(method="average").to_numpy()
    return {
        "observation_count": int(actual_array.size),
        "mse": float(np.mean(squared_error)),
        "rmse": float(np.sqrt(np.mean(squared_error))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": (
            1.0 - float(np.sum(squared_error)) / centered_denominator
            if centered_denominator > 0
            else math.nan
        ),
        "r2_vs_zero": (
            1.0 - float(np.sum(squared_error)) / zero_denominator
            if zero_denominator > 0
            else math.nan
        ),
        "information_coefficient": _correlation(prediction_array, actual_array),
        "spearman_information_coefficient": _correlation(prediction_ranks, actual_ranks),
        "directional_accuracy": float(np.mean(np.sign(prediction_array) == np.sign(actual_array))),
        "actual_mean": float(np.mean(actual_array)),
        "actual_standard_deviation": float(np.std(actual_array)),
        "prediction_mean": float(np.mean(prediction_array)),
        "prediction_standard_deviation": float(np.std(prediction_array)),
        "prediction_minimum": float(np.min(prediction_array)),
        "prediction_maximum": float(np.max(prediction_array)),
    }
