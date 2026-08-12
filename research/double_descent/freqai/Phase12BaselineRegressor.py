"""FreqAI adapter for frozen simple prediction and economic baselines."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas import DataFrame

from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from research.double_descent.freqai.Phase4CudaRFFRegressor import _timestamp
from research.double_descent.metrics import prediction_metrics


BASELINES = (
    "zero_return",
    "historical_mean",
    "market_ols",
    "market_ridge",
    "momentum_1h",
    "momentum_24h",
    "volatility_adjusted_momentum_24h",
    "buy_and_hold",
)
RAW_FEATURES = (
    "%-return_1h",
    "%-return_24h",
    "%-return_volatility_24h",
)


def _feature_column(frame: DataFrame, name: str) -> str:
    matches = [column for column in frame.columns if str(column) == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one raw feature named {name}, found {matches}")
    return str(matches[0])


class _BaselinePredictor:
    def __init__(
        self,
        baseline: str,
        coefficient: np.ndarray | None = None,
        intercept_scaled: float = 0.0,
        constant_raw: float = 0.0,
        feature_mean: np.ndarray | None = None,
        target_standard_deviation: float = 0.0,
        score_standard_deviation: float = 1.0,
        momentum_24h_scale: float = 24.0,
    ) -> None:
        self.baseline = baseline
        self.coefficient = coefficient
        self.intercept_scaled = intercept_scaled
        self.constant_raw = constant_raw
        self.feature_mean = feature_mean
        self.target_standard_deviation = target_standard_deviation
        self.score_standard_deviation = score_standard_deviation
        self.momentum_24h_scale = momentum_24h_scale

    def predict_scaled(self, features: DataFrame) -> np.ndarray:
        if self.coefficient is None:
            raise RuntimeError(f"{self.baseline} does not use transformed feature prediction")
        if self.feature_mean is None:
            raise RuntimeError(f"{self.baseline} is missing its fitted feature mean")
        centered = features.to_numpy(dtype=np.float64) - self.feature_mean
        return self.intercept_scaled + centered @ self.coefficient


class Phase12BaselineRegressor(BaseRegressionModel):
    """Run one predeclared simple baseline in every genuine rolling window."""

    def train(
        self,
        unfiltered_df: DataFrame,
        pair: str,
        dk: FreqaiDataKitchen,
        **kwargs: Any,
    ) -> Any:
        self._phase12_raw_rows = len(unfiltered_df)
        self._phase12_raw_start = _timestamp(unfiltered_df["date"].iloc[0])
        self._phase12_raw_end = _timestamp(unfiltered_df["date"].iloc[-1])
        required = ["date", *RAW_FEATURES]
        missing = [column for column in required if column not in unfiltered_df]
        if missing:
            raise ValueError(f"baseline raw features are missing: {missing}")
        raw = unfiltered_df[required].copy()
        raw["date"] = pd.to_datetime(raw["date"], utc=True)
        if raw["date"].duplicated().any():
            raise ValueError("baseline training dates must be unique")
        self._phase12_raw_training = raw.set_index("date")
        return super().train(unfiltered_df, pair, dk, **kwargs)

    @staticmethod
    def _inverse_labels(
        values: np.ndarray | DataFrame,
        dk: FreqaiDataKitchen,
    ) -> np.ndarray:
        frame = DataFrame(np.asarray(values).reshape(-1, 1), columns=dk.label_list)
        inverse, _, _ = dk.label_pipeline.inverse_transform(frame)
        return inverse.iloc[:, 0].to_numpy(dtype=np.float64)

    @staticmethod
    def _raw_momentum_prediction(
        baseline: str,
        raw: DataFrame,
        model: _BaselinePredictor,
    ) -> np.ndarray:
        if baseline == "momentum_1h":
            return raw[_feature_column(raw, "%-return_1h")].to_numpy(dtype=np.float64)
        if baseline == "momentum_24h":
            return (
                raw[_feature_column(raw, "%-return_24h")].to_numpy(dtype=np.float64)
                / model.momentum_24h_scale
            )
        if baseline == "volatility_adjusted_momentum_24h":
            score = raw[_feature_column(raw, "%-return_volatility_24h")].to_numpy(dtype=np.float64)
            return score / model.score_standard_deviation * model.target_standard_deviation
        raise ValueError(f"{baseline} is not a momentum baseline")

    def _raw_training_features(self, dates: Any) -> DataFrame:
        normalized = pd.to_datetime(dates, utc=True)
        raw = self._phase12_raw_training.reindex(normalized)
        if raw.isna().any().any():
            raise ValueError("raw baseline features and filtered training dates are misaligned")
        return raw

    def fit(
        self,
        data_dictionary: dict[str, Any],
        dk: FreqaiDataKitchen,
        **kwargs: Any,
    ) -> _BaselinePredictor:
        started = time.perf_counter()
        baseline = str(self.model_training_parameters["baseline"])
        if baseline not in BASELINES:
            raise ValueError(f"unknown Phase 12 baseline: {baseline}")
        ridge_alpha = float(self.model_training_parameters["ridge_alpha"])
        if ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be positive")
        train_features = data_dictionary["train_features"]
        train_labels = data_dictionary["train_labels"]
        dates = dk.data_dictionary["train_dates"]
        if len(dates) != len(train_features):
            raise ValueError("training dates and transformed features are misaligned")
        raw = self._raw_training_features(dates)
        actual_return = self._inverse_labels(train_labels, dk)
        scaled_target = train_labels.iloc[:, 0].to_numpy(dtype=np.float64)
        target_std = float(np.std(actual_return))
        model = _BaselinePredictor(
            baseline=baseline,
            target_standard_deviation=target_std,
            momentum_24h_scale=float(
                self.model_training_parameters.get("momentum_24h_scale", 24.0)
            ),
        )

        feature_rank = 0
        effective_rank = 0.0
        condition_number = 1.0
        if baseline == "zero_return":
            train_prediction = np.zeros_like(actual_return)
        elif baseline == "historical_mean":
            model.constant_raw = float(np.mean(actual_return))
            train_prediction = np.full_like(actual_return, model.constant_raw)
        elif baseline in {"market_ols", "market_ridge"}:
            inputs = train_features.to_numpy(dtype=np.float64)
            model.feature_mean = np.mean(inputs, axis=0)
            centered_inputs = inputs - model.feature_mean
            model.intercept_scaled = float(np.mean(scaled_target))
            centered_target = scaled_target - model.intercept_scaled
            singular_values = np.linalg.svd(centered_inputs, compute_uv=False)
            tolerance = max(centered_inputs.shape) * np.finfo(np.float64).eps * singular_values[0]
            retained = singular_values[singular_values > tolerance]
            feature_rank = int(retained.size)
            eigenvalues = retained**2
            effective_rank = float(eigenvalues.sum() ** 2 / np.square(eigenvalues).sum())
            condition_number = float(retained[0] / retained[-1])
            if baseline == "market_ols":
                model.coefficient = np.linalg.lstsq(
                    centered_inputs,
                    centered_target,
                    rcond=1e-12,
                )[0]
            else:
                system = centered_inputs.T @ centered_inputs
                system.flat[:: system.shape[0] + 1] += ridge_alpha
                model.coefficient = np.linalg.solve(
                    system,
                    centered_inputs.T @ centered_target,
                )
            scaled_prediction = model.intercept_scaled + centered_inputs @ model.coefficient
            train_prediction = self._inverse_labels(scaled_prediction, dk)
        elif baseline in {
            "momentum_1h",
            "momentum_24h",
            "volatility_adjusted_momentum_24h",
        }:
            if baseline == "volatility_adjusted_momentum_24h":
                score = raw[_feature_column(raw, "%-return_volatility_24h")].to_numpy(
                    dtype=np.float64
                )
                model.score_standard_deviation = float(np.std(score))
                if model.score_standard_deviation <= 0:
                    raise ValueError("volatility-adjusted momentum score has zero variance")
            train_prediction = self._raw_momentum_prediction(baseline, raw, model)
            feature_rank = 1
            effective_rank = 1.0
        elif baseline == "buy_and_hold":
            model.constant_raw = 1e-12
            train_prediction = np.full_like(actual_return, model.constant_raw)
        else:
            raise AssertionError("unreachable baseline")

        metrics = prediction_metrics(actual_return, train_prediction)
        record = {
            "event": "train",
            "phase": 12,
            "run_id": self.model_training_parameters["phase12_run_id"],
            "identifier": self.identifier,
            "baseline": baseline,
            "pair": dk.pair,
            "timeframe": self.config["timeframe"],
            "train_period_days": self.freqai_info["train_period_days"],
            "backtest_period_days": self.freqai_info["backtest_period_days"],
            "raw_rows": self._phase12_raw_rows,
            "raw_start": self._phase12_raw_start,
            "raw_end": self._phase12_raw_end,
            "effective_n": len(train_features),
            "filtered_start": _timestamp(dates.iloc[0]),
            "filtered_end": _timestamp(dates.iloc[-1]),
            "input_feature_count": train_features.shape[1],
            "feature_count": (
                25
                if baseline in {"market_ols", "market_ridge"}
                else (1 if baseline.startswith("momentum") or "momentum" in baseline else 0)
            ),
            "rank": feature_rank,
            "effective_rank": effective_rank,
            "condition_number": condition_number,
            "ridge_alpha": ridge_alpha if baseline == "market_ridge" else 0.0,
            "momentum_24h_scale": float(
                self.model_training_parameters.get("momentum_24h_scale", 24.0)
            ),
            "train_mse": metrics["mse"],
            "training_total_seconds": time.perf_counter() - started,
            "peak_vram_mib": 0.0,
            "cuda_device": "cpu_baseline",
            "dtype": "float64",
            "model_window": dk.data_path.name,
            **{f"train_{key}": value for key, value in metrics.items()},
        }
        output_path = Path(self.model_training_parameters["phase12_metrics_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return model

    def predict(
        self,
        unfiltered_df: DataFrame,
        dk: FreqaiDataKitchen,
        **kwargs: Any,
    ) -> tuple[DataFrame, npt.NDArray[np.int_]]:
        baseline = self.model.baseline
        dk.find_features(unfiltered_df)
        prediction_features, _ = dk.filter_features(
            unfiltered_df,
            dk.training_features_list,
            training_filter=False,
        )
        prediction_features, outliers, _ = dk.feature_pipeline.transform(
            prediction_features,
            outlier_check=True,
        )
        dk.data_dictionary["prediction_features"] = prediction_features
        if baseline in {"market_ols", "market_ridge"}:
            scaled = self.model.predict_scaled(prediction_features)
            predictions = self._inverse_labels(scaled, dk)
        elif baseline == "zero_return":
            predictions = np.zeros(len(unfiltered_df), dtype=np.float64)
        elif baseline in {"historical_mean", "buy_and_hold"}:
            predictions = np.full(
                len(unfiltered_df),
                self.model.constant_raw,
                dtype=np.float64,
            )
        elif baseline in {
            "momentum_1h",
            "momentum_24h",
            "volatility_adjusted_momentum_24h",
        }:
            predictions = self._raw_momentum_prediction(baseline, unfiltered_df, self.model)
        else:
            raise ValueError(f"unknown Phase 12 baseline: {baseline}")
        if not np.isfinite(predictions).all():
            raise ValueError(f"{baseline} produced non-finite predictions")
        pred_df = DataFrame(predictions, columns=dk.label_list)
        if dk.feature_pipeline["di"]:
            dk.DI_values = dk.feature_pipeline["di"].di_values
        else:
            dk.DI_values = np.zeros(outliers.shape[0])
        dk.do_predict = outliers
        if len(pred_df) != len(dk.do_predict):
            raise RuntimeError("baseline predictions and outlier mask are misaligned")
        return pred_df, dk.do_predict
