"""FreqAI adapter for the Phase 10 market/noise representation controls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas import DataFrame

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from research.double_descent.freqai.Phase4CudaRFFRegressor import (
    Phase4CudaRFFRegressor,
    _CudaWorkerClient,
    _timestamp,
)
from research.double_descent.metrics import prediction_metrics


def _date_sample_ids(values: Any) -> np.ndarray:
    dates = pd.to_datetime(values, utc=True)
    if bool(pd.isna(dates).any()):
        raise ValueError("sample dates contain missing values")
    sample_ids = np.asarray(dates.astype("int64"), dtype=np.int64)
    if np.unique(sample_ids).size != sample_ids.size:
        raise ValueError("sample dates must be unique")
    return sample_ids


class _Phase10Predictor:
    def __init__(self, client: _CudaWorkerClient) -> None:
        self._client = client

    def predict(self, features: DataFrame, sample_ids: np.ndarray) -> np.ndarray:
        return self._client.predict(features.to_numpy(), sample_ids=sample_ids)


class Phase10CudaControlRegressor(Phase4CudaRFFRegressor):
    """Fit one frozen market/noise control in every genuine rolling FreqAI window."""

    def _get_client(self) -> _CudaWorkerClient:
        if not hasattr(self, "_phase10_client"):
            metrics_path = Path(self.model_training_parameters["phase10_metrics_path"])
            worker_log = metrics_path.with_name(metrics_path.stem + "_cuda_worker.log")
            self._phase10_client = _CudaWorkerClient(
                self.model_training_parameters["cuda_python_executable"], worker_log
            )
        return self._phase10_client

    def fit(
        self,
        data_dictionary: dict[str, Any],
        dk: FreqaiDataKitchen,
        **kwargs: Any,
    ) -> _Phase10Predictor:
        train_features = data_dictionary["train_features"]
        train_labels = data_dictionary["train_labels"]
        representation = str(self.model_training_parameters["representation"])
        parameters = {
            "feature_count": int(self.model_training_parameters["feature_count"]),
            "chunk_size": int(self.model_training_parameters["chunk_size"]),
            "gamma": float(self.model_training_parameters["gamma"]),
            "seed": int(self.model_training_parameters["seed"]),
            "ridge": float(self.model_training_parameters["ridge"]),
            "rcond": float(self.model_training_parameters["rcond"]),
            "dtype": str(self.model_training_parameters["dtype"]),
            "representation": representation,
        }
        dates = dk.data_dictionary["train_dates"]
        if len(dates) != len(train_features):
            raise ValueError("training dates and transformed features are misaligned")
        sample_ids = _date_sample_ids(dates)
        client = self._get_client()
        scaled_prediction, diagnostics = client.fit(
            train_features.to_numpy(),
            train_labels.iloc[:, 0].to_numpy(),
            parameters,
            sample_ids=sample_ids,
        )
        actual_return = self._inverse_labels(train_labels, dk)
        predicted_return = self._inverse_labels(scaled_prediction, dk)
        metrics = prediction_metrics(actual_return, predicted_return)
        record = {
            "event": "train",
            "phase": 10,
            "run_id": self.model_training_parameters["phase10_run_id"],
            "identifier": self.identifier,
            "representation": representation,
            "pair": dk.pair,
            "timeframe": self.config["timeframe"],
            "train_period_days": self.freqai_info["train_period_days"],
            "backtest_period_days": self.freqai_info["backtest_period_days"],
            "raw_rows": self._phase4_raw_rows,
            "raw_start": self._phase4_raw_start,
            "raw_end": self._phase4_raw_end,
            "effective_n": len(train_features),
            "filtered_start": _timestamp(dates.iloc[0]),
            "filtered_end": _timestamp(dates.iloc[-1]),
            "input_feature_count": train_features.shape[1],
            "explicit_feature_count": parameters["feature_count"],
            "pn_ratio": parameters["feature_count"] / len(train_features),
            "model_window": dk.data_path.name,
            **{f"train_{key}": value for key, value in metrics.items()},
            **diagnostics,
        }
        output_path = Path(self.model_training_parameters["phase10_metrics_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return _Phase10Predictor(client)

    def predict(
        self,
        unfiltered_df: DataFrame,
        dk: FreqaiDataKitchen,
        **kwargs: Any,
    ) -> tuple[DataFrame, npt.NDArray[np.int_]]:
        dk.find_features(unfiltered_df)
        prediction_features, _ = dk.filter_features(
            unfiltered_df,
            dk.training_features_list,
            training_filter=False,
        )
        dates = unfiltered_df["date"]
        if len(dates) != len(prediction_features):
            raise ValueError("prediction dates and filtered features are misaligned")
        sample_ids = _date_sample_ids(dates)
        prediction_features, outliers, _ = dk.feature_pipeline.transform(
            prediction_features,
            outlier_check=True,
        )
        dk.data_dictionary["prediction_features"] = prediction_features
        predictions = self.model.predict(prediction_features, sample_ids)
        if self.CONV_WIDTH == 1:
            predictions = np.reshape(predictions, (-1, len(dk.label_list)))
        pred_df = DataFrame(predictions, columns=dk.label_list)
        pred_df, _, _ = dk.label_pipeline.inverse_transform(pred_df)
        if dk.feature_pipeline["di"]:
            dk.DI_values = dk.feature_pipeline["di"].di_values
        else:
            dk.DI_values = np.zeros(outliers.shape[0])
        dk.do_predict = outliers
        return pred_df, dk.do_predict
