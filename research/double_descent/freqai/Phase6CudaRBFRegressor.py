"""FreqAI adapter for the exact centered RBF-kernel limit on CUDA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from pandas import DataFrame

from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from research.double_descent.freqai.Phase4CudaRFFRegressor import (
    _CudaWorkerClient,
    _timestamp,
)
from research.double_descent.metrics import prediction_metrics


class _CudaKernelPredictor:
    def __init__(self, client: _CudaWorkerClient) -> None:
        self._client = client

    def predict(self, features: DataFrame) -> np.ndarray:
        return self._client.predict(features.to_numpy())


class Phase6CudaRBFRegressor(BaseRegressionModel):
    """Fit the exact ridgeless RBF kernel in each genuine rolling FreqAI window."""

    def train(
        self,
        unfiltered_df: DataFrame,
        pair: str,
        dk: FreqaiDataKitchen,
        **kwargs: Any,
    ) -> Any:
        self._phase6_raw_rows = len(unfiltered_df)
        self._phase6_raw_start = _timestamp(unfiltered_df["date"].iloc[0])
        self._phase6_raw_end = _timestamp(unfiltered_df["date"].iloc[-1])
        return super().train(unfiltered_df, pair, dk, **kwargs)

    def _get_client(self) -> _CudaWorkerClient:
        if not hasattr(self, "_phase6_client"):
            metrics_path = Path(self.model_training_parameters["phase6_metrics_path"])
            worker_log = metrics_path.with_name(metrics_path.stem + "_cuda_worker.log")
            self._phase6_client = _CudaWorkerClient(
                self.model_training_parameters["cuda_python_executable"],
                worker_log,
                worker_module="research.double_descent.kernel_cuda_worker",
                temporary_prefix="phase6-kernel-",
            )
        return self._phase6_client

    @staticmethod
    def _inverse_labels(
        values: np.ndarray | DataFrame,
        dk: FreqaiDataKitchen,
    ) -> np.ndarray:
        frame = DataFrame(np.asarray(values).reshape(-1, 1), columns=dk.label_list)
        inverse, _, _ = dk.label_pipeline.inverse_transform(frame)
        return inverse.iloc[:, 0].to_numpy(dtype=np.float64)

    def fit(
        self,
        data_dictionary: dict[str, Any],
        dk: FreqaiDataKitchen,
        **kwargs: Any,
    ) -> _CudaKernelPredictor:
        train_features = data_dictionary["train_features"]
        train_labels = data_dictionary["train_labels"]
        parameters = {
            "gamma": float(self.model_training_parameters["gamma"]),
            "ridge": float(self.model_training_parameters["ridge"]),
            "rcond": float(self.model_training_parameters["rcond"]),
            "dtype": str(self.model_training_parameters["dtype"]),
        }
        client = self._get_client()
        scaled_prediction, diagnostics = client.fit(
            train_features.to_numpy(),
            train_labels.iloc[:, 0].to_numpy(),
            parameters,
        )
        actual_return = self._inverse_labels(train_labels, dk)
        predicted_return = self._inverse_labels(scaled_prediction, dk)
        metrics = prediction_metrics(actual_return, predicted_return)
        dates = dk.data_dictionary["train_dates"]
        record = {
            "event": "train",
            "run_id": self.model_training_parameters["phase6_run_id"],
            "identifier": self.identifier,
            "pair": dk.pair,
            "timeframe": self.config["timeframe"],
            "train_period_days": self.freqai_info["train_period_days"],
            "backtest_period_days": self.freqai_info["backtest_period_days"],
            "raw_rows": self._phase6_raw_rows,
            "raw_start": self._phase6_raw_start,
            "raw_end": self._phase6_raw_end,
            "effective_n": len(train_features),
            "filtered_start": _timestamp(dates.iloc[0]),
            "filtered_end": _timestamp(dates.iloc[-1]),
            "input_feature_count": train_features.shape[1],
            "model_window": dk.data_path.name,
            **{f"train_{key}": value for key, value in metrics.items()},
            **diagnostics,
        }
        output_path = Path(self.model_training_parameters["phase6_metrics_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return _CudaKernelPredictor(client)
