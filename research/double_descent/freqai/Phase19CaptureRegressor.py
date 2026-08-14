"""Capture one genuine processed FreqAI window for Phase 19 diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen


def _date_ns(values: Any) -> np.ndarray:
    return pd.to_datetime(values, utc=True).to_numpy(dtype="datetime64[ns]").astype(np.int64)


class _ZeroCapturePredictor:
    def __init__(self, window_directory: Path) -> None:
        self.window_directory = window_directory

    def predict(self, features: DataFrame) -> np.ndarray:
        return np.zeros(len(features), dtype=np.float64)


class Phase19CaptureRegressor(BaseRegressionModel):
    """Persist scaled train/inference arrays without fitting a research model."""

    def train(
        self,
        unfiltered_df: DataFrame,
        pair: str,
        dk: FreqaiDataKitchen,
        **kwargs: Any,
    ) -> Any:
        self._phase19_raw_rows = len(unfiltered_df)
        self._phase19_raw_start = pd.Timestamp(unfiltered_df["date"].iloc[0]).isoformat()
        self._phase19_raw_end = pd.Timestamp(unfiltered_df["date"].iloc[-1]).isoformat()
        return super().train(unfiltered_df, pair, dk, **kwargs)

    @staticmethod
    def _inverse_label_anchors(dk: FreqaiDataKitchen) -> tuple[float, float]:
        anchors = DataFrame([0.0, 1.0], columns=dk.label_list)
        inverse, _, _ = dk.label_pipeline.inverse_transform(anchors)
        return float(inverse.iloc[0, 0]), float(inverse.iloc[1, 0])

    def fit(
        self,
        data_dictionary: dict[str, Any],
        dk: FreqaiDataKitchen,
        **kwargs: Any,
    ) -> _ZeroCapturePredictor:
        capture_root = Path(self.model_training_parameters["phase19_capture_directory"])
        window_directory = capture_root / "windows" / dk.data_path.name
        window_directory.mkdir(parents=True, exist_ok=True)
        features = data_dictionary["train_features"]
        labels = data_dictionary["train_labels"]
        dates = dk.data_dictionary["train_dates"]
        inverse_zero, inverse_one = self._inverse_label_anchors(dk)
        feature_values = features.to_numpy(dtype=np.float64)
        label_values = labels.iloc[:, 0].to_numpy(dtype=np.float64)
        if not np.isfinite(feature_values).all() or not np.isfinite(label_values).all():
            raise ValueError("Phase 19 capture received non-finite training arrays")
        np.savez_compressed(
            window_directory / "train.npz",
            features=feature_values,
            target_scaled=label_values,
            dates_ns=_date_ns(dates),
        )
        metadata = {
            "window": dk.data_path.name,
            "pair": dk.pair,
            "timeframe": self.config["timeframe"],
            "raw_rows": self._phase19_raw_rows,
            "raw_start": self._phase19_raw_start,
            "raw_end": self._phase19_raw_end,
            "effective_n": len(features),
            "input_feature_count": features.shape[1],
            "feature_columns": list(features.columns),
            "label": dk.label_list[0],
            "target_inverse_zero": inverse_zero,
            "target_inverse_one": inverse_one,
            "train_start": pd.Timestamp(dates.iloc[0]).isoformat(),
            "train_end": pd.Timestamp(dates.iloc[-1]).isoformat(),
        }
        (window_directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return _ZeroCapturePredictor(window_directory)

    def predict(
        self,
        unfiltered_df: DataFrame,
        dk: FreqaiDataKitchen,
        **kwargs: Any,
    ) -> tuple[DataFrame, np.ndarray]:
        prediction, do_predict = super().predict(unfiltered_df, dk, **kwargs)
        features = dk.data_dictionary["prediction_features"].to_numpy(dtype=np.float64)
        target = unfiltered_df[dk.label_list[0]].to_numpy(dtype=np.float64)
        dates = _date_ns(unfiltered_df["date"])
        if not (len(features) == len(target) == len(dates) == len(do_predict)):
            raise ValueError("Phase 19 inference capture arrays are misaligned")
        window_directory = self.model.window_directory
        np.savez_compressed(
            window_directory / "inference.npz",
            features=features,
            target_raw=target,
            dates_ns=dates,
            do_predict=np.asarray(do_predict, dtype=np.int8),
        )
        metadata_path = window_directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(
            {
                "inference_rows": len(features),
                "inference_start": pd.Timestamp(unfiltered_df["date"].iloc[0]).isoformat(),
                "inference_end": pd.Timestamp(unfiltered_df["date"].iloc[-1]).isoformat(),
                "finite_inference_targets": int(np.isfinite(target).sum()),
                "do_predict_rows": int(np.asarray(do_predict, dtype=bool).sum()),
            }
        )
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return prediction, do_predict
