"""FreqAI probe model that records the effective sample count of every training window."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.dummy import DummyRegressor

from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen


logger = logging.getLogger(__name__)


def _timestamp(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


class Phase3SampleCountRegressor(BaseRegressionModel):
    """Run the genuine preprocessing path, persist its dimensions, then fit a mean baseline."""

    def train(
        self,
        unfiltered_df: DataFrame,
        pair: str,
        dk: FreqaiDataKitchen,
        **kwargs,
    ) -> Any:
        self._phase3_raw_rows = len(unfiltered_df)
        self._phase3_raw_start = _timestamp(unfiltered_df["date"].iloc[0])
        self._phase3_raw_end = _timestamp(unfiltered_df["date"].iloc[-1])
        return super().train(unfiltered_df, pair, dk, **kwargs)

    def fit(
        self,
        data_dictionary: dict[str, Any],
        dk: FreqaiDataKitchen,
        **kwargs,
    ) -> DummyRegressor:
        train_features = data_dictionary["train_features"]
        train_labels = data_dictionary["train_labels"]
        test_features = data_dictionary["test_features"]
        filtered_dates = dk.data_dictionary["train_dates"]
        filtered_rows = len(filtered_dates)
        final_train_rows = len(train_features)
        final_test_rows = len(test_features)
        record = {
            "run_id": self.model_training_parameters["phase3_run_id"],
            "identifier": self.identifier,
            "pair": dk.pair,
            "timeframe": self.config["timeframe"],
            "train_period_days": self.freqai_info["train_period_days"],
            "backtest_period_days": self.freqai_info["backtest_period_days"],
            "label_period_candles": self.ft_params["label_period_candles"],
            "raw_rows": self._phase3_raw_rows,
            "raw_start": self._phase3_raw_start,
            "raw_end": self._phase3_raw_end,
            "filtered_rows_before_split": filtered_rows,
            "filtered_start": _timestamp(filtered_dates.iloc[0]),
            "filtered_end": _timestamp(filtered_dates.iloc[-1]),
            "nan_dropped_rows": self._phase3_raw_rows - filtered_rows,
            "final_train_rows": final_train_rows,
            "final_test_rows": final_test_rows,
            "pipeline_removed_rows": filtered_rows - final_train_rows - final_test_rows,
            "declared_feature_count": len(dk.training_features_list),
            "model_feature_count": train_features.shape[1],
            "label_count": train_labels.shape[1],
            "nonfinite_train_feature_values": int(
                np.size(train_features) - np.isfinite(train_features.to_numpy()).sum()
            ),
            "nonfinite_train_label_values": int(
                np.size(train_labels) - np.isfinite(train_labels.to_numpy()).sum()
            ),
            "model_window": dk.data_path.name,
        }
        output_path = Path(self.model_training_parameters["phase3_metrics_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        logger.info(
            "PHASE3_EFFECTIVE_N pair=%s train_days=%s N=%s raw=%s features=%s",
            dk.pair,
            self.freqai_info["train_period_days"],
            final_train_rows,
            self._phase3_raw_rows,
            train_features.shape[1],
        )

        model = DummyRegressor(strategy="mean")
        model.fit(
            train_features,
            train_labels,
            sample_weight=data_dictionary["train_weights"],
        )
        return model
