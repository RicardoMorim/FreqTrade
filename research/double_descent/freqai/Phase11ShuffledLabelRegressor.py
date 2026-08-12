"""FreqAI adapter for the Phase 11 shuffled-label negative control."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from research.double_descent.freqai.Phase4CudaRFFRegressor import (
    _CudaWorkerClient,
    _timestamp,
)
from research.double_descent.freqai.Phase10CudaControlRegressor import (
    Phase10CudaControlRegressor,
    _date_sample_ids,
    _Phase10Predictor,
)
from research.double_descent.metrics import prediction_metrics
from research.double_descent.rff_cuda_worker import _splitmix64


def deterministic_label_permutation(sample_ids: np.ndarray, seed: int) -> np.ndarray:
    """Return a timestamp-keyed cyclic derangement, invariant to input row order."""
    identifiers = np.asarray(sample_ids)
    if identifiers.ndim != 1 or identifiers.size < 2:
        raise ValueError("label shuffling requires at least two 1D sample ids")
    if not np.issubdtype(identifiers.dtype, np.integer):
        raise ValueError("sample ids must be integers")
    identifiers = identifiers.astype(np.int64, copy=False)
    if np.unique(identifiers).size != identifiers.size:
        raise ValueError("sample ids must be unique")
    counters = identifiers.view(np.uint64) ^ np.uint64(seed)
    order = np.argsort(_splitmix64(counters), kind="stable")
    source_order = np.roll(order, 1)
    permutation = np.empty(identifiers.size, dtype=np.int64)
    permutation[order] = source_order
    if np.any(permutation == np.arange(identifiers.size)):
        raise RuntimeError("the deterministic label permutation is not a derangement")
    return permutation


def permutation_fingerprint(sample_ids: np.ndarray, permutation: np.ndarray) -> str:
    """Hash the destination-to-source timestamp mapping for matched-P audits."""
    source_ids = np.asarray(sample_ids, dtype=np.int64)[permutation]
    mapping = np.column_stack((np.asarray(sample_ids, dtype=np.int64), source_ids))
    mapping = mapping[np.argsort(mapping[:, 0], kind="stable")]
    return hashlib.sha256(mapping.tobytes()).hexdigest()


class Phase11ShuffledLabelRegressor(Phase10CudaControlRegressor):
    """Fit nested market RFF after destroying the in-window X/y relationship."""

    def _get_client(self) -> _CudaWorkerClient:
        if not hasattr(self, "_phase11_client"):
            metrics_path = Path(self.model_training_parameters["phase11_metrics_path"])
            worker_log = metrics_path.with_name(metrics_path.stem + "_cuda_worker.log")
            self._phase11_client = _CudaWorkerClient(
                self.model_training_parameters["cuda_python_executable"], worker_log
            )
        return self._phase11_client

    def fit(
        self,
        data_dictionary: dict[str, Any],
        dk: FreqaiDataKitchen,
        **kwargs: Any,
    ) -> _Phase10Predictor:
        train_features = data_dictionary["train_features"]
        train_labels = data_dictionary["train_labels"]
        parameters = {
            "feature_count": int(self.model_training_parameters["feature_count"]),
            "chunk_size": int(self.model_training_parameters["chunk_size"]),
            "gamma": float(self.model_training_parameters["gamma"]),
            "seed": int(self.model_training_parameters["seed"]),
            "ridge": float(self.model_training_parameters["ridge"]),
            "rcond": float(self.model_training_parameters["rcond"]),
            "dtype": str(self.model_training_parameters["dtype"]),
            "representation": "market_rff",
        }
        dates = dk.data_dictionary["train_dates"]
        if len(dates) != len(train_features):
            raise ValueError("training dates and transformed features are misaligned")
        sample_ids = _date_sample_ids(dates)
        shuffle_seed = int(self.model_training_parameters["label_shuffle_seed"])
        permutation = deterministic_label_permutation(sample_ids, shuffle_seed)
        original_scaled = train_labels.iloc[:, 0].to_numpy()
        shuffled_scaled = original_scaled[permutation]
        if not np.array_equal(np.sort(original_scaled), np.sort(shuffled_scaled)):
            raise RuntimeError("label permutation changed the target multiset")

        client = self._get_client()
        scaled_prediction, diagnostics = client.fit(
            train_features.to_numpy(),
            shuffled_scaled,
            parameters,
            sample_ids=sample_ids,
        )
        original_return = self._inverse_labels(train_labels, dk)
        shuffled_return = self._inverse_labels(shuffled_scaled, dk)
        predicted_return = self._inverse_labels(scaled_prediction, dk)
        shuffled_fit_metrics = prediction_metrics(shuffled_return, predicted_return)
        original_fit_metrics = prediction_metrics(original_return, predicted_return)
        randomization_metrics = prediction_metrics(original_return, shuffled_return)
        record = {
            "event": "train",
            "phase": 11,
            "run_id": self.model_training_parameters["phase11_run_id"],
            "identifier": self.identifier,
            "representation": "market_rff",
            "labels_shuffled": True,
            "label_shuffle_seed": shuffle_seed,
            "label_permutation_identity": False,
            "label_multiset_preserved": True,
            "label_fixed_point_count": int(np.sum(permutation == np.arange(len(permutation)))),
            "label_permutation_sha256": permutation_fingerprint(sample_ids, permutation),
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
            **{f"train_{key}": value for key, value in shuffled_fit_metrics.items()},
            **{f"original_train_{key}": value for key, value in original_fit_metrics.items()},
            **{f"label_shuffle_{key}": value for key, value in randomization_metrics.items()},
            **diagnostics,
        }
        output_path = Path(self.model_training_parameters["phase11_metrics_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return _Phase10Predictor(client)
