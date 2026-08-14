"""FreqAI adapter for the persistent CUDA nested-RFF minimum-norm solver."""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from research.double_descent.metrics import prediction_metrics


def _timestamp(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


class _CudaWorkerClient:
    def __init__(
        self,
        python_executable: str,
        stderr_path: Path,
        worker_module: str = "research.double_descent.rff_cuda_worker",
        temporary_prefix: str = "phase4-rff-",
    ) -> None:
        executable = Path(python_executable)
        if not executable.is_file():
            raise FileNotFoundError(f"CUDA Python executable does not exist: {executable}")
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr_handle = stderr_path.open("w+b")
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        self._process = subprocess.Popen(
            [
                str(executable),
                "-u",
                "-m",
                worker_module,
                "--serve",
            ],
            cwd=Path.cwd(),
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_handle,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self._temporary_directory = tempfile.TemporaryDirectory(prefix=temporary_prefix)
        self._path = Path(self._temporary_directory.name)
        self._request_index = 0
        self._closed = False
        atexit.register(self.close)

    @property
    def closed(self) -> bool:
        return self._closed

    def _request(self, command: str, **payload: Any) -> dict[str, Any]:
        if self._closed or self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("CUDA worker is closed")
        request = {"command": command, **payload}
        self._process.stdin.write(json.dumps(request) + "\n")
        self._process.stdin.flush()
        response_line = self._process.stdout.readline()
        if not response_line:
            self._stderr_handle.flush()
            self._stderr_handle.seek(0)
            stderr = self._stderr_handle.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"CUDA worker exited with code {self._process.poll()}: {stderr[-4000:]}"
            )
        response = json.loads(response_line)
        if not response.get("ok"):
            raise RuntimeError(f"CUDA worker {response.get('error_type')}: {response.get('error')}")
        return response["result"]

    def fit(
        self,
        inputs: np.ndarray,
        target: np.ndarray,
        parameters: dict[str, Any],
        sample_ids: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self._request_index += 1
        prefix = self._path / f"fit_{self._request_index}"
        inputs_path = prefix.with_name(prefix.name + "_inputs.npy")
        target_path = prefix.with_name(prefix.name + "_target.npy")
        prediction_path = prefix.with_name(prefix.name + "_prediction.npy")
        np.save(inputs_path, inputs, allow_pickle=False)
        np.save(target_path, target, allow_pickle=False)
        sample_ids_payload = {}
        if sample_ids is not None:
            sample_ids_path = prefix.with_name(prefix.name + "_sample_ids.npy")
            np.save(sample_ids_path, sample_ids, allow_pickle=False)
            sample_ids_payload["sample_ids_path"] = str(sample_ids_path)
        diagnostics = self._request(
            "fit",
            inputs_path=str(inputs_path),
            target_path=str(target_path),
            train_prediction_path=str(prediction_path),
            **sample_ids_payload,
            **parameters,
        )
        return np.load(prediction_path, allow_pickle=False), diagnostics

    def predict(
        self,
        inputs: np.ndarray,
        sample_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        self._request_index += 1
        prefix = self._path / f"predict_{self._request_index}"
        inputs_path = prefix.with_name(prefix.name + "_inputs.npy")
        output_path = prefix.with_name(prefix.name + "_output.npy")
        np.save(inputs_path, inputs, allow_pickle=False)
        sample_ids_payload = {}
        if sample_ids is not None:
            sample_ids_path = prefix.with_name(prefix.name + "_sample_ids.npy")
            np.save(sample_ids_path, sample_ids, allow_pickle=False)
            sample_ids_payload["sample_ids_path"] = str(sample_ids_path)
        diagnostics = self._request(
            "predict",
            inputs_path=str(inputs_path),
            output_path=str(output_path),
            **sample_ids_payload,
        )
        prediction = np.load(output_path, allow_pickle=False)
        if not diagnostics["prediction_finite"]:
            raise RuntimeError("CUDA worker returned non-finite predictions")
        return prediction

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._process.poll() is None:
                self._request("shutdown")
                self._process.wait(timeout=10)
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            if self._process.poll() is None:
                self._process.kill()
                self._process.wait(timeout=5)
        finally:
            self._closed = True
            self._stderr_handle.close()
            self._temporary_directory.cleanup()


class _CudaRFFPredictor:
    def __init__(self, client: _CudaWorkerClient, close_after_predict: bool = False) -> None:
        self._client = client
        self._close_after_predict = close_after_predict

    def predict(self, features: DataFrame) -> np.ndarray:
        try:
            return self._client.predict(features.to_numpy())
        finally:
            if self._close_after_predict:
                self._client.close()


class Phase4CudaRFFRegressor(BaseRegressionModel):
    """Fit one deterministic RFF ridge model in each genuine FreqAI window."""

    def train(
        self,
        unfiltered_df: DataFrame,
        pair: str,
        dk: FreqaiDataKitchen,
        **kwargs: Any,
    ) -> Any:
        self._phase4_raw_rows = len(unfiltered_df)
        self._phase4_raw_start = _timestamp(unfiltered_df["date"].iloc[0])
        self._phase4_raw_end = _timestamp(unfiltered_df["date"].iloc[-1])
        return super().train(unfiltered_df, pair, dk, **kwargs)

    def _get_client(self) -> _CudaWorkerClient:
        if not hasattr(self, "_phase4_client") or self._phase4_client.closed:
            metrics_path = Path(self.model_training_parameters["phase4_metrics_path"])
            worker_log = metrics_path.with_name(metrics_path.stem + "_cuda_worker.log")
            self._phase4_client = _CudaWorkerClient(
                self.model_training_parameters["cuda_python_executable"], worker_log
            )
        return self._phase4_client

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
    ) -> _CudaRFFPredictor:
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
            "run_id": self.model_training_parameters["phase4_run_id"],
            "identifier": self.identifier,
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
            "rff_feature_count": parameters["feature_count"],
            "pn_ratio": parameters["feature_count"] / len(train_features),
            "model_window": dk.data_path.name,
            **{f"train_{key}": value for key, value in metrics.items()},
            **diagnostics,
        }
        output_path = Path(self.model_training_parameters["phase4_metrics_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return _CudaRFFPredictor(
            client,
            close_after_predict=bool(
                self.model_training_parameters.get(
                    "phase17_close_cuda_worker_after_predict", False
                )
            ),
        )
