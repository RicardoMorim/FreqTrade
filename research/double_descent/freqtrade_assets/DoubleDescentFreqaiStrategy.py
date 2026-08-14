from __future__ import annotations

import numpy as np
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import IStrategy


class DoubleDescentFreqaiStrategy(IStrategy):
    """Minimal FreqAI strategy for measuring model complexity, not strategy cleverness.

    The trading layer is intentionally simple: long when predicted future return is
    positive, short when negative, and exit on a sign flip. The purpose is to avoid
    hiding model-quality changes behind hand-tuned entry/exit logic.
    """

    timeframe = "1h"
    can_short = True
    process_only_new_candles = True
    startup_candle_count = 200
    stoploss = -0.99
    minimal_roi = {"0": 100.0}
    use_exit_signal = True

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        close = dataframe["close"]
        returns = close.pct_change()
        log_returns = np.log(close).diff()

        for lag in (1, 2, 3, 6, 12, 24, 72, 168):
            dataframe[f"%-return_{lag}"] = close.pct_change(lag)

        for window in (6, 12, 24, 72, 168):
            dataframe[f"%-vol_{window}"] = log_returns.rolling(window).std()
            dataframe[f"%-volume_ratio_{window}"] = (
                dataframe["volume"] / dataframe["volume"].rolling(window).mean()
            )

        dataframe["%-range"] = (dataframe["high"] - dataframe["low"]) / close
        dataframe["%-body"] = (dataframe["close"] - dataframe["open"]) / dataframe["open"]
        dataframe["%-rsi_14"] = ta.RSI(dataframe, timeperiod=14) / 100.0
        dataframe["%-adx_14"] = ta.ADX(dataframe, timeperiod=14) / 100.0
        dataframe["%-atr_14"] = ta.ATR(dataframe, timeperiod=14) / close
        dataframe["%-ema_gap_12"] = close / ta.EMA(dataframe, timeperiod=12) - 1.0
        dataframe["%-ema_gap_48"] = close / ta.EMA(dataframe, timeperiod=48) - 1.0
        dataframe["%-return_vol_interaction"] = returns / (
            log_returns.rolling(24).std() + 1e-8
        )
        dataframe["%-hour_sin"] = np.sin(2.0 * np.pi * dataframe["date"].dt.hour / 24.0)
        dataframe["%-hour_cos"] = np.cos(2.0 * np.pi * dataframe["date"].dt.hour / 24.0)
        dataframe["%-dow_sin"] = np.sin(
            2.0 * np.pi * dataframe["date"].dt.dayofweek / 7.0
        )
        dataframe["%-dow_cos"] = np.cos(
            2.0 * np.pi * dataframe["date"].dt.dayofweek / 7.0
        )
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        horizon = int(self.freqai_info["feature_parameters"]["label_period_candles"])
        dataframe["&-future_return"] = dataframe["close"].shift(-horizon) / dataframe["close"] - 1.0
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return self.freqai.start(dataframe, metadata, self)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        accepted = dataframe["do_predict"] == 1
        dataframe.loc[
            accepted & (dataframe["&-future_return"] > 0), ["enter_long", "enter_tag"]
        ] = (1, "dd_long")
        dataframe.loc[
            accepted & (dataframe["&-future_return"] < 0), ["enter_short", "enter_tag"]
        ] = (1, "dd_short")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        accepted = dataframe["do_predict"] == 1
        dataframe.loc[accepted & (dataframe["&-future_return"] < 0), "exit_long"] = 1
        dataframe.loc[accepted & (dataframe["&-future_return"] > 0), "exit_short"] = 1
        return dataframe
