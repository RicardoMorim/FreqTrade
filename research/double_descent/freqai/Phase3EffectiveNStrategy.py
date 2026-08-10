"""Minimal causal FreqAI strategy used only to measure effective training N."""

from __future__ import annotations

import numpy as np
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import IStrategy


class Phase3EffectiveNStrategy(IStrategy):
    """Create the frozen market-state vector and invoke the real FreqAI pipeline."""

    timeframe = "1h"
    startup_candle_count = 200
    process_only_new_candles = True
    can_short = True
    minimal_roi = {"0": 0.0}
    stoploss = -0.99
    use_exit_signal = False

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        close = dataframe["close"]
        volume = dataframe["volume"]
        one_hour_return = close.pct_change()

        for horizon in (1, 3, 6, 12, 24, 72, 168):
            dataframe[f"%-return_{horizon}h"] = close.pct_change(horizon)

        for window in (6, 24, 72):
            dataframe[f"%-volatility_{window}h"] = one_hour_return.rolling(window).std()

        dataframe["%-volume_change_1h"] = volume.pct_change()
        dataframe["%-relative_volume_24h"] = volume / volume.rolling(24).mean() - 1.0
        dataframe["%-candle_range"] = (dataframe["high"] - dataframe["low"]) / close
        dataframe["%-candle_body"] = (dataframe["close"] - dataframe["open"]) / dataframe["open"]
        dataframe["%-rsi_14"] = ta.RSI(dataframe, timeperiod=14) / 100.0
        dataframe["%-adx_14"] = ta.ADX(dataframe, timeperiod=14) / 100.0
        dataframe["%-atr_14_normalized"] = ta.ATR(dataframe, timeperiod=14) / close

        for window in (12, 24, 72):
            dataframe[f"%-sma_distance_{window}h"] = close / close.rolling(window).mean() - 1.0

        volatility_24h = dataframe["%-volatility_24h"].replace(0.0, np.nan)
        dataframe["%-return_volatility_24h"] = dataframe["%-return_24h"] / (
            volatility_24h * np.sqrt(24.0)
        )

        hour_angle = 2.0 * np.pi * dataframe["date"].dt.hour / 24.0
        weekday_angle = 2.0 * np.pi * dataframe["date"].dt.dayofweek / 7.0
        dataframe["%-hour_sin"] = np.sin(hour_angle)
        dataframe["%-hour_cos"] = np.cos(hour_angle)
        dataframe["%-weekday_sin"] = np.sin(weekday_angle)
        dataframe["%-weekday_cos"] = np.cos(weekday_angle)
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        horizon = self.freqai_info["feature_parameters"]["label_period_candles"]
        dataframe["&-forward_return"] = (
            dataframe["close"].shift(-horizon) / dataframe["close"] - 1.0
        )
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return self.freqai.start(dataframe, metadata, self)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe
