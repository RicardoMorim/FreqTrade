"""Clock-time-matched 15-minute strategy for Phase 15 frequency tests."""

from __future__ import annotations

import numpy as np
import talib.abstract as ta
from pandas import DataFrame

from research.double_descent.freqai.Phase4RFFStrategy import Phase4RFFStrategy


BARS_PER_HOUR = 4


class Phase15TimeframeStrategy(Phase4RFFStrategy):
    """Use 15-minute candles while preserving the Phase 4 feature horizons in hours."""

    timeframe = "15m"
    startup_candle_count = 200 * BARS_PER_HOUR + 1

    def feature_engineering_standard(
        self,
        dataframe: DataFrame,
        metadata: dict,
        **kwargs,
    ) -> DataFrame:
        close = dataframe["close"]
        volume = dataframe["volume"]
        bar_return = close.pct_change()

        for horizon_hours in (1, 3, 6, 12, 24, 72, 168):
            bars = horizon_hours * BARS_PER_HOUR
            dataframe[f"%-return_{horizon_hours}h"] = close.pct_change(bars)

        for window_hours in (6, 24, 72):
            bars = window_hours * BARS_PER_HOUR
            dataframe[f"%-volatility_{window_hours}h"] = bar_return.rolling(bars).std()

        one_hour_volume = volume.rolling(BARS_PER_HOUR).sum()
        dataframe["%-volume_change_1h"] = one_hour_volume.pct_change(BARS_PER_HOUR)
        dataframe["%-relative_volume_24h"] = (
            one_hour_volume / one_hour_volume.rolling(24 * BARS_PER_HOUR).mean() - 1.0
        )
        trailing_high = dataframe["high"].rolling(BARS_PER_HOUR).max()
        trailing_low = dataframe["low"].rolling(BARS_PER_HOUR).min()
        trailing_open = dataframe["open"].shift(BARS_PER_HOUR - 1)
        dataframe["%-candle_range"] = (trailing_high - trailing_low) / close
        dataframe["%-candle_body"] = (close - trailing_open) / trailing_open

        indicator_period = 14 * BARS_PER_HOUR
        dataframe["%-rsi_14"] = ta.RSI(dataframe, timeperiod=indicator_period) / 100.0
        dataframe["%-adx_14"] = ta.ADX(dataframe, timeperiod=indicator_period) / 100.0
        dataframe["%-atr_14_normalized"] = ta.ATR(dataframe, timeperiod=indicator_period) / close

        for window_hours in (12, 24, 72):
            bars = window_hours * BARS_PER_HOUR
            dataframe[f"%-sma_distance_{window_hours}h"] = close / close.rolling(bars).mean() - 1.0

        volatility_24h = dataframe["%-volatility_24h"].replace(0.0, np.nan)
        dataframe["%-return_volatility_24h"] = dataframe["%-return_24h"] / (
            volatility_24h * np.sqrt(24.0 * BARS_PER_HOUR)
        )

        hour = dataframe["date"].dt.hour + dataframe["date"].dt.minute / 60.0
        hour_angle = 2.0 * np.pi * hour / 24.0
        weekday_angle = 2.0 * np.pi * dataframe["date"].dt.dayofweek / 7.0
        dataframe["%-hour_sin"] = np.sin(hour_angle)
        dataframe["%-hour_cos"] = np.cos(hour_angle)
        dataframe["%-weekday_sin"] = np.sin(weekday_angle)
        dataframe["%-weekday_cos"] = np.cos(weekday_angle)
        return dataframe
