"""Simple sign strategy used downstream of Phase 4 return predictions."""

from __future__ import annotations

from pandas import DataFrame

from research.double_descent.freqai.Phase3EffectiveNStrategy import (
    Phase3EffectiveNStrategy,
)


class Phase4RFFStrategy(Phase3EffectiveNStrategy):
    """Trade prediction sign without thresholds, filters, ROI tuning, or leverage."""

    minimal_roi = {"0": 1000.0}
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        valid = (dataframe["do_predict"] == 1) & (dataframe["volume"] > 0)
        dataframe.loc[valid & (dataframe["&-forward_return"] > 0), "enter_long"] = 1
        dataframe.loc[valid & (dataframe["&-forward_return"] < 0), "enter_short"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        valid = dataframe["do_predict"] == 1
        dataframe.loc[valid & (dataframe["&-forward_return"] < 0), "exit_long"] = 1
        dataframe.loc[valid & (dataframe["&-forward_return"] > 0), "exit_short"] = 1
        return dataframe

    def leverage(
        self,
        pair: str,
        current_time,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        return 1.0
