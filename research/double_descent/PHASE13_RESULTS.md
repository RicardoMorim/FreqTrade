# Phase 13 results: causal market regimes

## Decision

Phase 13 completed the predeclared marginal-regime analysis, but the full adequacy gate did not
pass. All five marginal regimes had sufficient observations; one of the six joint cells,
`bull_high_volatility`, contained only 91 observations versus the frozen minimum of 250. Its
results are reported as descriptive and no conclusion is drawn from that cell. The threshold was
not relaxed after observing the shortfall.

Within the adequately sampled regimes, the result is clear: double-descent geometry persisted,
but useful benign overfitting did not. The interpolation peak and second descent appeared in bull,
bear, sideways, high-volatility, and low-volatility conditions in all three seeds. No market RFF
beat the zero-return forecast in any of those regimes.

## Frozen causal design

Regimes were assigned at prediction time using only trailing prices:

- trend score: trailing 30-day log return divided by trailing 30-day realized volatility;
- bull: score at least `+0.5`;
- bear: score at most `-0.5`;
- sideways: score between those boundaries;
- high volatility: trailing 30-day realized volatility above the median of its prior 60-day
  history;
- low volatility: at or below that prior rolling median.

The trend and volatility axes were primary marginal decompositions. Their six intersections were
predeclared as descriptive joint cells. Marginal cells required at least 500 observations and
joint cells at least 250. No regime changed a prediction, position, or trade; this phase only
conditioned already-frozen OOS evidence from Phases 10 and 12.

Prediction comparisons used a two-sided mean squared-error differential against zero with a
24-hour Bartlett HAC variance. The family of 135 non-zero-model hypotheses across the five
marginal regimes used Benjamini-Hochberg FDR control at 5%. Trading remained excluded from model
inference.

## Data and integrity

- all 8,759 chronological 2025 OOS targets were classified;
- prediction coverage was identical across 28 sources: 21 market-RFF cases and seven prediction
  baselines;
- economic attribution covered 29 sources, adding buy-and-hold;
- the analysis generated 308 prediction-regime cells and 319 economic-regime cells;
- all trade entries mapped to the regime known on the preceding signal candle;
- regime PnL contributions exactly reproduced each original costed backtest;
- no model was retrained;
- the 2026 holdout remained sealed.

An initial development execution exposed a boundary error: the last 2025 target could use the
`2026-01-01 00:00` close. The loader was corrected to truncate market data strictly before the
holdout boundary, a regression test was added, and all final artifacts were regenerated. The final
classified interval ends at `2025-12-31 22:00`, matching the 8,759 targets used in earlier phases.

## Regime coverage

| Axis | Regime | OOS observations | Adequate |
| --- | --- | ---: | --- |
| Trend | Bear | 2,463 | Yes |
| Trend | Sideways | 4,087 | Yes |
| Trend | Bull | 2,209 | Yes |
| Volatility | Low | 5,045 | Yes |
| Volatility | High | 3,714 | Yes |
| Joint | Bear / low volatility | 514 | Yes |
| Joint | Bear / high volatility | 1,949 | Yes |
| Joint | Sideways / low volatility | 2,413 | Yes |
| Joint | Sideways / high volatility | 1,674 | Yes |
| Joint | Bull / low volatility | 2,118 | Yes |
| Joint | Bull / high volatility | 91 | **No** |

## Prediction results

| Marginal regime | Best RFF P/N | Best RFF MSE / zero | RFF cells beating zero | Seeds with double descent |
| --- | ---: | ---: | ---: | ---: |
| Bear | 0.10 | 1.1287 | 0/21 | 3/3 |
| Sideways | 0.10 | 1.2156 | 0/21 | 3/3 |
| Bull | 0.10 | 1.2906 | 0/21 | 3/3 |
| Low volatility | 0.10 | 1.2512 | 0/21 | 3/3 |
| High volatility | 0.10 | 1.1178 | 0/21 | 3/3 |

The RFF curve recovered between 99.95% and 99.99% of its deterioration after the interpolation
peak, depending on regime and seed. That dramatic second descent still ended far above the useful
baselines: at `P/N=50`, MSE was 3.07-6.86 times the zero-return MSE across the five regimes.

All 105 market-RFF/marginal-regime cells were significantly worse than zero after HAC and FDR
control; none was significantly better. Zero return was the best simple forecast in every marginal
regime. Historical mean was the best non-zero alternative but was still 0.016%-0.279% worse than
zero, depending on regime.

The inadequately sampled `bull_high_volatility` joint cell contained an OLS result 2.79% below
zero MSE. With only 91 serially dependent hourly observations, joint cells excluded from the FDR
family, and no replication, it is not evidence of predictability.

## Economic diagnostics

All 28 active sources lost money over the full development period. Of the 105 market-RFF marginal
regime PnL cells, one adequately traded cell was positive: seed `1898170439`, `P=2,202`
(`P/N=1.02`) contributed +9.42% in the bear regime across 536 trades with profit factor 1.125.
The same model lost 72.63% in sideways conditions and 31.72% in bull conditions, finishing down
94.93% overall. This isolated descriptive cell does not rescue the model and was not used to build
a regime-switching strategy.

## Interpretation

The second descent is not concentrated in one friendly market state: its shape appears in every
adequately sampled marginal regime and every seed. That makes the geometry more robust, but not
more useful. In every market state, the recovered RFF forecasts remain worse than predicting zero.

Phase 13 therefore weakens explanations based purely on bullish periods, volatility clustering,
or a small number of stressed observations. It supports the narrower conclusion that the
interpolation phenomenon is representation-level geometry without demonstrated financial alpha.
Because one joint cell failed its sample floor, the full Phase 13 gate remains failed rather than
being retroactively relaxed.

## Artifacts

- `user_data/research_results/double_descent/phase13/summary.json`
- `user_data/research_results/double_descent/phase13/regime_assignments.csv`
- `user_data/research_results/double_descent/phase13/prediction_regime_metrics.csv`
- `user_data/research_results/double_descent/phase13/economic_regime_metrics.csv`
- `user_data/research_results/double_descent/phase13/regime_curve_assessments.csv`
- `user_data/research_results/double_descent/phase13/regime_comparisons.csv`
- `user_data/research_results/double_descent/phase13/market_regimes.html`
