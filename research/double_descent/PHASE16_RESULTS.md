# Phase 16 results: economic robustness and execution costs

## Decision

Phase 16 passed every integrity gate. It repriced all 87 primary Phase 15 cases under six frozen
cost scenarios, producing 522 results. The native 10 bps scenario reconstructed every Freqtrade
trade and each case's total PnL within `1e-5`.

No RFF remained profitable under even the optimistic executable-cost scenario. The only two
positive cases at the reference and stress costs were the same one-trade PAXG exposure represented
by `historical_mean` and `buy_and_hold`. Both failed the predeclared minimum of 30 trades, so they
are not economic evidence.

## Cost sensitivity

| Scenario | Median net return | Median daily Sharpe | Positive cases | Insolvent frozen paths |
| --- | ---: | ---: | ---: | ---: |
| Price only | -2.30% | -0.243 | 37/87 | 0 |
| Funding only | -2.49% | -0.239 | 37/87 | 0 |
| Optimistic: 7.5 bps/side | -73.74% | -5.552 | 2/87 | 0 |
| Phase 15 reference: 10 bps/side | -99.11% | -10.893 | 2/87 | 0 |
| Conservative: 15 bps/side | -144.04% | -5.078 | 2/87 | 78 |
| Stress: 30 bps/side | -282.64% | -9.202 | 2/87 | 78 |

The returns below -100% in the last two rows are sensitivities with Phase 15 position amounts held
fixed. They show that the original high-turnover exposures would become insolvent; they are not a
claim that a real account could continue trading after liquidation.

Before costs, 29/63 RFF cases had positive price PnL. None survived the optimistic scenario. The
median RFF case made 2,388 trades and generated approximately 956 times starting balance in
two-sided turnover. This explains why a weak gross signal disappears under modest execution
costs. Funding changed the median result only slightly and is not the explanation.

## Trading double ascent

The common RFF grid produced a descriptive trading-double-ascent shape in 3/9 asset/seed curves
at the reference cost and 3/9 under stress. Zero curves had useful high-complexity economics in
either scenario. Shape alone is therefore not evidence of profitable second descent, just as in
the prediction analysis.

## Interpretation

Phase 16 rejects the economic version of the hypothesis for this frozen experiment. Some gross
price paths are positive, but the strategy's sign-flip rule produces extreme turnover and all RFF
profitability disappears at 7.5 bps per side. Prediction MSE already rejected useful benign
overfitting; the cost study shows that downstream trading does not rescue it.

The result remains a development-period conclusion. The 2026 holdout is still sealed and should
not be opened to optimize thresholds or reduce turnover after observing these results.

## Artifacts

- `user_data/research_results/double_descent/phase16/summary.json`
- `user_data/research_results/double_descent/phase16/economic_results.csv`
- `user_data/research_results/double_descent/phase16/scenario_summary.csv`
- `user_data/research_results/double_descent/phase16/trading_double_ascent.csv`
- `user_data/research_results/double_descent/phase16/case_manifest.csv`
- `user_data/research_results/double_descent/phase16/execution_cost_robustness.html`
