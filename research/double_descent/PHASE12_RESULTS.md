# Phase 12 results: frozen simple baselines

## Decision

Phase 12 passed every integrity gate, but it does not support the use of a complex RFF model for
prediction or trading. The zero-return forecast had the lowest chronological OOS MSE. None of the
six other prediction baselines beat it, and the best Phase 10 market RFF was 20.33% worse than
zero. The best market RFF in the explicitly overparameterized region (`P/N >= 5`) was 4.52 times
worse than zero.

This result strengthens the distinction between double-descent geometry and useful benign
overfitting. Earlier phases found an interpolation peak and a second descent, including with up to
107,950 predictors (`P/N=50`). Phase 12 shows that the recovered high-dimensional models still do
not justify their complexity against predeclared simple references.

## Frozen design

Seven prediction baselines were declared before the run:

1. zero-return forecast;
2. rolling historical mean return;
3. OLS on the 25 standardized causal market inputs;
4. Ridge on the same inputs with frozen `alpha=1`;
5. one-hour momentum;
6. 24-hour momentum converted to an hourly forecast;
7. volatility-adjusted 24-hour momentum, scaled only with training-window statistics.

Buy-and-hold was included only as an economic reference. It was not ranked as a return-prediction
model and was not used to select any parameter. All cases used BTC perpetual futures at 1h,
90-day rolling training windows, 30-day evaluation steps, measured `N=2,159`, the 2025 development
period, 1x exposure, and the same 0.1% fee per side. The 2026 holdout remained sealed.

## Execution integrity

- 8/8 baseline cases succeeded;
- 104/104 rolling model fits succeeded (13 windows per baseline);
- all cases evaluated the same 8,759 chronological OOS observations;
- source data covered every required hour with no gaps or duplicate timestamps;
- the exact zero forecast generated no trades;
- buy-and-hold generated exactly one long trade;
- the independent CPU OLS reproduced the Phase 10 CUDA linear anchor: the OOS-MSE ratio was
  `1.000000000047` and prediction standard deviation agreed within the predeclared tolerance;
- trading metrics were recorded after fees but excluded from model selection;
- no 2026 observation was used.

## Prediction results

| Rank | Baseline | OOS MSE | MSE / zero | OOS R2 | IC |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Zero return | 0.0000224849 | 1.0000 | -0.0000003 | undefined (constant) |
| 2 | Historical mean | 0.0000224986 | 1.0006 | -0.000607 | -0.00214 |
| 3 | Market Ridge | 0.0000227710 | 1.0127 | -0.012723 | 0.02315 |
| 4 | Market OLS | 0.0000232984 | 1.0362 | -0.036179 | 0.00958 |
| 5 | Momentum 24h | 0.0000236643 | 1.0525 | -0.052453 | -0.03722 |
| 6 | Momentum 1h | 0.0000455313 | 2.0250 | -1.024967 | -0.01229 |
| 7 | Volatility-adjusted momentum 24h | 0.0000476691 | 2.1200 | -1.120046 | -0.02523 |

Ridge was the best learnable baseline and obtained a small positive IC and 51.10% directional
accuracy, but its forecast amplitudes still increased MSE by 1.27% relative to zero. Those weak
classification-style statistics did not translate into economic performance.

The best Phase 10 market RFF used only 216 predictors (`P/N=0.10`) and had OOS MSE
`0.0000270552`, 20.33% above zero and 18.81% above Ridge. The best tested market RFF at
`P/N >= 5` used 107,950 predictors (`P/N=50`) and had OOS MSE `0.000101533`, 4.52 times the zero
forecast. No RFF configuration beat the best simple baseline.

## Economic diagnostics

| Baseline | Net return | Sharpe | Profit factor | Trades |
| --- | ---: | ---: | ---: | ---: |
| Zero return | 0.00% | 0.00 | 0.000 | 0 |
| Historical mean | -5.23% | -0.02 | 0.813 | 4 |
| Buy-and-hold | -12.78% | -100.00 | 0.000 | 1 |
| Momentum 24h | -90.42% | -8.25 | 0.491 | 880 |
| Volatility-adjusted momentum 24h | -90.42% | -8.25 | 0.491 | 880 |
| Market Ridge | -93.66% | -10.84 | 0.627 | 1,859 |
| Market OLS | -94.16% | -11.76 | 0.619 | 1,922 |
| Momentum 1h | -99.11% | -23.23 | 0.420 | 2,507 |

The volatility adjustment changes prediction magnitude but not sign, so its sign strategy is
identical to unadjusted 24-hour momentum. Every active baseline lost money after costs. These
trading results are downstream diagnostics, not evidence used to rank prediction models.

## Interpretation

Phase 12 rejects the explanation that the observed RFF recovery is valuable merely because it
beats ordinary linear, momentum, or constant models. It does not. The strongest forecast in this
development sample is to predict no return at all.

The result is compatible with a real mathematical double-descent curve in the fitted
representation, but not with useful benign overfitting in financial return prediction. Positive IC
or slightly above-50% directional accuracy is insufficient when MSE, calibration, turnover, and
fees all deteriorate. The final holdout should remain unopened; the current evidence does not
justify promoting any model to it or to live capital.

## Artifacts

- `user_data/research_results/double_descent/phase12/summary.json`
- `user_data/research_results/double_descent/phase12/baseline_results.csv`
- `user_data/research_results/double_descent/phase12/simple_baselines.html`
- `user_data/research_results/double_descent/phase12/checkpoint.json`
