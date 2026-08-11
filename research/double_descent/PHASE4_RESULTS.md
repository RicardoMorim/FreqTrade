# Phase 4 Results: Financial P/N Sweep

- Run date: 2026-08-10 to 2026-08-11
- Market: Binance BTC/USDT perpetual futures
- Timeframe: 1 hour
- Development period: 2025-01-01 to 2026-01-01
- Training window: 90 days (`N=2,159`)
- Prediction window: 30 days
- RFF seed: `20260810`
- Gamma: `0.2`
- Ridge: `0`
- Precision: float64 on NVIDIA RTX 5080 Laptop GPU

## Outcome

Phase 4 passed every implementation and data-integrity gate. All 19 requested P/N points completed
13 genuine rolling FreqAI windows, produced the same 8,759 usable OOS prediction/realization pairs,
and exported a costed Freqtrade backtest. The 2026 holdout was not read to construct the final
realized return.

The single-seed curve has an unmistakable double-descent shape, but it does **not** provide
evidence of useful benign overfitting or financial predictability:

- OOS MSE peaks at exactly `P/N=1` in the aggregate and in all 13 individual windows;
- MSE recovers 99.94% of the interpolation-peak deterioration by `P/N=50`;
- the `P/N=50` model remains 10.56 times worse than the best underparameterized RFF model;
- it remains 14.08 times worse than the zero-return forecast;
- every RFF model has negative OOS R2;
- all economic backtests lose heavily after the frozen 0.1% fee per side.

This is evidence that the machinery can expose interpolation geometry in real financial feature
matrices. It is not evidence that extreme overparameterization recovers a useful return signal.

## Prediction curve

MSE is measured on next-hour close-to-close returns. The zero-return baseline MSE is
`2.2485e-05`; the one-hour momentum baseline MSE is `4.5500e-05`.

| P/N | P | Train MSE | OOS MSE | OOS R2 | IC | Direction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | 216 | 2.028e-05 | 2.997e-05 | -0.333 | 0.0138 | 50.71% |
| 0.50 | 1,080 | 7.019e-06 | 3.185e-04 | -13.16 | -0.0155 | 50.34% |
| 0.90 | 1,943 | 7.248e-07 | 4.500e-03 | -199.13 | 0.0033 | 50.52% |
| 0.98 | 2,116 | 1.176e-07 | 2.648e-02 | -1,176.59 | 0.0118 | 49.93% |
| 1.00 | 2,159 | 6.106e-10 | 4.793e-01 | -21,315.51 | 0.0027 | 51.20% |
| 1.02 | 2,202 | 6.050e-21 | 2.244e-02 | -997.05 | -0.0114 | 49.89% |
| 1.10 | 2,375 | 9.069e-23 | 5.884e-03 | -260.65 | 0.0076 | 50.33% |
| 2.00 | 4,318 | 2.678e-25 | 8.812e-04 | -38.19 | -0.0196 | 49.87% |
| 5.00 | 10,795 | 3.041e-26 | 4.817e-04 | -20.42 | -0.0045 | 50.54% |
| 10.00 | 21,590 | 1.551e-26 | 3.794e-04 | -15.88 | -0.0068 | 50.59% |
| 25.00 | 53,975 | 1.943e-26 | 3.269e-04 | -13.54 | -0.0132 | 51.12% |
| 50.00 | 107,950 | 1.523e-26 | 3.166e-04 | -13.08 | -0.0083 | 50.98% |

The smallest RFF model is the best RFF predictor by MSE, but even it is 1.33 times worse than the
zero forecast. No point on the curve has positive OOS R2. IC and directional accuracy remain near
their no-signal values.

## Interpolation and numerical diagnostics

The strict training-interpolation criterion is met in every window from `P/N=1.02` onward. At
exactly `P/N=1`, the mean training MSE is already only `6.11e-10`, but 6 of 13 windows cross the
more demanding `1e-16` threshold. The centered design has rank 2,158 once overparameterized, as
expected for 2,159 observations and an unpenalized intercept.

Conditioning deteriorates sharply near the threshold:

| P/N | Median condition number | Mean effective rank |
| ---: | ---: | ---: |
| 0.10 | 54.5 | 21.78 |
| 0.90 | 12,710 | 23.22 |
| 0.98 | 73,298 | 23.28 |
| 1.00 | 1,047,781 | 23.29 |
| 1.02 | 76,924 | 23.22 |
| 2.00 | 5,672 | 23.31 |
| 50.00 | 2,987 | 23.79 |

Nominal P therefore greatly overstates the statistical dimension of this representation. Even at
107,950 RFF, the spectral participation ratio is only about 24. The interpolation peak is real for
the fitted matrices, but it is associated with amplification of extremely weak spectral
directions. Gamma and effective-dimension experiments remain essential before interpreting the
curve economically.

The external CUDA solver was also compared with a direct NumPy minimum-norm solution on an
independent small problem. Predictions matched within the declared numerical tolerance. Phase 19
must still repeat critical real-data points with an independent solver/precision treatment.

## Economic results

Freqtrade transformed predictions into the deliberately simple sign strategy: positive forecasts
enter/retain long positions, negative forecasts enter/retain short positions, and sign changes
exit the current position. Leverage is fixed at 1x, no Hyperopt or discretionary filters are used,
and fees are 0.1% on entry and exit.

Every configuration fails economically:

- Sharpe ranges from -13.35 to -8.60;
- profit factor remains below 0.66;
- total return ranges from approximately -98.2% to -93.0%;
- each run makes 1,571 to 1,918 trades;
- turnover is approximately 821 to 1,254 times initial capital.

These results are downstream diagnostics, not the proof of the prediction curve. They show that
small, unstable forecasts plus sign-only execution generate prohibitive churn and costs.

## Scientific interpretation

The Phase 4 result is:

1. **Double-descent shape detected:** yes, for this seed, gamma, asset, timeframe, and development
   period.
2. **Second-descent recovery detected:** yes, relative to the interpolation catastrophe.
3. **Benign overfitting evidence:** no. The recovered models remain materially worse than simple
   prediction baselines.
4. **Financial predictability:** no evidence. R2 is negative and IC/direction remain near zero.
5. **Profitability:** decisively no after costs.
6. **Robust result:** not yet. Phase 5 must repeat the curve across independent RFF seeds.

The result currently fits the research project's distinction:

```text
complexity geometry != predictability != profitability
```

## Reproduction

```powershell
C:\Users\Ricar\Desktop\projects\freqtrade\.venv\Scripts\python.exe `
  scripts\run_double_descent_phase4.py `
  --data-dir user_data\data\binance `
  --cuda-python C:\Users\Ricar\AppData\Local\Python\pythoncore-3.14-64\python.exe
```

The runner is resumable. It validates existing configs, training diagnostics, prediction coverage,
and backtests before recovering a completed point, and writes `checkpoint.json` after every ratio.

Generated evidence is under `user_data/research_results/double_descent/phase4/` and remains ignored
by Git. It includes detailed JSON, a flat CSV, all frozen configs and logs, costed backtests, the
checkpoint, and an interactive curve.
