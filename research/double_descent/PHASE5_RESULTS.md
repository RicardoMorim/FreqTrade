# Phase 5 Results: Multiple RFF Seeds

## Decision

Phase 5 passed its complete implementation and data-integrity gate. The interpolation spike and
second-descent **shape is robust to the RFF projection seed**, but the experiment provides **no
evidence of benign overfitting, useful return prediction, or economic value**.

The distinction is important. All five seeds peak at `P/N=1`, all begin strict interpolation at
actual `P/N=1.0199`, and all recover more than 99.90% of the enormous threshold deterioration by
`P/N=50`. Nevertheless, the largest model remains 13.80 times worse than the zero-return forecast
and 10.31 times worse than the best underparameterized RFF model on mean OOS MSE. No tested RFF
model beats the zero forecast in any seed.

The Phase 5 result is therefore:

> A repeatable interpolation singularity followed by numerical/statistical recovery, not useful
> benign overfitting in financial return prediction.

## Frozen protocol

The experiment changes only the RFF seed from Phase 4:

- market: Binance BTC/USDT perpetual futures;
- timeframe: 1 hour;
- development period: `2025-01-01` through `2025-12-31`;
- untouched holdout boundary: `2026-01-01`;
- rolling training/backtest windows: 90/30 days;
- effective training size: `N=2,159`;
- P/N grid: 19 points from `0.10` to `50`;
- RFF gamma: `0.2`;
- Ridge lambda: `0`;
- numerical precision: CUDA `float64`;
- transaction fee: `0.1%` per side;
- OOS observations per model: `8,759`;
- training windows per grid point: 13.

Five seeds were fixed before inspecting their results:

| Role | Seed |
| --- | ---: |
| Phase 4 audited reference | 20,260,810 |
| SeedSequence child 1 | 1,898,170,439 |
| SeedSequence child 2 | 3,432,960,257 |
| SeedSequence child 3 | 2,841,638,297 |
| SeedSequence child 4 | 2,713,191,886 |

The four new seeds are deterministic children of `numpy.random.SeedSequence(20260810)`. The Phase 4
result was reused only after matching every frozen experimental field and verifying its passed gate.
This produced 95 matched model cases: 19 ratios times five seeds. Every reported 95% interval is a
Student-t interval across the five RFF seeds. It measures projection-seed dispersion, **not** market
sampling uncertainty.

## Integrity gate

All predeclared checks passed:

| Check | Result |
| --- | :---: |
| Five unique predeclared seeds completed | Pass |
| Every underlying Phase 4 gate passed | Pass |
| Complete matched 19-point grid for every seed | Pass |
| Same zero-return baseline across all cases | Pass |
| Same OOS observation count across all cases | Pass |
| All aggregate statistics finite | Pass |
| Every aggregate uses all five seeds | Pass |
| Complete metric-by-ratio aggregate grid | Pass |
| 2026 holdout unused | Pass |

## Prediction results

Selected five-seed means are shown below. Parentheses contain the 95% seed interval for MSE.

| P/N | Mean OOS MSE (95% seed interval) | Mean OOS R2 | Mean IC | Directional accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 0.10 | 0.00003010 (0.00002946, 0.00003074) | -0.339 | 0.0089 | 50.79% |
| 0.50 | 0.00029893 (0.00027232, 0.00032555) | -12.295 | -0.0027 | — |
| 0.90 | 0.00507203 (0.00452282, 0.00562125) | -224.57 | 0.0021 | — |
| 0.98 | 0.02535095 (0.0225484, 0.0281535) | -1,126.5 | 0.0048 | — |
| 1.00 | 0.43749191 (0.233411, 0.641572) | -19,456.1 | 0.0073 | 50.37% |
| 1.02 | 0.02506066 (0.0215470, 0.0285743) | -1,113.6 | -0.0061 | 50.30% |
| 2.00 | 0.00094220 (0.00084055, 0.00104386) | -40.904 | -0.0111 | — |
| 5.00 | 0.00046817 (0.00043946, 0.00049688) | -19.821 | -0.0067 | — |
| 10.00 | 0.00037806 (0.00037200, 0.00038411) | -15.814 | -0.0040 | — |
| 25.00 | 0.00032347 (0.00031794, 0.00032900) | -13.386 | -0.0072 | — |
| 50.00 | 0.00031032 (0.00030388, 0.00031677) | -12.801 | -0.0065 | 51.17% |

The constant zero-return forecast has MSE `0.0000224849`. The best mean RFF point is the smallest
model, `P/N=0.10`, but even it has 1.339 times the zero-forecast MSE. Across seeds:

- threshold peak at `P/N=1`: 5/5;
- double-descent shape detected: 5/5;
- largest model beats the best underparameterized model: 0/5;
- largest model beats zero: 0/5;
- any RFF model anywhere on the grid beats zero: 0/5.

The mean peak MSE is 14,535 times the best underparameterized mean MSE. Because the denominator of
the recovery statistic is this extreme spike, a 99.929% recovery still leaves the final model much
worse than both meaningful baselines. Recovery percentage must not be interpreted as forecast
quality.

At `P/N=50`, mean IC is `-0.00648`, with a seed interval of `[-0.01221, -0.00076]`. This says the
negative association is stable to these RFF projections; it is not a statistical confidence
interval over independent market histories. Together with mean OOS R2 of `-12.80`, it is evidence
against treating the forecasts as useful return estimates.

## Interpolation and effective dimension

| P/N | Mean train MSE | Mean maximum condition number | Mean effective rank |
| ---: | ---: | ---: | ---: |
| 0.10 | 2.02e-5 | 65.7 | 21.47 |
| 1.00 | 8.87e-10 | 1.32e6 | 23.71 |
| 1.02 | 3.46e-20 | 1.23e5 | 23.72 |
| 50.00 | 1.46e-26 | 5.02e3 | 23.72 |

Strict interpolation begins at `P/N=1.0199` in every seed. The worst conditioning and OOS error
occur together at `P/N=1`; both decline after crossing the threshold. This consistency makes a
random favorable seed an implausible explanation for the curve shape.

However, nominal dimension grows to 107,950 features while mean spectral effective rank remains
near 24. The experiment is massively overparameterized algebraically and interpolates the training
sample, but the RBF representation has a very low effective dimension on these market states. This
is a material caveat for any claim about one million statistically independent predictors and a
direct motivation for the later effective-dimension analysis.

## Economic results

The sign strategy remains deliberately simple and pays the frozen 0.1% fee on each side.

| P/N | Mean trades | Mean turnover | Mean net return | Mean Sharpe | Mean profit factor | Mean max drawdown |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | 1,851 | 1,050x | -96.23% | -11.81 | 0.571 | 96.33% |
| 1.00 | 1,814 | 971x | -96.49% | -11.22 | 0.579 | 96.55% |
| 1.02 | 1,806 | 1,060x | -96.64% | -10.69 | 0.598 | 96.68% |
| 50.00 | 1,581 | 1,025x | -93.53% | -9.66 | 0.592 | 93.61% |

At `P/N=50`, every seed loses between 92.24% and 95.19%, every Sharpe is between -10.18 and -9.14,
and every profit factor is below 0.616. Zero of five extreme models pass even the preliminary
economic gate. The slight relative improvement after the interpolation threshold is not remotely
tradable and does not rescue the predictive failure.

## Scientific interpretation

Phase 5 rules out the simple explanation that Phase 4's curve was produced by one unusually bad or
favorable RFF projection. It strengthens evidence for a reproducible double-descent **geometry** in
this exact ridgeless RBF-feature system.

It simultaneously strengthens the negative result for financial benign overfitting:

1. the second descent does not return to the performance of small models;
2. no model beats a zero-return forecast;
3. all OOS R2 values remain negative;
4. the extreme-model IC is slightly negative across seeds;
5. costed trading fails in every seed;
6. nominal P greatly overstates spectral effective dimension.

The correct current classification is closest to the research plan's **Result E — no useful double
descent**, with a refinement: a robust interpolation peak and second-descent shape exists, but it
does not constitute improved generalization or benign overfitting relative to valid baselines.

This is still an initial robustness experiment. Five seeds are enough to reject a single-seed
artifact but not to support publication-level uncertainty claims; the final version should use the
predeclared 10–20 seeds. The 2026 holdout remains untouched. The next planned experiment is the
kernel-limit comparison, which can test whether the poor extreme-RFF limit is coherent with the
corresponding exact RBF kernel.

## Reproduction

```powershell
C:\Users\Ricar\Desktop\projects\freqtrade\.venv\Scripts\python.exe `
  scripts\run_double_descent_phase5.py `
  --data-dir user_data\data\binance `
  --cuda-python C:\Users\Ricar\AppData\Local\Python\pythoncore-3.14-64\python.exe
```

Generated evidence is under `user_data/research_results/double_descent/phase5/` and remains ignored
by Git. The key artifacts are `summary.json`, `seed_results.csv`, `aggregate_metrics.csv`, and
`multi_seed_double_descent.html`.
