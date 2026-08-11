# Phase 6 Results: Exact RBF-Kernel Limit

## Decision

Phase 6 passed every implementation and data-integrity gate. The finite Random Fourier Feature
models converge cleanly to the exact centered RBF kernel, and one million features satisfy the
predeclared practical-convergence criterion.

That numerical success does **not** support useful benign overfitting. The exact kernel has OOS MSE
`0.000292118`, which is 12.99 times worse than the zero-return forecast and 9.71 times worse than
the best underparameterized RFF model. Its OOS R2 is `-11.99`, IC is `-0.0093`, and the costed sign
strategy loses 93.44%.

The Phase 6 result is therefore:

> The observed second descent is a genuine and reproducible approach toward the corresponding RBF
> kernel limit, but that limit is a bad return predictor. Numerical convergence is not financial
> benign overfitting.

## Frozen protocol

No predictive or trading parameter was selected in this phase. The experiment retains:

- Binance BTC/USDT perpetual futures, 1-hour candles;
- development period `2025-01-01` through `2025-12-31`;
- untouched holdout boundary `2026-01-01`;
- 90-day rolling training and 30-day backtest windows;
- effective `N=2,159` and the frozen 25-variable market state;
- gamma `0.2`, Ridge lambda `0`, `rcond=1e-12`;
- CUDA `float64` and the same centered minimum-norm solver convention;
- fee `0.1%` per side and the unchanged sign strategy;
- the five predeclared Phase 5 RFF seeds.

The exact kernel is:

```text
K(x, x') = exp(-0.2 * ||x - x'||^2)
```

It is centered from each rolling training window before the ridgeless solve, exactly matching the
limit of the centered RFF representation. The eigensolver applies the same relative numerical
floor as the finite-feature CUDA solver, so the comparison is between like-for-like numerical
estimators rather than different pseudoinverse conventions.

The convergence grid is:

| P | P/N | Source |
| ---: | ---: | --- |
| 10,795 | 5.00 | Phase 5 |
| 53,975 | 25.00 | Phase 5 |
| 107,950 | 50.00 | Phase 5 |
| 250,000 | 115.79 | New Phase 6 run |
| 500,000 | 231.59 | New Phase 6 run |
| 1,000,000 | 463.18 | New Phase 6 run |
| infinity | infinity | Exact centered RBF kernel |

Phase 6 executed 15 new large-RFF cases plus one exact-kernel case. With 13 rolling windows per
case, this is 208 new model fits. It then compared 30 matched RFF/kernel prediction vectors: six
finite dimensions times five seeds.

There are 8,760 aligned hourly predictions in each vector. Prediction metrics use 8,759 rows because
the last candle correctly has no realized one-hour-forward return.

## Predeclared convergence criterion

At one million features, both conditions had to hold:

1. mean relative L2 prediction error versus the exact kernel at most 10%;
2. relative OOS-MSE difference versus the exact kernel at most 10%.

This criterion was fixed before executing the exact kernel and large-RFF grid. Passing it establishes
practical numerical convergence only. Useful benign overfitting additionally requires the kernel to
beat the zero-return forecast and the best underparameterized model.

## Integrity gate

All checks passed:

| Check | Result |
| --- | :---: |
| Phase 5 reference passed and matched the frozen protocol | Pass |
| Exact rolling kernel case succeeded | Pass |
| All 15 large-RFF Phase 4 gates passed | Pass |
| Complete 30-case prediction-comparison grid | Pass |
| All 8,760-point prediction vectors aligned | Pass |
| All comparison metrics finite | Pass |
| Complete five-seed aggregate grid | Pass |
| CUDA `float64` kernel diagnostics recorded | Pass |
| 2026 holdout unused | Pass |

## Convergence results

The table reports five-seed means. Relative L2 error is measured directly between the complete RFF
and exact-kernel OOS prediction vectors.

| P | Relative prediction error | Correlation with kernel | Sign agreement | OOS MSE | RFF training time |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10,795 | 74.66% | 0.8150 | 82.90% | 0.000468170 | 5.3 s |
| 53,975 | 31.26% | 0.9553 | 91.66% | 0.000323472 | 16.1 s |
| 107,950 | 22.03% | 0.9771 | 93.88% | 0.000310323 | 29.2 s |
| 250,000 | 14.69% | 0.9895 | 95.90% | 0.000301035 | 64.3 s |
| 500,000 | 10.42% | 0.9947 | 97.05% | 0.000296982 | 126.3 s |
| 1,000,000 | 7.33% | 0.9973 | 97.79% | 0.000294585 | 249.0 s |
| Exact kernel | 0% | 1.0000 | 100% | 0.000292118 | 2.7 s |

The convergence evidence is unusually clean:

- Spearman correlation between P and prediction error: `-1.0`;
- log-log slope of prediction error against P: `-0.509`;
- theoretical Monte Carlo RFF rate: approximately `P^-0.5`;
- one-million-feature relative prediction error: 7.33%, passing the 10% gate;
- one-million-feature OOS-MSE gap: 0.84%, passing the 10% gate;
- exact-kernel MSE lies inside the one-million-feature five-seed 95% interval.

The observed slope almost exactly matches the expected inverse-square-root Monte Carlo rate. This
is strong implementation evidence that the nested RFF generator and solver approach the intended
kernel rather than an unrelated limiting estimator.

The exact kernel requires 2.68 seconds of summed model-training time across its 13 windows, compared
with a 249.0-second mean for one million explicit RFFs: the finite approximation is 92.8 times slower
inside the solver. End-to-end Freqtrade wall time is approximately 16 seconds for the kernel versus
283 seconds per one-million-feature seed. Peak CUDA allocation remains modest: about 285 MiB for the
kernel and 341 MiB for the streamed RFF solver.

## Prediction quality at the limit

| Model | OOS MSE | Relative to zero | OOS R2 | Pearson IC | Directional accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero-return forecast | 0.000022485 | 1.00x | approximately 0 | N/A | N/A |
| Best underparameterized RFF (`P/N=0.10`) | 0.000030098 | 1.34x | -0.339 | 0.0089 | 50.79% |
| One-million RFF mean | 0.000294585 | 13.10x | -12.10 | -0.0087 | 50.98% |
| Exact RBF kernel | 0.000292118 | 12.99x | -11.99 | -0.0093 | 50.98% |

Moving from 107,950 features to the exact kernel improves mean MSE by only 5.87%. Moving from one
million features to the exact kernel improves it by less than 1%. There is no hidden improvement
beyond the tested finite grid that could plausibly bridge the order-of-magnitude gap to the simple
baselines.

The exact kernel forecast standard deviation is `0.01635`, while realized one-hour returns have
standard deviation `0.00474`. The model therefore produces forecasts roughly 3.45 times too volatile.
Its rank correlation is slightly positive (`0.0144`), but Pearson IC is negative and squared-error
calibration is extremely poor. Directional accuracy near 51% does not compensate for the magnitude
error or turnover.

## Interpolation and effective dimension

The exact kernel has:

- algebraic rank 2,158 in every window, one below N because of centering;
- mean training MSE `1.39e-26`;
- 100% interpolated windows;
- mean spectral effective rank `23.75`;
- maximum feature-space condition number `4,894`.

At one million RFFs, mean effective rank is `23.76` and mean maximum condition number is `4,901`.
The agreement with the exact kernel is another independent convergence check.

It also sharpens the effective-dimension caveat. The estimator has one million nominal coefficients
and can interpolate all training targets, but the kernel spectrum behaves like roughly 24 balanced
dimensions. Claims based only on nominal P would materially overstate the independent statistical
complexity of this representation.

## Economic results

The exact kernel produces:

| Metric | Result |
| --- | ---: |
| Trades | 1,539 |
| Turnover | 1,015.6x |
| Net return | -93.44% |
| Sharpe | -9.84 |
| Sortino | -9.96 |
| Profit factor | 0.583 |
| Win rate | 41.91% |
| Maximum drawdown | 93.44% |

The one-million-feature mean is similarly unusable: return `-94.21%`, Sharpe `-9.69`, and profit
factor `0.582`. The economic results converge to the same failed neighborhood as the prediction
metrics. Costs are not hiding a promising forecast model; the model already fails before trading.

## Scientific interpretation

Phase 6 answers two different questions:

1. **Are the RFF experiments approaching the kernel they claim to approximate?** Yes. Direct
   predictions converge monotonically at almost exactly the theoretical `P^-1/2` rate.
2. **Is that kernel limit a useful form of benign overfitting for returns?** No. It interpolates,
   but generalizes far worse than a zero forecast and a small RFF model.

This rules out the possibility that Phase 5 stopped too early and that much larger P would reveal a
profitable second descent. The asymptote is now observed directly and is poor. The robust phenomenon
is best described as **double descent toward a bad kernel limit**.

This conclusion remains development-only. It applies to BTC, 1-hour data, the 2025 period, gamma
`0.2`, and the frozen market-state representation. It does not use the 2026 holdout and is not a
publication-level claim about all financial markets.

The next planned phase is the regularization map over `(P/N, lambda)`. Its purpose should be to test
whether Ridge suppresses the interpolation catastrophe and improves absolute OOS performance, not
to search for a profitable configuration on the final holdout.

## Reproduction

```powershell
C:\Users\Ricar\Desktop\projects\freqtrade\.venv\Scripts\python.exe `
  scripts\run_double_descent_phase6.py `
  --data-dir user_data\data\binance `
  --cuda-python C:\Users\Ricar\AppData\Local\Python\pythoncore-3.14-64\python.exe
```

Generated evidence is under `user_data/research_results/double_descent/phase6/` and remains ignored
by Git. The main artifacts are `summary.json`, `kernel_result.json`,
`prediction_convergence_by_seed.csv`, `prediction_convergence_aggregate.csv`, and
`kernel_limit_convergence.html`.
