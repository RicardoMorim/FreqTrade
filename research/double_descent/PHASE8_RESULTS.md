# Phase 8 results: training-window sensitivity

## Outcome

Phase 8 passed every integrity gate and produced a clear but negative scientific result.

The double-descent **shape** persisted as the rolling training history increased from 30 to 365
days. Every reference-seed curve had its global OOS-MSE maximum in the frozen near-threshold band
and recovered by at least 98.51% at `P/N=5`. The same pattern appeared in all three seeds tested at
both endpoint windows. This rejects the narrow explanation that the observed shape exists only
because the original 90-day sample was small.

It is not evidence of useful benign overfitting. No one of the 61 models beat the zero-return MSE
baseline. No `P/N=5` model beat the best underparameterized model. Every costed sign strategy had
negative return, negative Sharpe, and profit factor below one. Longer windows made the
post-threshold model worse relative to zero, not better.

The appropriate interpretation is therefore:

> A robust interpolation-associated variance curve exists in this RFF return-prediction system,
> but the second descent recovers toward a bad predictor rather than uncovering financial alpha.

## Frozen design

Only the rolling training history and its Phase 3-measured N changed:

| Training history | Effective N | Largest P at P/N=5 |
| ---: | ---: | ---: |
| 30 days | 719 | 3,595 |
| 60 days | 1,439 | 7,195 |
| 90 days | 2,159 | 10,795 |
| 180 days | 4,319 | 21,595 |
| 365 days | 8,759 | 43,795 |

The frozen controls were BTC/USDT perpetual futures, 1h candles, the 2025 chronological OOS
period, the same 25 causal inputs, nested RFF, seed `20260810`, gamma `0.2`, ridge `0`, float64,
the same sign strategy, and a 0.1% fee per side. The 2026 holdout was not accessed.

The reference seed used the grid:

```text
P/N = 0.10, 0.50, 0.90, 0.98, 1.00, 1.02, 1.10, 2, 5
```

The 30- and 365-day endpoints repeated `0.10, 1.00, 1.02, 5` with seeds `20260810`,
`1898170439`, and `3432960257`. The shape rule required a global MSE peak in
`0.90 <= P/N <= 1.10` and at least 50% recovery by `P/N=5`. Useful benign overfitting additionally
required the largest model to beat both zero and the best underparameterized model.

## Main window results

All rows below use the reference seed. Ratios against zero greater than one are worse than the
zero-return forecast.

| Days | Peak P/N | Peak MSE / zero | Recovery at P/N=5 | P/N=5 MSE / zero | P/N=5 MSE / best small |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | 1.00 | 61,013x | 99.982% | 12.26x | 9.70x |
| 60 | 1.00 | 52,371x | 99.971% | 16.30x | 12.38x |
| 90 | 1.00 | 21,317x | 99.906% | 21.42x | 16.07x |
| 180 | 1.00 | 7,272x | 99.679% | 24.76x | 17.70x |
| 365 | 0.98 | 2,469x | 98.515% | 38.00x | 27.98x |

Two effects coexist:

1. the catastrophic peak becomes smaller in absolute and relative terms as N grows;
2. the `P/N=5` predictor deteriorates monotonically relative to zero as the window grows.

The recovery fraction was monotonically decreasing across the five predeclared windows (Spearman
`rho=-1`). With only five ordered window choices, the associated unadjusted p-value is descriptive
and is not treated as confirmatory evidence.

## Endpoint seed robustness

The shape rule passed in 3/3 seeds at 30 days and 3/3 seeds at 365 days.

| Endpoint | Recovery range across seeds | Mean P/N=5 MSE | 95% t interval |
| ---: | ---: | ---: | ---: |
| 30 days | 99.806% to 99.982% | 0.0002739 | [0.0002370, 0.0003108] |
| 365 days | 97.869% to 98.269% | 0.0008323 | [0.0006838, 0.0009809] |

All six endpoint seed curves failed the useful-benign-overfitting test: `P/N=5` was worse than
zero and worse than the underparameterized control in every seed.

## Prediction and trading evidence

The best MSE among all 61 cells was the 30-day, `P/N=0.1` reference-seed model:

- OOS MSE `2.8403e-05`, versus zero `2.2485e-05` (1.263x worse);
- OOS R2 `-0.2632`;
- IC `0.0286`;
- directional accuracy `51.31%`;
- net return `-90.80%`, Sharpe `-9.57`, profit factor `0.659`;
- 1,665 trades and turnover of 1,277.5 times starting capital.

The largest observed IC (`0.0536`) occurred at 365 days and `P/N=1.02`, but its MSE was roughly
1,721 times the zero baseline and its costed strategy lost 97.93%. A correlation headline near the
interpolation singularity is therefore not evidence of a usable return forecast.

Across all 61 cells:

- no model beat zero-return MSE;
- every OOS R2 was negative;
- every trading return and Sharpe was negative;
- every profit factor was below one.

The trading results are downstream diagnostics only. They are not used to establish double
descent, and this experiment gives no basis for live or dry-run promotion.

## Numerical and effective-dimension warning

The near-threshold OOS peak stayed close to nominal `P/N=1`, but strict all-window training
interpolation did not always begin at exactly one:

| Training history | First P/N with all 13 training MSE values <= 1e-16 |
| ---: | ---: |
| 30 days | 1.00 |
| 60 days | 1.02 |
| 90 days | 1.02 |
| 180 days | 1.02 |
| 365 days | 1.10 |

At 365 days the median numerical rank at `P/N=1.02` was 8,721 rather than the centered maximum
8,758, while the entropy-based effective rank was only about 17.3. The maximum recorded condition
number near the threshold was about `7.17e5`. These float64 diagnostics passed, but the drift between
nominal P, numerical rank, strict interpolation, and effective dimension is scientifically
material. Phase 20's effective-dimension analysis is necessary before calling the threshold a
pure `P/N=1` phenomenon.

## Integrity and compute

All automated checks passed:

- 61/61 predeclared cases and 793/793 rolling fits completed;
- all rolling FreqAI substudy gates passed;
- feature counts matched the exact Phase 3 N for every window;
- every case used the same 8,759 OOS observations and zero baseline;
- the nine overlapping 90-day cases reproduced Phase 5 with maximum relative error `2.52e-13`;
- CUDA float64 diagnostics were present for every case;
- the 2026 holdout remained untouched.

The run used 4,203.6 summed Freqtrade wall-seconds and 3,033.6 CUDA training-seconds. Peak tracked
VRAM was 4,063.9 MiB, well inside the RTX 5080 Laptop GPU's 16 GB capacity.

## Limitations and decision

Increasing the training history changes N and also changes the recency and regime composition of
the training data. This is a window-sensitivity experiment, not a pure causal intervention on N.
It covers one asset, one OOS year, and three seeds only at the endpoint critical points. The main
grid stops at `P/N=5` for the 365-day computation, although Phase 6 independently showed that much
larger finite RFF models converge toward an exact kernel limit that is also predictively poor.

Decision: retain the robust double-descent shape as a research finding, reject useful benign
overfitting and economic value, keep the final holdout sealed, and do not deploy any Phase 8 model.

## Reproduction

```powershell
python scripts/run_double_descent_phase8.py `
  --data-dir user_data\data\binance
```

Generated evidence is under `user_data/research_results/double_descent/phase8/` and remains ignored
by Git. The main artifacts are `summary.json`, `window_map.csv`, `endpoint_robustness.csv`, and
`training_window_sensitivity.html`.
