# Phase 7 Results: Explicit Ridge Regularization

## Decision

Phase 7 passed every implementation, data-integrity, and solver-mechanism gate. Explicit Ridge
regularization removes the interpolation catastrophe and flattens the double-descent curve. It does
not uncover useful financial predictability.

At `P/N=1`, the predeclared `lambda=1` control reduces OOS MSE by 99.995% relative to the ridgeless
model in all three tested RFF seeds. However, the best reference-map cell remains 0.096% worse than
the zero-return forecast. Across the three-seed robustness grid, the best cell remains 0.082% worse
than zero and zero of three seeds beats zero individually.

The Phase 7 result is therefore:

> Ridge cures the interpolation singularity by shrinking unstable directions toward a nearly
> constant forecast. The apparent improvement is variance suppression, not recovered alpha or
> benign overfitting.

## Frozen protocol

The experiment retains the Phase 4-6 market and data protocol:

- Binance BTC/USDT perpetual futures, 1-hour candles;
- development period `2025-01-01` through `2025-12-31`;
- untouched holdout boundary `2026-01-01`;
- 90-day rolling training and 30-day prediction windows;
- effective `N=2,159` and the frozen 25-variable market state;
- gamma `0.2`, `rcond=1e-12`, CUDA `float64`;
- 0.1% fee per side and the unchanged sign strategy.

Ridge solves the centered system

```text
(K + lambda I) alpha = y
```

while leaving the rolling-window intercept unregularized. The predeclared lambda grid was:

```text
0, 1e-8, 1e-6, 1e-4, 1e-2, 1, 100
```

The audited reference seed `20260810` ran all 19 P/N points. Seeds `1898170439` and `3432960257`
repeated the critical ratios `0.10`, `1.00`, `1.02`, and `50`. This produced:

- 133 reference-map cells;
- 84 three-seed critical-grid cells, including 28 cells shared with the reference map;
- 189 unique cases and 2,457 genuine rolling model fits.

The grid was fixed before inspecting its outcomes. No lambda, P/N, threshold, or strategy rule was
selected using the final holdout.

## Integrity and interruption recovery

All ten global checks passed:

| Check | Result |
| --- | :---: |
| All 21 underlying Phase 4 substudies passed | Pass |
| Complete predeclared 189-case map | Pass |
| Complete three-seed critical-grid aggregates | Pass |
| Same 8,759 OOS observations in every case | Pass |
| Same zero-return baseline in every case | Pass |
| All reported metrics finite | Pass |
| `lambda=0` reproduced Phase 5 | Pass |
| Ridge diagnostics obeyed theoretical monotonicity | Pass |
| CUDA `float64` diagnostics recorded | Pass |
| 2026 holdout unused | Pass |

The ridgeless repetition matched Phase 5 across all 27 repeated cells. Maximum relative OOS-MSE
difference was `2.4e-13`; prediction dispersion matched within `3.6e-14`, and costed returns matched
exactly.

A Windows blue screen interrupted the final seed. Checkpoints preserved 186 completed cases. One
partially written metadata file contained null bytes; recovery detected it as invalid, discarded
only that generated case directory, and recomputed it plus the final three cases. The completed map
then passed every gate. Recovery handling was hardened so future corrupt backtest metadata triggers
a scoped case rerun rather than aborting the research.

## Reference-seed regularization map

The interpolation peak is the maximum MSE inside `0.9 <= P/N <= 1.1`. Dynamic range is the ratio
between the worst and best MSE along the complete P/N curve at a fixed lambda.

| Lambda | Threshold-peak MSE | Peak removed | Best MSE / zero | Best P/N | Curve dynamic range |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.479301 | 0% | 1.33277x | 0.10 | 15,994x |
| 1e-8 | 0.0360003 | 92.49% | 1.33277x | 0.10 | 1,201x |
| 1e-6 | 0.00401439 | 99.16% | 1.33277x | 0.10 | 134.0x |
| 1e-4 | 0.000465921 | 99.90% | 1.33248x | 0.10 | 15.55x |
| 1e-2 | 0.0000624831 | 99.987% | 1.30692x | 0.10 | 2.161x |
| 1 | 0.0000236892 | 99.9951% | 1.04987x | 10 | 1.0060x |
| 100 | 0.0000225089 | 99.9953% | 1.00096x | 2 | 1.0003x |

Ridge progressively removes the non-monotonic P/N geometry. At `lambda=1`, changing P/N across the
entire 0.10-to-50 range changes MSE by only 0.60%. At `lambda=100`, it changes MSE by only 0.032%.
The apparent best P/N moves from 10 to 2, but the differences are economically and statistically
negligible. Strong regularization makes nominal feature count almost irrelevant.

No one of the 133 predeclared reference-map cells beats the zero-return MSE.

## Mechanism at the interpolation threshold

The reference seed at `P/N=1` shows exactly how Ridge removes the peak:

| Lambda | OOS MSE / zero | Prediction std | Train MSE | Effective df | Coefficient norm | Ridge-system condition |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 21,316.5x | 0.6920 | 6.11e-10 | 2,157.5 | 6,534.8 | 1.08e12 |
| 1 | 1.05350x | 0.001194 | 1.97e-5 | 171.3 | 1.488 | 156.9 |
| 100 | 1.00106x | 0.000169 | 2.37e-5 | 10.5 | 0.0388 | 2.56 |

Across every one of the 27 matched seed/P/N paths:

- training MSE is non-decreasing with lambda;
- effective degrees of freedom are non-increasing;
- normalized coefficient norm is non-increasing;
- regularized condition number is non-increasing.

This rules out a bookkeeping explanation for the smoother curve. The ridgeless model fits almost
perfectly by amplifying weak spectral directions, producing a forecast standard deviation 146
times larger than realized one-hour volatility. `lambda=100` reduces the effective degrees of
freedom to roughly 10 and the forecast standard deviation to only 3.6% of realized volatility.

## Three-seed prediction result

The best cell inside the predeclared three-seed critical grid is `lambda=100`, `P/N=0.10`:

| Metric | Three-seed result |
| --- | ---: |
| Mean OOS MSE | 0.0000225034 |
| MSE relative to zero | 1.000821x |
| Seeds beating zero | 0 / 3 |
| Mean OOS R2 | -0.000821 |
| Mean Pearson IC | 0.00810 |
| IC 95% seed interval | [-0.00544, 0.02164] |
| Mean directional accuracy | 50.96% |
| Mean prediction standard deviation | 0.000174 |
| Mean effective degrees of freedom | 10.12 |

The MSE interval slightly overlaps the zero baseline because three seeds provide a very imprecise
t interval, but every individual seed is worse than zero. The IC interval crosses zero. These
intervals measure only random-feature projection dispersion, not independent market uncertainty.

The reference-map minimum occurs at `lambda=100`, `P/N=2`, whereas the critical three-seed minimum
occurs at `P/N=0.10`. This is not evidence for two optima: at `lambda=100` the reference curve has a
dynamic range of only 0.032%, so tiny seed fluctuations reorder nearly identical cells.

## Economic result

The best three-seed prediction cell remains economically unusable:

| Metric | Three-seed mean |
| --- | ---: |
| Trades | 697 |
| Turnover | 898.5x |
| Net return | -66.45% |
| Sharpe | -3.24 |
| Profit factor | 0.702 |

Strong shrinkage reduces trade count and losses relative to the ridgeless strategy, but it does not
produce a tradable forecast. Tiny near-zero predictions still change sign often enough to incur
large costs. No seed passes the preliminary economic gate of positive return, Sharpe above 0.5,
profit factor above 1, and at least 30 trades.

## Scientific interpretation

Phase 7 distinguishes three statements that could otherwise be conflated:

1. **Explicit regularization removes the interpolation catastrophe.** Yes, decisively and in all
   three seeds.
2. **Regularization improves absolute OOS prediction.** It improves the unstable RFF models, but
   only until they approach the zero-return forecast from above.
3. **Regularization reveals financial alpha.** No. No reference-map cell or robust seed beats zero,
   and the costed strategy remains strongly negative.

Together with Phase 6, the evidence supports a coherent explanation. Ridgeless RFFs amplify weak,
ill-conditioned directions near interpolation and then converge along a second descent toward a
bad RBF-kernel limit. Ridge suppresses those directions and eventually suppresses almost the entire
model. The resulting MSE improvement is benign regularization behavior, not benign overfitting and
not return predictability.

This remains development-only evidence for BTC, 1-hour data, 2025, gamma `0.2`, and three RFF seeds
at the critical points. The best cell is a descriptive result from 133 predeclared development
cells, not an unbiased estimate. It must not be promoted to the untouched holdout.

The next planned phase is the training-window experiment. It should vary only the historical window
length and build P/N relative to each newly measured effective N, while retaining explicit controls
for the shrink-to-zero explanation established here.

## Compute and reproduction

The final evidence contains 189 cases and 2,457 rolling fits. Summed Freqtrade case wall time is
approximately 80.7 minutes; summed CUDA training time is 22.0 minutes. Maximum allocated VRAM was
357 MiB on the NVIDIA GeForce RTX 5080 Laptop GPU.

```powershell
C:\Users\Ricar\Desktop\projects\freqtrade\.venv\Scripts\python.exe `
  scripts\run_double_descent_phase7.py `
  --data-dir user_data\data\binance
```

Generated evidence is under `user_data/research_results/double_descent/phase7/` and remains ignored
by Git. The main artifacts are `summary.json`, `regularization_map.csv`,
`robustness_aggregates.csv`, and `regularization_map.html`.
