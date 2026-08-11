# Phase 9 results: pre-holdout gamma calibration

## Outcome

Phase 9 completed all 84 predeclared cases and passed every integrity gate. The conservative
prediction-only rule selected and froze `gamma=0.5` for subsequent development experiments.

Relative to the existing `gamma=0.2` baseline, `gamma=0.5` lowered the predeclared geometric-mean
OOS-MSE score by 48.04% and won all nine matched seed/P/N comparisons. It also retained the
double-descent shape in all three seeds. This is a valid kernel-scale calibration result, not
evidence of useful predictability: none of the nine selection cells beat the zero-return forecast.

Across the complete 84-case map:

- zero models beat zero-return MSE or produced positive OOS R2;
- zero costed sign strategies had positive return, Sharpe, or profit factor above one;
- the best overall model was still 20.32% worse than predicting zero return;
- every gamma displayed a large interpolation peak followed by a second descent in all three
  seeds, but the recovered predictors remained worse than both zero and their small-model control.

The appropriate interpretation is:

> A broader RBF kernel reduces the variance of the finite-RFF predictor in this calibration period,
> but gamma tuning does not convert double descent into benign overfitting or financial alpha.

## Frozen design

Gamma was calibrated strictly before the previously inspected 2025 experiment and the sealed 2026
holdout:

```text
calibration:     2024-07-01 through 2025-01-01
main experiment: begins 2025-01-01 and was excluded
final holdout:   begins 2026-01-01 and remained sealed
```

The candidate grid was `gamma = 0.05, 0.1, 0.2, 0.5`. Seed `20260810` used the 11-point map
`P/N = 0.10, 0.50, 0.90, 0.98, 1.00, 1.02, 1.10, 2, 5, 10, 50`; seeds `1898170439` and
`3432960257` repeated `0.10, 1.00, 1.02, 5, 50`. All other controls stayed fixed: BTC perpetual
futures, 1h candles, 90-day rolling training windows, 30-day evaluation steps, effective
`N=2,159`, 25 causal inputs, nested RFF, ridge zero, float64 CUDA, and a 0.1% fee per side.

Changing gamma rescales the same seeded Gaussian projections. This preserves paired random draws
across candidates instead of comparing unrelated random-feature spaces.

## Selection result

The frozen score was the geometric mean of OOS MSE divided by zero-return MSE at `P/N=0.10, 5,
50` across all three seeds. Trading metrics could not affect selection.

| Gamma | Geometric MSE / zero | Median MSE / zero | Wins vs 0.2 | Shape seeds | Selection |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.05 | 18.9748x | 40.8122x | 0/9 | 3/3 | reject |
| 0.10 | 11.1579x | 20.0251x | 0/9 | 3/3 | reject |
| 0.20 | 6.2555x | 10.9667x | baseline | 3/3 | replaced |
| 0.50 | **3.2506x** | **4.3892x** | **9/9** | 3/3 | **freeze** |

The challenger rule required at least a 5% score improvement and six of nine matched wins.
`gamma=0.5` delivered a 48.04% improvement and nine wins, so both predeclared hurdles passed.
Because the chosen value is the upper edge of the small candidate grid, it should be described as
the best tested scale rather than a globally optimal gamma.

The score improved monotonically over the four tested values. Even so, its best value remained
3.25 times the zero baseline on a geometric-mean basis, and all nine selected cells individually
lost to zero. Gamma is therefore frozen for comparability, not promoted as an alpha discovery.

## Prediction curve at the selected gamma

The full reference-seed curve shows the interpolation catastrophe and subsequent recovery clearly:

| P/N | OOS MSE / zero | OOS R2 | IC | Training MSE |
| ---: | ---: | ---: | ---: | ---: |
| 0.10 | 1.203x | -0.204 | 0.0165 | 2.500e-5 |
| 0.50 | 6.258x | -5.260 | 0.0249 | 8.980e-6 |
| 0.90 | 68.10x | -67.12 | -0.0486 | 9.250e-7 |
| 0.98 | 346.93x | -346.05 | -0.0181 | 1.386e-7 |
| 1.00 | 11,063.65x | -11,066.52 | -0.0321 | 2.640e-11 |
| 1.02 | 324.53x | -323.65 | -0.0127 | 4.921e-24 |
| 1.10 | 87.94x | -86.97 | -0.0246 | 1.632e-25 |
| 2.00 | 12.83x | -11.83 | -0.0029 | 4.963e-28 |
| 5.00 | 6.432x | -5.434 | -0.0049 | 1.873e-28 |
| 10.00 | 5.289x | -4.291 | -0.0024 | 1.565e-28 |
| 50.00 | 4.364x | -3.365 | 0.0133 | 1.079e-28 |

The global peak occurred at `P/N=1` and the `P/N=50` model recovered 99.971% of the deterioration
from the best underparameterized point. This is a strong double-descent *shape*. It is not benign
overfitting: the recovered model was 4.364 times worse than zero and 3.627 times worse than the
best underparameterized model.

## Three-seed robustness

The same qualitative result held in every seed:

| Gamma | Seeds with shape | Mean MSE / zero at P/N=0.1 | at P/N=5 | at P/N=50 |
| ---: | ---: | ---: | ---: | ---: |
| 0.05 | 3/3 | 1.673x | 96.42x | 43.76x |
| 0.10 | 3/3 | 1.519x | 44.04x | 20.90x |
| 0.20 | 3/3 | 1.344x | 16.79x | 10.87x |
| 0.50 | 3/3 | **1.215x** | **6.411x** | **4.410x** |

At `gamma=0.5`, the `P/N=50` MSE ratio ranged only from 4.364x to 4.477x across seeds. Thus the
relative advantage over smaller gamma values is not driven by one favorable projection, while the
failure to beat zero is equally consistent.

## Economic diagnostics

Trading was deliberately excluded from gamma selection. Its independent diagnostic result is
uniformly negative:

- 0/84 cases had positive net return;
- 0/84 had positive Sharpe;
- 0/84 had profit factor above one;
- the least bad strategy still lost 57.52%, with Sharpe `-5.80` and profit factor `0.801`;
- the best-MSE model lost 84.34%, with Sharpe `-12.92`, profit factor `0.651`, and 1,067 trades.

These outcomes include the fixed 0.1% fee per side. They provide no basis for live or dry-run
promotion and reinforce why economic output must not be used as proof of prediction quality.

## Numerical and effective-dimension findings

Larger gamma makes the random Fourier representation less redundant on these standardized market
states. At `P/N=50`, the mean entropy effective rank increased from about 10.30 at `gamma=0.05` to
65.06 at `gamma=0.5`. For the selected reference curve, effective rank was still only about 65 at
107,950 nominal features. Nominal P therefore remains a very poor description of statistical
dimension.

The maximum selected-curve condition number peaked near interpolation at roughly `4.34e5` and then
fell to about `1.09e3` at `P/N=50`. Float64 training and the stored rank/conditioning diagnostics
separate this numerical spike from the persistent post-threshold prediction failure.

The strict training-interpolation onset for `gamma=0.5` was `P/N=1.00` in two seeds and `1.02` in
the reference seed. For `gamma=0.05`, it was delayed to `P/N=5`, despite the OOS peak already
occurring at one. This is further evidence that nominal P/N, numerical interpolation, and effective
dimension should not be treated as interchangeable.

## Integrity and compute

All automated checks passed:

- 84/84 cases, 12/12 substudies, and 588/588 rolling model fits completed;
- all rolling FreqAI gates passed and every metric was finite;
- every case used the same 4,415 chronological OOS observations and zero baseline;
- feature counts matched the Phase 3 effective `N=2,159`;
- the four gamma candidates used paired seeds and projection draws;
- CUDA float64 diagnostics were present for every case;
- selection used prediction error only;
- the 2025 main experiment and 2026 holdout were excluded.

The run consumed 1,297.2 summed Freqtrade wall-seconds and 335.8 summed CUDA training-seconds. Peak
tracked VRAM was only 340.7 MiB, comfortably below the RTX 5080 Laptop GPU capacity.

## Limitations and decision

This calibration covers one asset, one six-month development interval, four gamma candidates, and
three seeds only at the robust points. The selected value lies on the upper grid boundary, so the
experiment identifies the best tested scale, not the global optimum. Four candidates were compared;
the 48.04% improvement and 9/9 matched wins are selection criteria, not a multiple-testing-adjusted
claim of statistical significance.

Decision: freeze `gamma=0.5` for subsequent development experiments, preserve `gamma=0.2` as the
historical baseline, reject any predictive or economic-alpha claim from Phase 9, keep the final
holdout sealed, and do not deploy any Phase 9 model.

## Reproduction

```powershell
python scripts/run_double_descent_phase9.py `
  --data-dir user_data\data\binance
```

Generated evidence is under `user_data/research_results/double_descent/phase9/` and remains ignored
by Git. The main artifacts are `summary.json`, `gamma_map.csv`, `robustness_aggregates.csv`,
`frozen_gamma.json`, and `gamma_calibration.html`.
