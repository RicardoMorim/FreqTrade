# Phase 11 results: shuffled-label negative control

## Outcome

Phase 11 completed all 21 predeclared cases and 273 rolling model fits. Every integrity gate
passed, all cases used the same 8,759 chronological OOS observations as Phase 10, and the 2026
holdout remained sealed.

Shuffling the training labels produced the expected negative-control result:

- 0/21 cases beat the zero-return forecast on OOS MSE;
- 0/21 produced positive OOS R2;
- maximum absolute OOS Pearson IC was only `0.0310`;
- 0/21 costed sign strategies had positive return, positive Sharpe, or profit factor above one;
- no P/N ratio beat zero in all three paired seeds;
- shuffled-label MSE was 2.394 times unshuffled MSE by geometric mean and 2.372 times by median
  across the 15 robust matched cells.

The negative control therefore passed: the framework did not manufacture useful predictability or
economic performance after the X/y relationship was deliberately destroyed.

The shuffled models still showed an interpolation peak and second descent in all three seeds. This
reinforces the Phase 10 conclusion that the curve shape is a property of ridgeless
high-dimensional fitting and is not, on its own, evidence of financial signal.

## Frozen design

Phase 11 reused the exact Phase 10 market-RFF setup: the same 25 causal market inputs, nested RFF
projections, feature seeds, `gamma=0.5`, ridge zero, float64 CUDA solver, 90-day rolling training
windows, 30-day evaluation steps, measured `N=2,159`, 2025 OOS period, and 0.1% fee per side.

Only the training labels changed. Within each rolling window they were assigned through a
timestamp-keyed deterministic cyclic derangement:

- the label multiset was preserved exactly;
- no observation retained its own label;
- row ordering could not change the timestamp-to-source-label mapping;
- the same permutation was reused across every P for a matched seed/window;
- different paired label seeds produced different permutations;
- OOS labels were never shuffled.

The reference feature/permutation seed mapped
`P/N = 0.10, 0.50, 0.90, 0.98, 1.00, 1.02, 1.10, 2, 5, 10, 50`. Two additional paired seeds
repeated `0.10, 1.00, 1.02, 5, 50`, giving 21 cases. Every case had an exact unshuffled Phase 10
counterpart with the same feature seed and P.

## Reference-seed curve

OOS MSE is reported relative to the zero-return forecast. A value below one would indicate a null
failure; none did.

| P/N | Predictors | MSE / zero | OOS R2 | Pearson IC | Prediction std |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | 216 | 1.145x | -0.145 | 0.0200 | 0.00190 |
| 0.50 | 1,080 | 6.219x | -5.219 | -0.0054 | 0.01080 |
| 0.90 | 1,943 | 181.48x | -180.48 | -0.0177 | 0.06362 |
| 0.98 | 2,116 | 1,487.90x | -1,486.90 | -0.0205 | 0.18275 |
| 1.00 | 2,159 | 88,631.83x | -88,630.86 | -0.0069 | 1.41113 |
| 1.02 | 2,202 | 1,850.47x | -1,849.47 | -0.0310 | 0.20374 |
| 1.10 | 2,375 | 379.65x | -378.65 | -0.0230 | 0.09215 |
| 2.00 | 4,318 | 40.53x | -39.53 | -0.0135 | 0.02974 |
| 5.00 | 10,795 | 18.09x | -17.09 | -0.0165 | 0.01952 |
| 10.00 | 21,590 | 13.09x | -12.09 | -0.0253 | 0.01637 |
| 50.00 | 107,950 | 9.500x | -8.500 | -0.0157 | 0.01375 |

The global peak occurred exactly at `P/N=1`. The largest model recovered 99.991% of the increase
from the best underparameterized point to the peak, so the second-descent shape is unambiguous.
However, `P/N=50` remained 9.50 times worse than zero and 8.30 times worse than the small-model
control. Recovery from an interpolation catastrophe is not the same as useful generalization.

## Three-seed robustness

The same conclusion held in all paired seeds:

| P/N | Mean MSE / zero | Range | Mean IC | IC range |
| ---: | ---: | ---: | ---: | ---: |
| 0.10 | 1.159x | 1.145-1.173x | 0.0067 | -0.0013 to 0.0200 |
| 1.00 | 64,120.77x | 38,080.95-88,631.86x | -0.0028 | -0.0069 to 0.0027 |
| 1.02 | 1,510.25x | 1,219.37-1,850.47x | -0.0022 | -0.0310 to 0.0163 |
| 5.00 | 16.63x | 15.90-18.09x | -0.0096 | -0.0165 to -0.0057 |
| 50.00 | 8.520x | 8.026-9.500x | -0.0104 | -0.0157 to -0.0038 |

Every curve peaked at one and recovered by more than 99.98%, but none of the extreme models beat
zero or its own `P/N=0.1` model. The `P/N=50` model remained 6.84-8.30 times worse than the matched
small model.

## Matched original-versus-shuffled comparison

Among the 15 robust cells, shuffled labels had lower MSE than original labels in only 3/15. All
three occurred at `P/N=0.1`, where the improvements were only 4.7-5.5% and both versions remained
worse than predicting zero. At every matched robust point at or above interpolation, original
labels had lower MSE.

The shuffled-to-original MSE ratio ranged from `1.61x` to `10.50x` at `P/N=1`, from `2.37x` to
`2.77x` at `P/N=5`, and from `1.77x` to `2.10x` at `P/N=50`. The original feature-label relation
therefore contains some structure that reduces prediction error relative to arbitrary labels, but
Phase 10 showed that the remaining error is still far above zero. This is relative structure, not
usable predictability.

## Permutation integrity

All rolling permutation checks passed:

- 273/273 target multisets were preserved exactly;
- 273/273 permutations were non-identity derangements with zero fixed points;
- permutation fingerprints were identical across P for every matched seed/window;
- mean absolute original-versus-shuffled training-label IC was `0.0198`, `0.0207`, and `0.0138`
  for the three paired seeds;
- the maximum absolute in-window label IC was `0.0613`, below the predeclared `0.10` bound;
- OOS coverage, target moments, zero baseline, dates, and market features reproduced the Phase 10
  reference within numerical tolerance.

This separates the intended intervention from accidental changes to data coverage, model inputs,
or evaluation labels.

## Economic diagnostics

All 21 costed sign strategies lost money. The least-negative case still lost 95.73%, with Sharpe
`-12.34` and profit factor `0.562`. The apparently best prediction case, `P/N=0.1` in the reference
seed, lost 96.67%, with Sharpe `-11.86` and profit factor `0.617`.

Trading was excluded from inference, so these results do not determine whether the statistical
null passed. They independently show that no shuffled-label accident survived the fixed costs and
turnover of the deliberately simple sign strategy.

## Integrity and compute

All automated gates passed:

- 21/21 cases, 3/3 paired-seed substudies, and 273/273 rolling fits completed;
- every rolling FreqAI gate passed and every required metric was finite;
- the shuffled grid exactly matched the frozen P/N and feature-seed design;
- every case used the same 8,759 OOS observations and `2.24849e-5` zero MSE as Phase 10;
- training-label correlations stayed below the frozen randomization bound;
- all fits used CUDA float64;
- trading was excluded from inference;
- the 2026 holdout was not accessed.

The run used 499.4 summed Freqtrade wall-seconds and 155.2 summed CUDA training-seconds. Peak
tracked VRAM was 340.7 MiB. The largest tested model again used 107,950 explicit predictors,
corresponding to `P/N=50`.

## Limitations and decision

The test uses within-window permutations, three paired randomizations at the robust points, one
asset, one frequency, and one development year. Overlapping training windows reuse observations,
so the 273 fits are not 273 independent statistical trials. The IC values are descriptive and are
not treated as independent p-values.

Because the original Phase 10 models already failed to beat zero, this control cannot prove that a
successful model is leakage-free; it only shows that the present pipeline behaves sensibly under
label destruction. Later positive models would need this control repeated under their frozen
configuration, plus the scheduled lookahead and recursive analyses.

Decision: accept the shuffled-label negative control, reject any leakage or alpha alarm from this
phase, retain the stronger conclusion that double-descent shape survives complete label
destruction, reject all Phase 11 models for deployment, and keep the 2026 holdout sealed.

## Reproduction

```powershell
python scripts/run_double_descent_phase11.py `
  --data-dir user_data\data\binance
```

Generated evidence is under `user_data/research_results/double_descent/phase11/` and remains
ignored by Git. The main artifacts are `summary.json`, `shuffled_label_map.csv`,
`matched_comparisons.csv`, `robustness_aggregates.csv`, and `shuffled_label_control.html`.
