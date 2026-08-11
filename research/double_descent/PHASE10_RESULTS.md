# Phase 10 results: market information versus noise features

## Outcome

Phase 10 completed all 64 predeclared cases and 832 rolling model fits. Every integrity gate
passed, every case used the same 8,759 chronological OOS observations, and the 2026 holdout
remained sealed.

The experiment found a robust interpolation peak and second descent in all nine
representation/seed curves. Crucially, the same shape appeared with predictors containing no
market information at all. The prediction results were uniformly negative:

- 0/64 cases beat the zero-return forecast on OOS MSE;
- 0/64 produced positive OOS R2;
- `market_rff` beat matched pure noise in only 1/15 robust cells, at the unstable interpolation
  point rather than in a useful recovered regime;
- adding the 25 market inputs to the matched noise design beat pure noise in only 4/15 cells, with
  the three apparent wins at `P/N=50` smaller than 0.01%;
- 0/64 costed sign strategies had positive return, positive Sharpe, or profit factor above one.

The appropriate interpretation is:

> This dataset exhibits double-descent geometry around interpolation, but Phase 10 finds no
> evidence that the recovery is caused by market information, no useful benign overfitting, and
> no financial alpha.

## Frozen design

The experiment compared four representations without using trading performance for selection:

1. `market_linear`: the 25 standardized causal market inputs, once, as a low-dimensional anchor;
2. `market_rff`: P nested RFF of the market inputs at the Phase 9-frozen `gamma=0.5`;
3. `pure_noise`: P deterministic iid-N(0,1) predictors keyed only by timestamp, feature id, and
   seed;
4. `market_plus_noise`: the first 25 columns of the exact matched noise design replaced by the 25
   market inputs, while columns 26 through P remain identical.

The reference seed used
`P/N = 0.10, 0.50, 0.90, 0.98, 1.00, 1.02, 1.10, 2, 5, 10, 50`. Two additional seeds repeated
`0.10, 1.00, 1.02, 5, 50`. All high-dimensional cases therefore have the same total P at a matched
seed and ratio. The noise prefixes are nested across P and invariant to processing chunk size.

All cases used BTC perpetual 1h candles, the 2025 rolling OOS interval, 90-day training windows,
30-day evaluation steps, effective `N=2,159`, ridge zero, float64 CUDA, and a 0.1% fee per side.
The primary endpoint was chronological OOS MSE. Trading was diagnostic only.

## Reference-seed prediction curves

The table reports OOS MSE divided by the zero-return MSE. Values below one would beat the zero
forecast; none did.

| P/N | Market RFF | Pure noise | Market + noise | RFF IC | Noise IC | Mixed IC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | 1.203x | 1.121x | 1.136x | 0.0201 | -0.0050 | 0.0020 |
| 0.50 | 7.096x | 2.022x | 2.039x | -0.0147 | 0.0081 | 0.0059 |
| 0.90 | 69.62x | 10.47x | 10.88x | -0.0037 | 0.0230 | 0.0166 |
| 0.98 | 336.42x | 53.55x | 53.96x | 0.0119 | 0.0182 | -0.0045 |
| 1.00 | 8,439.08x | 12,157.43x | 68,179.34x | 0.0022 | 0.0179 | 0.0113 |
| 1.02 | 300.00x | 53.12x | 65.60x | 0.0025 | 0.0139 | 0.0025 |
| 1.10 | 71.87x | 11.56x | 12.18x | 0.0201 | 0.0129 | 0.0050 |
| 2.00 | 12.44x | 2.050x | 2.063x | -0.0005 | 0.0007 | -0.0007 |
| 5.00 | 6.522x | 1.268x | 1.268x | -0.0022 | -0.0084 | -0.0081 |
| 10.00 | 5.374x | 1.124x | 1.124x | -0.0033 | -0.0081 | -0.0081 |
| 50.00 | 4.516x | 1.027x | 1.027x | 0.0003 | -0.0178 | -0.0178 |

Every curve peaked exactly at `P/N=1`. All recovered more than 99.96% of the increase from the
best underparameterized point to the peak. Pure noise and the mixed representation eventually
fell below their own `P/N=0.1` error, satisfying the stricter first-versus-second-descent shape
criterion. Market RFF did not: at `P/N=50` it remained 3.67-3.75 times worse than its small-model
control across seeds.

The shape result is therefore real and repeatable, but it is not specific to financial signal.

## Matched representation controls

The robust comparison uses five ratios and three seeds, for 15 exactly matched cells per pair.

| Left representation | Right representation | Left wins | Geometric MSE ratio | Median MSE ratio |
| --- | --- | ---: | ---: | ---: |
| Market RFF | Pure noise | 1/15 | 3.251x | 4.430x |
| Market RFF | Market + noise | 3/15 | 1.386x | 4.419x |
| Market + noise | Pure noise | 4/15 | 2.346x | 1.017x |

A ratio below one favours the left representation. Market RFF's only win over pure noise occurred
for the reference seed at `P/N=1`, where both estimators are exploding and RFF happened to have the
smaller interpolation catastrophe. It did not beat pure noise in any matched recovered point.

The mixed representation was consistently worse than pure noise at `P/N=0.1`, dramatically less
stable at interpolation, and 19-24% worse immediately above it. At `P/N=50`, mixed MSE was lower in
all three seeds, but only by approximately 0.0012%, 0.0090%, and 0.0045%. Those tiny differences do
not overcome the zero forecast and are not credible evidence of incremental market information.

The geometric mean for mixed versus pure noise is distorted upward by the intentionally retained
interpolation catastrophes. The median and the cell-level results are therefore more informative
than a single average effect.

## Negative control and the meaning of the second descent

Pure noise produced the full textbook shape in all three seeds:

- at `P/N=0.1`, mean MSE was 1.120x zero;
- at `P/N=1`, the MSE ratio ranged from about 4,226x to 12,157x;
- at `P/N=5`, mean MSE was 1.263x zero;
- at `P/N=50`, mean MSE was 1.024x zero.

Nevertheless, pure noise beat zero in 0/15 robust cells and had positive R2 in 0/15. Its apparent
high-dimensional improvement is best understood as variance collapse toward the constant
forecast. At `P/N=50`, the median OOS prediction standard deviation was only `0.000698`, versus
`0.004742` for realized returns, while its MSE remained 2.2-2.7% above zero.

This control changes the interpretation of earlier phases. A peak followed by recovery is
necessary to identify double-descent geometry, but it is not sufficient evidence that a financial
model has learned signal or that overfitting is benign.

## Raw-market anchor and economic diagnostics

The 25-feature linear anchor was the least-bad case overall, but still failed prediction and
trading criteria:

- OOS MSE `2.3298e-5`, or 1.036x the zero baseline;
- OOS R2 `-0.0362`;
- IC `0.0096`;
- net return `-94.16%`;
- Sharpe `-11.76` and profit factor `0.619`.

Across all 64 cases, every sign strategy lost money after the fixed 0.1% fee per side. The least
negative result was the linear anchor above. No case had positive Sharpe or profit factor above
one. These strategies traded between 1,671 and 2,983 times, confirming that costs and turnover are
economically material, but the prediction failure already exists before trading is considered.

## Numerical and effective-dimension findings

The controls explain why nominal P cannot be read as statistical dimension:

- market RFF had mean effective rank only about 55 at `P/N=0.1` and 75 at `P/N=50`;
- pure noise increased from effective rank about 196 to about 2,116, close to the 2,159 training
  observations;
- the mixed design converged to the same effective rank and condition number as pure noise at
  `P/N=50`;
- median condition number at `P/N=50` was about 583 for market RFF but only 1.328 for pure and mixed
  noise.

All three representations interpolated at or immediately above the nominal threshold. The
interpolation-region explosions persisted in float64 and varied substantially by seed, as expected
for an ill-conditioned ridgeless system. Away from the threshold, the pure-noise minimum-norm
solution became well conditioned and its predictions shrank toward the target mean. That is a
geometric recovery, not learned return predictability.

## Integrity and compute

All automated gates passed:

- 64/64 cases, 10/10 substudies, and 832/832 rolling fits completed;
- every rolling FreqAI gate passed and every required metric was finite;
- every case used the same 8,759 OOS targets and `2.24849e-5` zero MSE;
- feature counts matched measured `N=2,159`;
- runtime diagnostics verified the requested representation and market/noise composition;
- noise was timestamp-keyed, nested across P, independent of values and targets, and tested for
  chunk invariance and exact matched-column replacement;
- all fits used CUDA float64;
- trading was excluded from inference;
- the 2026 holdout was not accessed.

The run used 6,385.0 summed Freqtrade wall-seconds and 2,522.1 summed CUDA training-seconds. Peak
tracked VRAM was 340.7 MiB, well below the RTX 5080 Laptop GPU capacity. The restart-safe runner
recovered completed cases after the interrupted machine session rather than silently recomputing or
mixing partial outputs.

## Limitations and decision

This phase covers one asset, one frequency, one development year, three seeds only at the robust
points, one frozen gamma, and one ridgeless training-window size. The pairwise counts are
descriptive rather than a high-powered significance test. Pure noise is a negative control, not a
candidate model, and any isolated favourable noise result would require a leakage audit rather
than an alpha interpretation.

Decision: accept the double-descent curve as a robust property of these ridgeless high-dimensional
fits, reject the claim that its second descent demonstrates financial benign overfitting, reject
all Phase 10 models for deployment, preserve the 2026 final holdout, and move to the shuffled-label
control before making any predictive claim.

## Reproduction

```powershell
python scripts/run_double_descent_phase10.py `
  --data-dir user_data\data\binance
```

Generated evidence is under `user_data/research_results/double_descent/phase10/` and remains
ignored by Git. The principal artifacts are `summary.json`, `representation_map.csv`,
`robustness_aggregates.csv`, `noise_feature_controls.html`, and the per-case rolling diagnostics.
