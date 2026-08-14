# Final report: Double Descent in Financial Return Prediction

Report date: 2026-08-14

Final evaluation: 2026-01-01 through 2026-07-31 inclusive

Frozen-protocol commit: `c8d561ececa7a2ed205eb7b5a1ed42cefe19ffbb`

## Executive conclusion

The project found a real and highly reproducible **double-descent-shaped error curve** in the
ridgeless RFF regression system. OOS MSE rises catastrophically near `P/N = 1` and then falls again
as `P` grows far beyond `N`. That geometry appeared in synthetic controls, real market features,
pure noise, shuffled-label controls, multiple seeds, different training windows, BTC, ETH, the
PAXG gold proxy, 1h data, 15m data, and a matched one-hour target.

The project did **not** find useful benign overfitting in financial return prediction. The second
descent only recovers from the interpolation catastrophe. It never recovers enough to beat a
zero-return forecast, the best small RFF, or simple baselines. This remained true in the final
one-shot 2026 YTD holdout:

- double-descent shape replicated on all three assets and all tested primary seeds;
- 0/78 final-holdout RFF cases beat zero-return MSE;
- 0/108 non-zero prediction cases were significantly better than zero after Bartlett-HAC and
  Benjamini-Hochberg FDR control;
- 105/108 were significantly worse than zero;
- no RFF was profitable under even the optimistic executable-cost scenario;
- no RFF was profitable under the reference 10 bps-per-side fee;
- the only positive costed case was a two-trade PAXG historical-mean baseline, which failed the
  frozen 30-trade minimum.

The correct answer to the research question is therefore nuanced:

> Financial RFF regression exhibits robust interpolation geometry, but this study finds no
> evidence that the second descent represents useful financial generalization, benign
> overfitting, alpha, or tradable profitability.

This is a scientifically useful negative result. It separates the visual phenomenon from the much
stronger claim that extreme overparameterization improves real return prediction.

## What was tested

The experiment used 25 causal market-state variables and deterministic nested RFF. Increasing `P`
therefore appended features from the same random basis instead of drawing an unrelated model. The
main estimator was ridgeless, centered, minimum-norm regression in float64, with a persistent CUDA
worker for the sample-space dual solution.

The research program included:

- controlled synthetic linear and nonlinear problems;
- direct RBF-kernel-limit comparison;
- explicit Ridge regularization;
- multiple RFF seeds;
- training windows from 30 to 365 days;
- pre-holdout gamma calibration;
- market, pure-noise, and market-plus-noise features;
- shuffled training labels;
- zero, mean, linear, Ridge, momentum, volatility-adjusted momentum, and buy-and-hold baselines;
- causal regime attribution;
- BTC-to-ETH replication;
- BTC, ETH, and PAXG tests at 15m, plus a matched one-hour target;
- native Freqtrade lookahead and recursive-indicator audits;
- cost, funding, slippage, numerical-precision, and effective-dimension audits;
- a one-shot 2026 YTD final holdout frozen in Git before its data were opened.

Trading was downstream throughout. Prediction MSE, MAE, OOS R2, IC, directional accuracy, training
error, and `P/N` determined the scientific prediction conclusion; PnL and Sharpe never selected a
model or changed the prediction decision.

## Final holdout design and integrity

The final protocol copied the complete Phase 15 design without selecting a prior winner. It was
committed before downloading or inspecting the 2026 evaluation data.

| Item | Frozen value |
| --- | ---: |
| Holdout | 2026-01-01 to 2026-08-01, end exclusive |
| Assets | BTC, ETH, PAXG perpetual futures |
| Candle frequency | 15 minutes |
| Targets | next 15m return; matched next 1h return |
| Training/evaluation cadence | 90-day rolling train / 30-day step |
| Effective N | 8,639 at 15m; 8,636 for matched 1h target |
| Primary RFF grid | `P/N = 0.1, 0.5, 0.9, 0.98, 1, 1.02, 1.1, 2, 5, 10, 50` |
| Robustness grid | `P/N = 0.1, 1, 1.02, 5, 50` |
| Seeds | 3 primary; 1 matched-horizon control |
| Largest model | 431,950 predictors |
| RFF / baseline cases | 78 / 39 |
| Rolling fits | 936 |
| Economic repricings | 522 |
| Gamma / Ridge / precision | `0.5` / `0` / float64 |
| Prediction inference | 24h Bartlett-HAC, BH-FDR 5% |

All final integrity checks passed:

- 117/117 cases and 936/936 rolling fits completed;
- all prediction metrics were finite;
- all commands used the frozen holdout and 15m configurations;
- chronological prediction dates were unique and accepted;
- the six cost scenarios produced all 522 expected rows;
- reference-cost repricing reproduced native Freqtrade PnL;
- performance was explicitly excluded from the integrity gate;
- the protocol commit predates opening the holdout.

The run used the RTX 5080 Laptop GPU, took approximately 6 hours 8 minutes wall time, and recorded a
maximum per-case peak allocation of 3,945 MiB.

## Final prediction results

### Shape replicated; usefulness did not

All 12 assessed curves — three primary seeds and one matched-horizon control for each of three
assets — showed a global or near-global OOS-MSE peak at `P/N` 0.98 or 1.00 followed by a large
second descent. Recovery from the peak was 99.4% to 99.8%.

This relative recovery is misleading without an absolute baseline. The smallest RFF was always the
least-bad RFF, and even it was much worse than predicting zero:

| Asset / target | Best RFF P/N | Best RFF MSE / zero | Best simple forecast |
| --- | ---: | ---: | --- |
| BTC / next 15m | 0.10 | 2.4125x | zero return |
| ETH / next 15m | 0.10 | 2.3517x | zero return |
| PAXG / next 15m | 0.10 | 3.6852x | zero return |
| BTC / matched 1h | 0.10 | 3.5320x | zero return |
| ETH / matched 1h | 0.10 | 3.2058x | zero return |
| PAXG / matched 1h | 0.10 | 4.9772x | zero return |

At `P/N=50`, the primary-horizon MSE remained dramatically worse than zero across seeds:

| Asset | P/N=50 MSE / zero, seed range | P/N=50 MSE / best small RFF, approximate range |
| --- | ---: | ---: |
| BTC | 99.05x–101.95x | 40.15x–41.06x |
| ETH | 84.21x–88.32x | 30.94x–36.66x |
| PAXG | 171.88x–174.77x | 39.45x–46.64x |

For the reference seed, the median across assets moved from 2.54x zero at `P/N=0.1` to roughly
41,008x at `P/N=1`, then fell to about 102x at `P/N=50`. This is unmistakable double-descent
geometry, but it is not an improvement over the underparameterized regime or the null forecast.

### Statistical inference

The frozen test compared each non-zero model's squared-error loss with the zero forecast using a
two-sided Bartlett-HAC mean test with a 96-candle lag, then controlled all comparisons with
Benjamini-Hochberg FDR at 5%.

| Outcome | Count |
| --- | ---: |
| Non-zero prediction cases | 108 |
| Significantly better than zero | 0 |
| Significantly worse than zero | 105 |
| Not significantly different | 3 |

The three unrejected cases were the rolling historical-mean baselines for BTC, ETH, and PAXG at
the 15m target. Their MSE ratios were 1.00008, 1.00005, and 1.00005: statistically indistinguishable
from, but not better than, zero. Every RFF was significantly worse after FDR correction.

### IC and direction are not enough

The best RFF cells had directional accuracy around 49.6%–51.2% and Pearson IC close to zero. A few
small positive IC or directional-accuracy values did not compensate for badly calibrated forecast
amplitudes. Their MSE and OOS R2 remained decisively worse than zero. This is why the report does
not interpret near-50% direction or isolated positive correlation as predictability.

## Trading and costs

The sign-only strategy generated high turnover for RFF models. At the reference fee, the median RFF
case made 1,799 trades. Gross price-only behavior could look positive: 32 RFF cases had positive
price-only return. That apparent edge disappeared before reaching even the optimistic executable
scenario.

| Cost scenario | Cost assumption | Positive RFF cases | Median all-case return |
| --- | --- | ---: | ---: |
| Price only | no fee, no slippage, no funding | 32 | 0.000 |
| Funding only | funding, no fee/slippage | 32 | 0.000 |
| Optimistic | 5 bps fee + 2.5 bps slippage per side | 0 | -0.722 |
| Reference | 10 bps fee per side | 0 | -0.968 |
| Conservative | 10 bps fee + 5 bps slippage per side | 0 | -1.440 |
| Stress | 20 bps fee + 10 bps slippage per side | 0 | -2.847 |

At the reference cost, all 63 primary RFF economic cases lost money. The least-bad RFF still
returned approximately -91.1%; the median returned -97.1%. The sole positive non-zero-cost case was
the PAXG historical-mean baseline with only two trades, so it failed the frozen minimum-trade gate
and cannot support an economic claim.

The economic conclusion is therefore stronger than “Sharpe was not high enough”: the underlying
prediction test already failed, and turnover converts noisy sign changes into severe cost drag.

## What the 20 phases taught us

| Phase | Question | Result and lesson |
| ---: | --- | --- |
| 1 | Can the framework recover known double descent? | Yes. Both controlled synthetic problems passed, with interpolation at `P/N≈1` and second descent. |
| 2 | Can the hardware handle extreme P? | Yes. CPU and CUDA matched through one million RFF; CUDA made the streamed dual practical. |
| 3 | What is real FreqAI N? | Measured exact rolling N for 30–365 day windows; 25 causal inputs remained stable after preprocessing. |
| 4 | Does the financial curve appear? | Yes geometrically. The peak was at `P/N=1`, but `P/N=50` was still 14.08x worse than zero. |
| 5 | Is it seed-robust? | The shape replicated in 5/5 seeds; useful recovery replicated in 0/5. |
| 6 | Does finite RFF converge to its kernel? | Yes. One million RFF approached the exact RBF kernel, which itself was 12.99x worse than zero. |
| 7 | Is the spike removed by Ridge? | Yes. Ridge removed 99.995% of the spike by shrinking toward zero, but did not reveal alpha. |
| 8 | Does more training data remove the shape? | No. It persisted from 30 to 365 days; absolute performance stayed worse than zero and often deteriorated. |
| 9 | Does kernel scale rescue performance? | `gamma=0.5` was relatively best in development, yet 0/84 cells beat zero. |
| 10 | Does the shape require market information? | No. Pure noise reproduced it; market RFF beat matched noise in only 1/15 robust cells. |
| 11 | Do shuffled labels expose leakage? | The negative control passed: no shuffled model beat zero, while the shape still appeared. |
| 12 | Is complexity justified versus simple baselines? | No. Zero return was best; no RFF or learnable baseline beat it. |
| 13 | Is performance hidden in a market regime? | No useful regime-specific alpha; 105/105 adequate marginal RFF cells were significantly worse than zero. One sparse joint cell remained quarantined. |
| 14 | Does the BTC result replicate on ETH? | Yes for curve geometry, no for useful benign overfitting or alpha. |
| 15 | Does it replicate across frequency, horizon, and gold? | Geometry replicated on BTC, ETH, and PAXG at 15m and the matched 1h target; no RFF beat zero. |
| 16 | Does gross behavior survive costs? | No RFF survived even optimistic costs; all 522 repricings passed integrity checks. |
| 17 | Is there lookahead bias in the trading paths? | Native audit passed all nine representative paths with zero biased entries/exits. |
| 18 | Are indicators recursively unstable? | Native recursive audit passed; all 225 market-feature checks at the frozen startup had zero variance. |
| 19 | Is the interpolation peak numerically robust? | Float64 was repeatable and CPU/CUDA agreed relatively, but one strict absolute-tolerance check failed near the ill-conditioned threshold. Float32 erased the peak through implicit regularization. |
| 20 | Does nominal P equal statistical dimension? | No. At 15m, 431,950 RFF collapsed to effective rank about 59–65; one million RFF converged to kernel effective rank about 23.76. |

## Mechanistic interpretation

The results support an interpolation-instability account:

1. As nominal `P` approaches `N`, the ridgeless system becomes severely ill-conditioned.
2. Training error approaches zero, while coefficient sensitivity and OOS variance explode.
3. Moving beyond the threshold gives the minimum-norm solver more representational redundancy,
   causing a large relative fall in test error.
4. The representation's effective spectral dimension remains tiny compared with nominal `P`.
5. The model therefore converges toward a low-effective-rank RBF-kernel solution, not toward
   hundreds of thousands of independent predictive market directions.
6. The same geometry with pure noise and shuffled labels proves that the curve shape alone does not
   identify market signal.

In the final holdout, the reference-seed median effective rank across assets grew only from about
57.8 at `P/N=0.1` to 62.9 at `P/N=50`, while nominal P grew 500-fold. Near `P/N=1`, the median
condition number reached roughly 713,000 and median OOS MSE reached about 41,008 times zero. At
`P/N=50`, training MSE was effectively zero and the curve had recovered, but OOS MSE was still
roughly 102 times zero. This is interpolation plus variance recovery, not useful generalization.

## Assessment against the original success criteria

| Original criterion | Outcome |
| --- | --- |
| Peak near `P/N=1` | Met robustly |
| Recovery for `P/N>1` | Met robustly in relative curve shape |
| Consistency across seeds | Met |
| OOS prediction improvement | Failed |
| Survival after transaction costs | Failed |
| Survival across periods/windows | Shape survived; usefulness failed |
| Replication on another asset | Shape replicated on ETH and PAXG; usefulness failed |
| Coherence with kernel limit | Met; the limiting kernel was also poor |
| Absence of alpha with shuffled labels | Met |
| Robustness to precision | Qualified: float64 robust; float32 changes the estimator materially |
| Favorable comparison with simple baselines | Failed |
| Performance on final holdout | Failed decisively |

No live or paper-trading deployment is justified by this study.

## Limitations

- The final holdout covers seven complete months of 2026, not a full calendar year.
- Later prequential fits train on earlier 2026 observations, while predictions remain strictly
  chronological. This is realistic rolling evaluation, not a single untouched static train/test
  split.
- PAXG perpetual is a tokenized-gold proxy, not XAU spot or COMEX futures.
- All three assets use Binance; cross-venue replication remains open.
- The deterministic cost model includes fees, funding, and slippage sensitivity but not order-book
  market impact or latency.
- Phase 19's strict gate remains failed for one maximum-absolute CPU/CUDA tolerance at the most
  ill-conditioned seed. This qualifies exact interpolation amplitudes, although it does not change
  the no-alpha conclusion: neither backend nor precision beat zero.
- Three final primary seeds are enough to expose consistency, but not to estimate a broad
  distribution over random representations with high precision.
- The conclusions apply to this causal feature set, RBF/RFF family, rolling protocol, and tested
  assets/horizons. They do not prove that all financial models lack benign overfitting.

## Paper-level contribution

The publishable contribution is not “a huge neural model predicts crypto.” It is a careful
separation of three claims that are often conflated:

1. **Interpolation geometry exists.** The error curve has a robust double-descent shape.
2. **Benign overfitting is useful.** The overparameterized estimator beats relevant predictive
   baselines. This was not observed.
3. **The forecast is economically usable.** A frozen trading transformation survives realistic
   costs. This was not observed.

A defensible paper framing is:

> Robust double-descent geometry without useful benign overfitting in financial return prediction.

The strongest evidence is the convergence of independent controls: synthetic validation,
kernel-limit agreement, seed and window robustness, pure-noise and shuffled-label curves, simple
baselines, cross-asset/frequency replication, native bias audits, numerical diagnostics, effective
dimension, and a Git-frozen final holdout. Together they show why a visually impressive second
descent can be genuine as estimator geometry and still irrelevant for prediction or trading.

## Reproduction and artifacts

- Frozen protocol: [FINAL_HOLDOUT_PROTOCOL.md](FINAL_HOLDOUT_PROTOCOL.md)
- Machine-readable final snapshot: [FINAL_HOLDOUT_RESULTS.json](FINAL_HOLDOUT_RESULTS.json)
- Phase-by-phase methodology: [README.md](README.md)
- Generated final evidence: `user_data/research_results/double_descent/final_holdout_2026_ytd/`
- Interactive report: `user_data/research_results/double_descent/final_holdout_2026_ytd/final_holdout.html`

The generated evidence directory is intentionally ignored by Git because it contains large,
reproducible run artifacts. The frozen protocol, source, tests, this report, and the compact result
snapshot are versioned.
