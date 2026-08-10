# Phase 3 Results: Effective FreqAI Sample Size

- Run date: 2026-08-10
- Commit base: `0f5dcc785`
- Market: Binance BTC/USDT perpetual futures
- Timeframe: 1 hour
- Frozen development timerange: 2025-01-01 to 2026-01-01

## Outcome

Phase 3 passed its predeclared gate. The experiment observed 65 genuine rolling FreqAI training
windows: 13 for each of the 30, 60, 90, 180, and 365-day training periods. Every subprocess
completed, every requested period produced measurements, and the processed matrices contained no
non-finite values.

| Training window | Raw rows | Effective N | Retained | Windows | Predictors |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 30 days | 720 | 719 | 99.8611% | 13 | 25 |
| 60 days | 1,440 | 1,439 | 99.9306% | 13 | 25 |
| 90 days | 2,160 | 2,159 | 99.9537% | 13 | 25 |
| 180 days | 4,320 | 4,319 | 99.9769% | 13 | 25 |
| 365 days | 8,760 | 8,759 | 99.9886% | 13 | 25 |

The single removed row in every window is the final observation whose next-hour return target is
not yet available. Rolling indicators are causally warmed with candles preceding the nominal
training slice, so their startup NaNs do not remove observations inside that slice.

## Data audit

Freqtrade downloaded the market data directly from Binance. The phase loaded 17,721 unfilled
hourly futures candles covering the required 2023-12-24 16:00 UTC through 2026-01-01 00:00 UTC
range. The timestamps were unique and contiguous: zero duplicate timestamps and zero detected
hourly gaps.

The evaluation period ends at 2026-01-01. More recent data was not used for model selection or
evaluation and remains available for a later untouched holdout.

## Frozen preprocessing

The probe went through the actual `self.freqai.start()` rolling-training path. To make N directly
interpretable, this phase froze the following controls:

- no internal random test split (`test_size=0`, `shuffle=false`);
- no PCA, dissimilarity-index removal, SVM outlier removal, or shifted candles;
- no recency/sample weighting;
- a one-candle forward-return label;
- a fixed 25-variable causal market-state vector;
- a mean regressor used only to traverse and measure the pipeline.

The strategy deliberately emits no entries or exits. This phase makes no predictive or economic
performance claim.

## Grid implied by the observation

The initial main sweep should use the 90-day window, for which `N=2,159`. This gives a dense
interpolation neighbourhood of:

| Target P/N | RFF predictors P |
| ---: | ---: |
| 0.90 | 1,943 |
| 0.95 | 2,051 |
| 0.98 | 2,116 |
| 1.00 | 2,159 |
| 1.02 | 2,202 |
| 1.05 | 2,267 |
| 1.10 | 2,375 |

The full predeclared grid ranges from 216 to 107,950 RFF predictors for this window. Exact grids
for all training periods are stored in `aggregate.json`. `P` denotes explicit RFF predictors; a
fitted intercept adds one scalar degree of freedom, so numerical interpolation should be located
empirically from rank and training error rather than assumed to occur at an exact rounded ratio.

## Reproduction

```powershell
C:\Users\Ricar\Desktop\projects\freqtrade\.venv\Scripts\python.exe `
  scripts\run_double_descent_phase3.py `
  --data-dir user_data\data\binance `
  --train-period-days 30 --train-period-days 60 --train-period-days 90 `
  --train-period-days 180 --train-period-days 365
```

Generated evidence is written to
`user_data/research_results/double_descent/phase3/` and is intentionally ignored by Git. The key
artifacts are `summary.json`, `measurements.csv`, `aggregate.json`, per-run configs and logs, and an
interactive `effective_n.html` plot.

## Decision

The effective-sample-size gate is satisfied. Phase 4 may construct the financial RFF sweep from
the observed N values, beginning with the frozen 90-day development experiment. Before extending
the real-N sweep to the most extreme ratios, compute and memory estimates must be refreshed at
`N=2,159`, because the Phase 2 benchmark used `N=128` and the streamed dual solve scales
quadratically in N.
