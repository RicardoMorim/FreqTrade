# Phase 15 results: 15-minute frequency robustness

## Decision

Phase 15 passed its complete integrity gate: all 117 predeclared cases and all 1,287 rolling fits
completed across BTC, ETH, and the PAXG tokenized-gold proxy. The largest model used 431,950 RFF
predictors (`P/N=50`) with float64 CUDA training.

The interpolation peak and second-descent shape replicated on all three assets and all three
primary seeds. This was not useful benign overfitting. No RFF case beat the zero-return forecast,
and the best RFF remained between 2.19 and 2.84 times the zero-forecast MSE, depending on the
asset. At `P/N=50`, the recovery from the interpolation peak was visually large but remained far
worse than both the small RFF and zero-return forecasts.

## Integrity and recovery

- preparation and hardware benchmark gates passed;
- 78/78 RFF cases and 39/39 baseline cases passed;
- 1,287/1,287 expected rolling fits were recovered;
- every generated configuration used 15-minute candles and the predeclared target horizon;
- the measured effective training sizes were `N=8,639` for the next-15-minute target and
  `N=8,636` for the matched one-hour control;
- the 25 causal input features and the 10 bps reference fee remained frozen;
- the 2026 holdout remained sealed.

A Windows crash interrupted the largest ETH case. On resume, two partial diagnostic records from
the interrupted process coexisted with the 13 windows from the valid completed run. Recovery now
filters diagnostics by the exact FreqAI run identifier. No model, prediction, or backtest was
changed; only stale partial diagnostics were excluded from aggregation.

## Prediction results

| Asset | Primary seeds with double-descent shape | Useful second descent | Best RFF MSE / zero | Best simple forecast |
| --- | ---: | ---: | ---: | --- |
| BTC | 3/3 | 0/3 | 2.5066 | zero return |
| ETH | 3/3 | 0/3 | 2.8361 | zero return |
| PAXG | 3/3 | 0/3 | 2.1928 | historical mean (`0.99988` of zero MSE) |

The matched one-hour control also produced an interpolation peak and second descent for each
asset, but no RFF beat zero there either. Increasing sampling density therefore did not turn the
geometric phenomenon into predictive value.

## Interpretation

Phase 15 strengthens the existence of a representation-level interpolation phenomenon: its shape
survives a new frequency, a second cryptocurrency, and a tokenized-gold proxy. It simultaneously
weakens the economic claim. The second descent recovers from catastrophic interpolation error but
does not recover far enough to beat a trivial forecast.

The evidence supports `double-descent geometry`, not `benign overfitting for financial return
prediction`.

## Artifacts

- `user_data/research_results/double_descent/phase15/summary.json`
- `user_data/research_results/double_descent/phase15/rff_results.csv`
- `user_data/research_results/double_descent/phase15/baseline_results.csv`
- `user_data/research_results/double_descent/phase15/curve_assessments.csv`
- `user_data/research_results/double_descent/phase15/frequency_robustness.html`
