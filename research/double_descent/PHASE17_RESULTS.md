# Phase 17 results: native lookahead-bias audit

## Decision

Phase 17 passed every predeclared gate. The source/configuration audit covered all 117 frozen
Phase 15 cases, and all nine native representatives completed Freqtrade's sliced-backtest
`lookahead-analysis`. Across 180 tested signals, Freqtrade reported zero biased entries and zero
biased exits. No market feature or unknown indicator was flagged, and the sealed 2026 holdout was
not used.

The native result therefore supports the causal integrity of the frozen Phase 15 pipeline. It does
not change the scientific conclusion of the earlier phases: the observed double-descent geometry
did not beat the zero-return prediction baseline and did not survive executable costs.

## Native coverage

| Path represented | Cases | Signals per case | Biased entries | Biased exits | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| RFF, native 15m, BTC/ETH/PAXG | 3 | 20 | 0 | 0 | Pass |
| RFF, matched 1h target, BTC | 1 | 20 | 0 | 0 | Pass |
| OLS and Ridge, native 15m, BTC | 2 | 20 | 0 | 0 | Pass |
| Momentum baselines, native 15m, BTC | 3 | 20 | 0 | 0 | Pass |
| **Total** | **9** | **180 total** | **0** | **0** | **Pass** |

All 117 source configurations passed the static causal audit. P, RFF seed, and asset change the
numerical representation or data, not the feature and signal implementation, so one native run
per distinct causal path was predeclared. The inactive `zero_return`, `historical_mean`, and
`buy_and_hold` branches cannot produce 20 independent trades and remained explicitly source-only.

## Raw Freqtrade output and adjudication

Freqtrade's raw `has_bias` field was true in all nine runs. The reported columns were limited to
the forward target and its FreqAI metadata (`&-forward_return`, mean, and standard deviation), plus
known runtime/signal columns (`do_predict`, `enter_long`, `enter_short`, `exit_long`, and
`exit_short`). These values differ at the deliberately truncated tail of sliced dataframes.

Protocol version 2 froze the adjudication before the representative matrix: those known columns
are ignored only when native entry and exit mismatch counts are both zero. Any `%` market feature,
unknown column, or changed trade would fail the gate. All nine runs met the strict condition, so
the raw output is retained while the actionable-bias result is a pass.

## Operational stability

Native lookahead analysis constructs many backtest instances inside one process. The initial RFF
attempt retained one CUDA worker per sliced backtest until process exit, eventually filling the
16 GB GPU. A Phase-17-only lifecycle flag now closes the worker immediately after sliced
inference; it does not alter model arithmetic or any Phase 15 configuration.

The corrected matrix completed 168 RFF rolling fits across the four representatives. Observed VRAM
remained around 5.2 GB under load and returned close to idle between fits, instead of increasing
with every signal. Focused lifecycle and research tests passed before the matrix was resumed.

## Interpretation

This phase removes a major alternative explanation for the negative research result: the frozen
features, targets, predictions, and sign-based trading rules did not change their entries or exits
when future rows were removed. It is therefore less plausible that leakage created the double-
descent curves seen in Phases 10, 13, 14, and 15.

The result is a pipeline-validity statement, not evidence of predictability or profitability.
Phase 15 still found zero RFF models beating zero-return MSE, and Phase 16 found zero RFF cases
surviving even the optimistic execution-cost scenario.

## Artifacts

- `user_data/research_results/double_descent/phase17/summary.json`
- `user_data/research_results/double_descent/phase17/preparation.json`
- `user_data/research_results/double_descent/phase17/case_manifest.csv`
- `user_data/research_results/double_descent/phase17/config_audit.csv`
- `user_data/research_results/double_descent/phase17/native_results.csv`
- `user_data/research_results/double_descent/phase17/checkpoint.json`
- per-case configs, logs, CSVs, results, and training diagnostics under
  `user_data/research_results/double_descent/phase17/runs/`
