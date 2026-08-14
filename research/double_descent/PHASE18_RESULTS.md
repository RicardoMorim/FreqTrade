# Phase 18 results: recursive-indicator stability

## Decision

Phase 18 passed its frozen gate. Freqtrade's native `recursive-analysis` completed for all nine
predeclared representatives of the Phase 15 pipeline. No market feature or other actionable
indicator changed with startup history, and the sealed 2026 holdout was not used.

This result supports the causal integrity of the feature pipeline. It does not change the negative
prediction and economic conclusions from Phases 15 and 16: stable indicators are necessary for a
valid experiment, but they are not evidence that the models predict returns profitably.

## Completed protocol

- 117/117 Phase 15 source configurations passed the static configuration and causal-source audit.
- 9/9 native representatives passed: four RFF paths and five active baseline branches.
- BTC, ETH, and PAXG diagnostic intervals contained 5,664, 5,664, and 5,952 unique real 15-minute
  candles respectively, with no missing or duplicate timestamps.
- All seven requested startup calculations ran for every representative: `199, 399, 499, 801,
  999, 1999, 2494` (63 calculations in total).
- The configured strategy startup, `801`, was included as the frozen decision point.
- All 225 market-feature checks at that point were finite and had exactly zero relative variance,
  below the predeclared `1e-6` tolerance.
- The native output contained zero recursive-indicator rows and reported no variance for all nine
  representatives.
- All runs used development data ending before the 2026 holdout.

The complete matrix took 573.3 seconds of case wall time. The four RFF checks accounted for 529.9
seconds; the five baseline checks accounted for 43.4 seconds. During the RFF checks, CUDA memory
remained near 5.2 GB on the 16 GB GPU rather than accumulating between internal analyses.

## Native representatives

| Asset/path | Model branch | Features | Result | Maximum market-feature variance at 801 |
|---|---:|---:|---:|---:|
| BTC, next 15m | RFF | 864 | pass | 0 |
| ETH, next 15m | RFF | 864 | pass | 0 |
| PAXG, next 15m | RFF | 864 | pass | 0 |
| BTC, matched 1h target | RFF | 864 | pass | 0 |
| BTC, next 15m | OLS | 25 | pass | 0 |
| BTC, next 15m | Ridge | 25 | pass | 0 |
| BTC, next 15m | 1h momentum | 25 | pass | 0 |
| BTC, next 15m | 24h momentum | 25 | pass | 0 |
| BTC, next 15m | volatility-adjusted 24h momentum | 25 | pass | 0 |

The RFF representatives use `P/N=0.1` because changing P or the nested seed prefix changes the
regression representation, not the indicator formulas under test. Every P/seed configuration is
still covered by the static audit and the shared-code declaration.

## Indicator-only lookahead output

Freqtrade reported the same four indicator-only lookahead names in every representative:

- `&-forward_return`;
- `&-forward_return_mean`;
- `&-forward_return_std`;
- `do_predict`.

The first three are FreqAI target/statistic columns and the last is FreqAI runtime state at the
truncated diagnostic tail. These 36 raw findings remain present in
`indicator_lookahead.csv`; they were not silently removed. They are quarantined only under the
predeclared rule that Phase 17 independently had to show zero entry and exit mismatches. Phase 17
did so for the same nine paths. Any `%` market feature or unknown column would have failed Phase
18; none appeared.

## Protocol amendment before the matrix

The initially considered startup values `3999` and `4999` exceeded the maximum accepted by native
Freqtrade for this Binance 15-minute configuration. An OLS smoke test exposed the native maximum
of `2494` before the representative matrix was inspected. The grid was therefore frozen with
`2494` as its largest value before the full run, and the configured value `801` remained unchanged.

## Interpretation limits

1. `recursive-analysis` compares the final indicator row; it does not test entry or exit
   decisions. Phase 17 supplies the separate sliced-signal audit.
2. FreqAI adds the frozen 90-day training history to each startup window. This tests the deployed
   FreqAI pipeline, not a bare indicator function given exactly N candles.
3. Freqtrade stops the comparison loop after the smallest startup window has no variance. It still
   calculates all seven requested windows, but the displayed comparison is against the smallest
   window. No variance at 199 candles is at least as strict as the configured 801-candle check for
   the native comparison, while the Phase 18 parser separately requires all 25 features at 801.
4. Shared-code representative coverage is appropriate for recursive indicators, but this phase is
   not a numerical-stability test of every RFF fit. That is a separate experiment.

## Reproduction

```powershell
& 'C:\Users\Ricar\Desktop\projects\freqtrade\.venv\Scripts\python.exe' `
  scripts\run_double_descent_phase18.py --stage prepare

& 'C:\Users\Ricar\Desktop\projects\freqtrade\.venv\Scripts\python.exe' `
  scripts\run_double_descent_phase18.py --stage run
```

The runner is resumable and fingerprints the configuration, research code, strategy, model
adapter, and Freqtrade recursive-analysis engine. A filtered `--case-id` run cannot pass the
full-matrix gate.

## Artifacts

- `user_data/research_results/double_descent/phase18/summary.json`
- `user_data/research_results/double_descent/phase18/preparation.json`
- `user_data/research_results/double_descent/phase18/case_manifest.csv`
- `user_data/research_results/double_descent/phase18/config_audit.csv`
- `user_data/research_results/double_descent/phase18/data_audit.csv`
- `user_data/research_results/double_descent/phase18/native_results.csv`
- `user_data/research_results/double_descent/phase18/recursive_variances.csv`
- `user_data/research_results/double_descent/phase18/indicator_lookahead.csv`
- `user_data/research_results/double_descent/phase18/checkpoint.json`
