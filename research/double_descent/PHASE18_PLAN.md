# Phase 18 plan: recursive-indicator stability

## Objective

Phase 18 runs Freqtrade's native `recursive-analysis` against the frozen Phase 15 pipeline. It
tests whether the final indicator values change when Freqtrade varies the amount of startup
history. It does not fit a new research specification, select a model, evaluate PnL, or enter the
sealed 2026 holdout.

All 117 Phase 15 configurations retain the static causal/configuration audit. The nine native
representatives are identical to Phase 17: RFF across BTC, ETH, PAXG and both target horizons,
plus every active OLS, Ridge, and momentum branch. P and seed do not change indicator source code.

## Frozen protocol

- benchmark intervals contain at least 5,000 real 15-minute candles;
- startup candles: `199, 399, 499, 801, 999, 1999, 2494`;
- `801` is the decision point because it is the strategy's configured startup value;
- maximum allowed relative variance at 801: `1e-6`;
- all 25 `%` market features must be finite and within tolerance;
- any other recursively unstable indicator at 801 fails the gate;
- any market feature or unknown indicator-only lookahead finding fails the gate;
- known FreqAI target and `do_predict` tail findings are retained and quarantined only because
  Phase 17 independently found zero entry and exit mismatches;
- Phase 18 diagnostic identifiers and `backtest_cache = none` prevent stale model reuse;
- the lifecycle-only CUDA close flag from Phase 17 prevents recursive-analysis from retaining a
  GPU worker for every internal Backtesting instance.

The maximum value of 2,494 is the native limit reported by this Binance 15-minute configuration.
The initially considered 3,999 and 4,999 points were rejected during the pre-matrix OLS smoke test,
before inspecting any recursive result.

## Interpretation limits

`recursive-analysis` compares only the final indicator row; it does not test entries or exits.
FreqAI also expands each startup window by the frozen 90-day training history, so this is a test of
the actual FreqAI pipeline rather than a bare calculation with exactly N raw candles. Phase 17's
prefix-invariance and native signal audits remain complementary requirements.

## Execution

Prepare the complete inventory and data audit:

```powershell
& 'C:\Users\Ricar\Desktop\projects\freqtrade\.venv\Scripts\python.exe' `
  scripts\run_double_descent_phase18.py --stage prepare
```

Run or resume all native representatives:

```powershell
& 'C:\Users\Ricar\Desktop\projects\freqtrade\.venv\Scripts\python.exe' `
  scripts\run_double_descent_phase18.py --stage run
```

A filtered `--case-id` invocation is useful for diagnostics but is deliberately unable to pass the
final full-matrix gate.
