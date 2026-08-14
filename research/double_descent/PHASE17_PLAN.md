# Phase 17 plan: lookahead-bias audit

## Objective

Phase 17 audits the frozen Phase 15 feature, target, prediction, and signal pipeline with the native
`freqtrade lookahead-analysis` mechanism. It does not select a model by prediction error or PnL,
change a parameter, or enter the sealed 2026 holdout.

All 117 Phase 15 configurations receive a source/configuration audit. Native sliced-backtest runs
are predeclared by code-path equivalence:

- the small RFF case (`P/N=0.1`) on BTC, ETH, and PAXG covers asset-dependent data paths;
- the BTC RFF case is repeated for the matched one-hour target;
- BTC native-15m OLS, Ridge, one-hour momentum, 24-hour momentum, and volatility-adjusted
  momentum cover every active branch of `Phase12BaselineRegressor`.

P, seed, and asset do not introduce different feature or signal implementations. Testing every
large-P cell natively would rerun dozens of rolling GPU fits per signal without adding a new causal
path. The complete manifest nevertheless verifies every original config and maps all cases to the
audited source.

`zero_return`, `historical_mean`, and `buy_and_hold` cannot supply the frozen minimum of 20 trades
to the native mechanism. They remain in the config/source audit and are explicitly reported as
source-only, not silently declared bias-free.

## Frozen native protocol

- 20 signals per representative;
- one-month development-only diagnostic intervals;
- market orders with the required diagnostic-only entry/exit `price_side = "other"` overrides;
- unique Phase 17 FreqAI identifiers, leaving Phase 15 configs and artifacts unchanged;
- no biased entry signals, exit signals, market features, or unknown indicators permitted;
- raw `has_bias` output is always retained. FreqAI `&` targets and the known derived runtime
  columns (`do_predict` and the four entry/exit columns) are adjudicated only when Freqtrade
  reports zero mismatched entries and zero mismatched exits;
- diagnostic model caches are deleted after each completed case, while logs, CSVs, configs, and
  summaries remain reproducible.
- RFF diagnostic configs close their CUDA worker after each sliced inference. This prevents the
  native lookahead loop from retaining one GPU process per cut until CLI exit; model arithmetic is
  unchanged and the Phase 15 configs do not enable this lifecycle-only flag.

Protocol version 2 froze the runtime-column adjudication after the pre-matrix OLS smoke test. The
native tool reported 20 checked trades with zero entry/exit mismatches but flagged target-derived
values at the truncated dataframe tail. This amendment happened before the representative matrix,
does not depend on prediction or economic performance, and cannot excuse a changed trade.

## Execution

Prepare and audit all 117 configurations:

```powershell
& 'C:\Users\Ricar\Desktop\projects\freqtrade\.venv\Scripts\python.exe' `
  scripts\run_double_descent_phase17.py --stage prepare
```

Run the resumable native matrix:

```powershell
& 'C:\Users\Ricar\Desktop\projects\freqtrade\.venv\Scripts\python.exe' `
  scripts\run_double_descent_phase17.py --stage run
```

After a crash, repeat the same command. A single representative can be exercised with `--case-id`,
but such a filtered invocation is deliberately marked partial and cannot pass the final gate.
