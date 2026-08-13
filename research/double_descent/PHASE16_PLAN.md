# Phase 16 plan: economic robustness and execution costs

## Objective

Phase 16 measures whether the chronological OOS predictions from Phase 15 remain economically
usable after funding and increasingly conservative transaction-cost assumptions. It does not tune
entry thresholds, choose a model by PnL, retrain FreqAI, or touch the sealed 2026 holdout.

All 87 cases from the primary `native_15m` study are included:

- 63 RFF cases: 21 predeclared seed/ratio cells for each of BTC, ETH, and PAXG;
- 24 baselines: eight frozen baselines for each asset.

The amounts, directions, entry times, and exit times exported by Phase 15 remain fixed. This is a
cost sensitivity on identical trades, so a scenario cannot improve itself by changing exposure
after seeing a different equity path.

## Frozen scenarios

| Scenario | Fee/side | Slippage/side | Funding | Purpose |
|---|---:|---:|---:|---|
| `price_only` | 0 bps | 0 bps | excluded | isolate price PnL |
| `funding_only` | 0 bps | 0 bps | actual | isolate funding effect |
| `optimistic` | 5 bps | 2.5 bps | actual | low-cost sensitivity |
| `phase15_reference` | 10 bps | 0 bps | actual | reproduce native Phase 15 |
| `conservative` | 10 bps | 5 bps | actual | adverse execution |
| `stress` | 20 bps | 10 bps | actual | severe cost stress |

Actual `funding_fees` come from the native Binance-futures backtests. Slippage is deterministic and
does not claim to be an order-book replay. Market impact and capacity remain outside the evidence.

The main outputs are net return, daily Sharpe and Sortino, realized close-to-close drawdown, profit
factor, win rate, turnover, holding time, funding PnL, fees, and slippage. The native-reference
scenario must reproduce every exported trade PnL within `1e-5` before the integrity gate passes.

Trading double ascent is reported only descriptively at the common seed grid
`P/N = 0.1, 1, 1.02, 5, 50`. Prediction MSE remains the primary evidence for double descent.

## Execution

Phase 15 must first have a passing final summary. Then run:

```powershell
& 'C:\Users\Ricar\Desktop\projects\freqtrade\.venv\Scripts\python.exe' `
  scripts\run_double_descent_phase16.py
```

The analysis reuses exported trades and does not retrain models, so it is expected to finish much
faster than Phase 15. Outputs are written to
`user_data/research_results/double_descent/phase16/`.
