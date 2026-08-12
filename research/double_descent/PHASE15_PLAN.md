# Phase 15 plan: 15-minute frequency robustness with a gold proxy

## Objective

Phase 15 tests whether the interpolation peak and second descent observed at 1h remain visible when
the sampling frequency increases to 15m. It does not tune a new model to improve the Phase 14 result.
Gamma, regularization, seeds, costs, rolling dates, and the 2026 holdout remain frozen.

The asset set is BTC, ETH, and `PAXG/USDT:USDT`. PAXG is a Binance perpetual on tokenized gold. It
is a useful same-engine gold proxy, but it is not XAU spot or COMEX gold futures; conclusions must
retain that limitation.

Binance PAXG futures history starts on 27 March 2025. BTC and ETH therefore retain the complete
2025 OOS interval, while the PAXG arm starts on 4 July 2025 so that its first prediction still has
the frozen 90-day training window and 801 startup candles available. The extra candle is an
inclusive-boundary safety buffer; the feature history itself requires 800. PAXG is a coverage-limited
diagnostic with seven rolling windows, not a date-matched full-year replication.

## Two predeclared studies

The primary study predicts the next 15-minute return. Its 25 market features retain the same
clock-time lookbacks used by the 1h experiment: for example, a one-hour return uses four bars and a
24-hour return uses 96 bars.

The diagnostic control predicts the next one-hour return from the same 15-minute data. This helps
separate a change in forecast horizon from a change in sampling density. It uses one seed and the
sparse grid `P/N = 0.1, 1, 1.02, 5, 50`; it is not a second confirmatory sweep.

Expected effective training sizes are `N=8,639` for the native target and `N=8,636` for the
one-hour target. These values are measured through the real FreqAI pipeline before P is calculated.
The largest primary model has 431,950 RFF predictors.

The full design contains 117 cases and 1,287 expected rolling fits:

- primary: 21 RFF cases plus eight baselines for each of three assets;
- control: five RFF cases plus five baselines for each of three assets.
- BTC and ETH contribute 13 rolling fits per case; PAXG contributes seven.

Trading remains downstream and descriptive. Prediction inference uses chronological OOS MSE
normalized by the zero-return forecast. Shape replication and useful benign overfitting are reported
separately.

## Staged execution

The preparation stage downloads no data by itself. With `--download-data`, it downloads 15m
futures, funding-rate, and mark candles for all three instruments, audits coverage, and measures N:

```powershell
& 'C:\Users\Ricar\Desktop\projects\freqtrade\.venv\Scripts\python.exe' `
  scripts\run_double_descent_phase15.py `
  --stage prepare `
  --download-data
```

The benchmark runs one BTC window at `P/N = 0.1, 1, 50`. It must fit inside the configured timeout
and 90% of available VRAM before the full sweep is allowed:

```powershell
& 'C:\Users\Ricar\Desktop\projects\freqtrade\.venv\Scripts\python.exe' `
  scripts\run_double_descent_phase15.py `
  --stage benchmark
```

After both gates pass, start the resumable full experiment:

```powershell
& 'C:\Users\Ricar\Desktop\projects\freqtrade\.venv\Scripts\python.exe' `
  scripts\run_double_descent_phase15.py `
  --stage run
```

If Windows or the GPU stops during the full run, repeat the same command. Completed cases are
recovered from their isolated asset/study artifacts. A small end-to-end pipeline check can be run
in a separate output directory with `--stage all --smoke-test --download-data`.
