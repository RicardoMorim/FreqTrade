# Phase 14 plan: frozen BTC-to-ETH replication

## Objective

Phase 14 tests whether the Phase 10 BTC findings transfer to ETH without selecting any parameter
on ETH. It is a cross-asset replication, not a new optimization exercise.

The two findings to replicate are deliberately separated:

1. **curve geometry:** an interpolation peak near `P/N=1` followed by a second descent;
2. **financial usefulness:** whether the recovered models beat zero return, simple baselines, and
   costs.

The geometry can replicate while useful benign overfitting fails to replicate.

## Frozen design

- asset: `ETH/USDT:USDT` Binance perpetual futures;
- timeframe: 1h;
- development OOS period: `2025-01-01` through `2025-12-31`;
- rolling training/evaluation: 90 days / 30 days;
- expected effective sample size: `N=2,159`, verified in every fit;
- causal market inputs and one-hour forward-return target: unchanged;
- gamma: `0.5`, frozen by Phase 9 on the pre-BTC-development period;
- RFF Ridge: zero;
- solver: float64 CUDA, `rcond=1e-12`;
- fee: 0.1% per side;
- strategy: unchanged sign strategy at 1x;
- 2026 holdout: sealed.

The reference seed evaluates all 11 Phase 10 ratios:

`0.10, 0.50, 0.90, 0.98, 1.00, 1.02, 1.10, 2, 5, 10, 50`.

Two additional seeds evaluate `0.10, 1.00, 1.02, 5, 50`. This produces 21 RFF cases, including
`P=107,950` at `P/N=50`.

The eight frozen Phase 12 references are also repeated: zero return, historical mean, OLS, Ridge
with `alpha=1`, one-hour momentum, 24-hour momentum, volatility-adjusted momentum, and
buy-and-hold. The full run contains 29 cases and 377 rolling fits.

## Integrity and interpretation

The run fails if ETH does not provide the same OOS coverage, measured N, feature dimension,
float64 CUDA execution, fee, or generated pair configuration. Resume checkpoints are enabled so a
crash does not require restarting completed cases.

The primary outcome is chronological OOS MSE normalized by the ETH zero-return MSE. Trading cannot
select or rescue a model. Shape replication requires double descent in a majority of the three ETH
seeds and in the existing BTC reference. Useful benign-overfitting replication separately requires
the extreme model to beat both zero and the best underparameterized point in a majority of seeds.

ETH is an additional asset but not a fully independent market: it shares the same dates, exchange,
crypto risk factors, and correlated regimes with BTC. A positive replication would therefore still
need broader assets or an independent time period.

## Execution

Download ETH futures data and run the full resumable experiment:

```powershell
& 'C:\Users\Ricar\Desktop\projects\freqtrade\.venv\Scripts\python.exe' `
  scripts\run_double_descent_phase14.py `
  --download-data
```

To validate the pipeline first with three RFF points, one seed, and three baselines:

```powershell
& 'C:\Users\Ricar\Desktop\projects\freqtrade\.venv\Scripts\python.exe' `
  scripts\run_double_descent_phase14.py `
  --download-data `
  --smoke-test `
  --output-dir user_data\research_results\double_descent\phase14_smoke
```

After a smoke test, run the full command without `--smoke-test`. The smoke output is isolated and
cannot be recovered into the confirmatory run. If the computer stops, rerun the same full command
without `--no-resume`; completed cases will be recovered from their artifacts.
