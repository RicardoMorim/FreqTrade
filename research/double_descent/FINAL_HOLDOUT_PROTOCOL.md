# Final supervised holdout protocol

## Purpose

This is the one-shot final evaluation of the frozen supervised double-descent experiment. It is
not Phase 21 and does not tune another model. The complete Phase 15 design is transferred without
selection to data beginning on 1 January 2026.

## Frozen interval

- evaluation start, inclusive: `2026-01-01 00:00 UTC`;
- evaluation end, exclusive: `2026-08-01 00:00 UTC`;
- label: **2026 YTD holdout**, not full-year 2026;
- rolling design: 90 training days followed by 30 prediction days;
- eight chronological prediction windows per case;
- later windows may train on earlier 2026 observations, but never on observations after the
  prediction timestamp.

The end date was fixed before downloading or inspecting any 2026 candle. It is the last complete
calendar month available when the protocol was frozen.

## Frozen design

- assets: BTC, ETH, and Binance PAXG perpetual as a tokenized-gold proxy;
- 15-minute candles;
- primary target: next 15-minute return;
- diagnostic target: next one-hour return;
- nested market RFF, `gamma=0.5`, ridgeless float64 solver;
- the Phase 15 P/N maps and three random seeds;
- every Phase 12 baseline;
- sign-only trading at 1x with no threshold or Hyperopt;
- 10 bps fee per side in the native run;
- every Phase 16 cost scenario applied after positions and timings are frozen.

This produces 78 RFF cases, 39 baseline cases, 936 rolling fits, 87 primary economic cases, and
522 cost-scenario rows.

## Inference

Prediction is primary. Every non-zero prediction case receives a two-sided Bartlett-HAC test of
its squared-error differential versus the zero-return forecast, using a 24-hour/96-candle lag.
Benjamini-Hochberg controls the false-discovery rate at 5% across the complete family.

Trading is downstream. PnL, Sharpe, or costs cannot make the integrity gate pass and cannot be
used to select another model after the holdout is opened.

## Numerical qualification

Phase 19 completed but failed one strict CPU maximum-absolute tolerance near `P/N=1`. The final
holdout retains those points because they are part of the frozen curve, but any interpolation
claim remains qualified by that numerical result.

## One-shot rule

The machine-readable `FINAL_HOLDOUT_PROTOCOL.json` records source and development-artifact hashes.
It must be committed before the 2026 data are downloaded. Interrupted cases may resume from the
same configuration; changing code, data, parameters, model membership, or interval after results
are visible invalidates the pristine holdout claim.
