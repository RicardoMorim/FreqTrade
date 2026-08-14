# Phase 19 results: numerical stability at interpolation

## Decision

Phase 19 completed the full frozen matrix, but its strict integrity gate did **not** pass. All 15
critical cases completed, every CUDA prediction was finite, and every float64 repeat was bitwise
identical. However, one of the three independent CPU references exceeded the predeclared maximum
absolute prediction-difference tolerance. The tolerance was not changed after observing the result.

The evidence supports a nuanced conclusion:

> The float64 interpolation peak is reproducible and is not caused by CUDA nondeterminism, but its
> magnitude is numerically sensitive to precision and spectral rank truncation. It is not evidence
> of useful financial benign overfitting.

## Genuine FreqAI anchor

The capture gate passed before any numerical case ran:

- pair: `BTC/USDT:USDT`;
- timeframe: 1h;
- training period: 1 January–31 March 2025;
- effective training observations: exactly 2,159;
- input features: the frozen 25 causal market features;
- inference period: April 2025;
- inference rows: 720, with 719 finite realized targets;
- holdout use: none.

The capture came from FreqAI's real filtering and MinMax preprocessing path. The offline solver
matrix therefore received the same scaled input geometry that the rolling models use, rather than
an approximate reconstruction.

## Completed numerical matrix

Five critical P/N levels were tested for each of the three frozen Phase 10 seeds:

```text
0.90, 0.98, 1.00, 1.02, 1.10
```

Each case ran CUDA float64 twice and CUDA float32 once. The three `P/N=1` cases additionally ran a
full-N independent CPU primal SVD. Total case wall time was 19.0 seconds, of which the CPU
references used 5.5 seconds.

| P/N | P | Median float64 MSE | Median float32 MSE | Median float64 condition number | Median float64 rank | Median float32 rank |
|---:|---:|---:|---:|---:|---:|---:|
| 0.90 | 1,943 | 0.002074 | 0.00009595 | 3,127 | 1,943 | 967 |
| 0.98 | 2,116 | 0.009979 | 0.00009284 | 18,012 | 2,116 | 984 |
| 1.00 | 2,159 | 0.125235 | 0.00009089 | 434,647 | 2,158 | 987 |
| 1.02 | 2,202 | 0.012041 | 0.00009011 | 17,262 | 2,158 | 993 |
| 1.10 | 2,375 | 0.002676 | 0.00008855 | 5,062 | 2,158 | 1,004 |

## Float64 curve

The anchor window reproduced the local double-descent geometry across the three seeds:

- the median peak occurred exactly at `P/N=1`;
- the peak MSE was 60.37 times the value at `P/N=0.9`;
- by `P/N=1.1`, median MSE had fallen by a factor of 46.80 from the peak;
- median condition number at the peak was 139 times the value at `P/N=0.9`;
- all 15 repeated float64 train and OOS predictions had maximum absolute difference exactly zero.

This rules out run-to-run CUDA nondeterminism as the explanation for the frozen float64 curve. It
also confirms that the interpolation explosion coincides with severe ill-conditioning.

## Independent CPU reference and failed strict gate

At `P/N=1`, CPU primal SVD and CUDA sample-space eigendecomposition retained the same rank, 2,158,
for all three seeds. Their effective ranks and condition numbers also agreed closely.

| Seed | Float64 condition number | CPU/CUDA OOS relative L2 | CPU/CUDA maximum absolute scaled difference | Frozen max-absolute gate |
|---:|---:|---:|---:|---:|
| 20260810 | 226,665 | 6.96e-8 | 2.33e-6 | pass |
| 1898170439 | 434,647 | 1.60e-7 | 9.26e-6 | pass |
| 3432960257 | 983,137 | 5.45e-7 | 2.12e-5 | **fail** |

The failing seed still had identical rank, prediction correlation effectively equal to one, and
100% sign agreement. It passed the frozen `1e-5` relative-L2 tolerance but failed the separate
`1e-5` maximum-absolute tolerance. This is consistent with error amplification in the most
ill-conditioned seed, but the formal gate remains failed.

## Float32 is implicit regularization

Float32 did not reproduce the float64 curve:

- the float64 relative eigenvalue floor was `4.79e-13`;
- the float32 floor was `2.57e-4`;
- at `P/N=1`, median retained rank fell from 2,158 to 987;
- median condition number fell from 434,647 to about 62;
- median float32-versus-float64 OOS relative L2 difference across all cases was 0.995;
- the correlation between the two log-MSE curves was -0.253;
- float32's largest MSE occurred at `P/N=0.9`, not at interpolation.

The apparent float32 improvement is therefore not evidence that a lower-precision solver predicts
returns better. Its much larger machine-epsilon floor removes nearly singular directions and acts
as strong implicit spectral regularization. It is solving a materially different effective model.

## Baseline context

The zero-return forecast had OOS MSE `3.1137e-5` in the same anchor window.

- the best critical float64 result was 66.62 times worse than zero;
- the best critical float32 result was 2.84 times worse than zero;
- neither precision beat the zero-return baseline.

Thus the local second descent is a real feature of the frozen float64 interpolation system, but it
does not recover useful generalization. Precision-induced smoothing makes the curve look better
without creating financial predictability.

## Interpretation limits

1. The solver comparison uses one predeclared real FreqAI window. Phase 10 remains the evidence for
   the complete 2025 rolling float64 curve.
2. The expensive full-N CPU reference is restricted to `P/N=1`, the most critical point, for all
   three seeds.
3. Phase 19 tests numerical and prediction behavior only. Trading cannot pass or fail the phase.
4. The failed absolute-tolerance gate prevents a claim of backend equivalence under every frozen
   criterion, even though relative and directional agreement are very strong.
5. None of these results unseal or evaluate the 2026 final holdout.

## Reproduction

```powershell
& 'C:\Users\Ricar\Desktop\projects\freqtrade\.venv\Scripts\python.exe' `
  scripts\run_double_descent_phase19.py --stage all
```

The run is resumable and fingerprints the real captured dataset, protocol, FreqAI adapter, CUDA
worker, source summaries, seed, and P/N level.

## Artifacts

- `user_data/research_results/double_descent/phase19/summary.json`
- `user_data/research_results/double_descent/phase19/preparation.json`
- `user_data/research_results/double_descent/phase19/case_manifest.csv`
- `user_data/research_results/double_descent/phase19/capture/summary.json`
- `user_data/research_results/double_descent/phase19/capture/anchor_dataset.npz`
- `user_data/research_results/double_descent/phase19/case_results.csv`
- `user_data/research_results/double_descent/phase19/aggregate_metrics.csv`
- `user_data/research_results/double_descent/phase19/numerical_stability.html`
- `user_data/research_results/double_descent/phase19/checkpoint.json`
