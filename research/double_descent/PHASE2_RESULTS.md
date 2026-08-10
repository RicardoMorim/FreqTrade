# Phase 2 Results: Computational Benchmark

Run date: 2026-08-10

## Decision

**PASS for the CPU/OpenBLAS benchmark.** The streamed-dual implementation completed the declared
grid through `P=1,000,000`, kept peak process RAM below 130 MiB, and agreed numerically with the
direct primal SVD in the overlap case.

This is a computational result, not evidence of financial predictability or double descent in
market data. Feasibility at the real rolling-window sample size remains conditional on Phase 3,
because the dual Gram computation scales approximately as `O(P*N^2)`.

## Environment

- OS: Windows 11
- CPU: Intel64 Family 6 Model 198, 24 physical/logical cores
- RAM: 31.42 GiB total; 12.29 GiB available at benchmark start
- Numerical backend: NumPy 2.4.6 with CPU OpenBLAS
- GPU detected: NVIDIA GeForce RTX 5080 Laptop GPU, 16,303 MiB VRAM
- GPU computation: not run; PyTorch is CPU-only and CuPy is absent
- Precision: float64
- Workload: `N_train=128`, `N_inference=256`, eight input variables, RFF gamma `0.2`
- Chunk size: 8,192 features

VRAM is deliberately recorded as not measured rather than zero. A separate GPU benchmark requires
a CUDA-enabled Python environment and must not be inferred from the CPU result.

## Measurements

| P | P/N | Solver | Wall (s) | Train (s) | Inference (s) | Peak RSS (MiB) | Full design (GiB) |
|---:|---:|:---|---:|---:|---:|---:|---:|
| 64 | 0.5 | primal SVD | 0.180 | 0.001 | <0.001 | 42.9 | <0.001 |
| 128 | 1.0 | primal SVD | 0.180 | 0.003 | <0.001 | 43.1 | <0.001 |
| 512 | 4.0 | primal SVD | 0.179 | 0.011 | <0.001 | 47.2 | 0.001 |
| 4,096 | 32.0 | primal SVD | 0.241 | 0.061 | 0.002 | 76.1 | 0.012 |
| 4,096 | 32.0 | streamed dual | 0.224 | 0.015 | 0.030 | 64.6 | 0.012 |
| 10,000 | 78.1 | streamed dual | 0.309 | 0.034 | 0.100 | 79.9 | 0.029 |
| 100,000 | 781.2 | streamed dual | 1.272 | 0.291 | 0.809 | 126.6 | 0.286 |
| 250,000 | 1,953.1 | streamed dual | 2.769 | 0.717 | 1.885 | 129.8 | 0.715 |
| 500,000 | 3,906.2 | streamed dual | 5.370 | 1.425 | 3.771 | 125.5 | 1.431 |
| 1,000,000 | 7,812.5 | streamed dual | 10.719 | 2.904 | 7.621 | 125.9 | 2.861 |

The small workload is dominated by process startup at low `P`. From 100,000 to one million
features, wall time is approximately linear in `P`, while RSS remains flat because the feature
dimension is streamed.

## Numerical controls

- All ten subprocesses succeeded.
- All predictions were finite.
- At `P=4096`, primal and dual OOS predictions matched within the predeclared `1e-6` tolerance.
- Both solvers produced OOS MSE `0.4357920339` in that overlap case.
- The centered design reached rank 127, as expected for 128 observations with a separate intercept.
- At one million features the condition number was `13.405` and effective rank was `55.815`.
- The one-million-feature streamed process used 125.9 MiB RSS versus 2.861 GiB for the full
  train-plus-inference design matrix alone.

The synthetic OOS MSE values in this phase are implementation diagnostics only. They are not part
of the Phase 1 double-descent claim and must not be treated as market evidence.

## Operational recommendation

- Use direct primal SVD around and below the interpolation threshold where accurate singular-value
  diagnostics are required.
- Use streamed dual for `P >> N`; it makes million-feature prefix experiments practical without
  storing an `N x P` matrix.
- Treat the current one-million-feature run as a short session only for `N=128`.
- Do not extrapolate its 10.7-second runtime to FreqAI windows until Phase 3 measures effective `N`.
- After Phase 3, benchmark representative real `N` values before committing to the final P/N grid.
- Reserve GPU enablement for a clean CUDA environment; it is not a blocker for Phase 3.

Generated evidence is in `user_data/research_results/double_descent/phase2/`:

- `summary.json`: configuration, hardware inventory, gate, and raw measurements;
- `benchmark.csv`: one row per solver/feature-count case;
- `benchmark.html`: interactive timing, memory, and throughput plots.
