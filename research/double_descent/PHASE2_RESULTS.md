# Phase 2 Results: Computational Benchmark

Run date: 2026-08-10

## Decision

**PASS for the combined CPU/OpenBLAS and PyTorch/CUDA benchmark.** Both backends completed the
declared grid through `P=1,000,000`. CUDA and CPU used the same deterministic nested RFF stream and
produced matching predictions at all six shared feature counts.

This is a computational result, not evidence of financial predictability or double descent in
market data. Feasibility at the real rolling-window sample size remains conditional on Phase 3,
because the dual Gram computation scales approximately as `O(P*N^2)`.

## Environment

- OS: Windows 11
- CPU: Intel64 Family 6 Model 198, 24 physical/logical cores
- RAM: 31.42 GiB
- CPU backend: NumPy 2.4.6 with OpenBLAS
- GPU: NVIDIA GeForce RTX 5080 Laptop GPU, 16,303 MiB VRAM
- GPU backend: PyTorch 2.10.0+cu130, CUDA 13.0, compute capability 12.0
- CUDA Python: `C:\Users\Ricar\AppData\Local\Python\pythoncore-3.14-64\python.exe`
- Primary precision: float64; an additional float32 sweep was also completed
- Workload: `N_train=128`, `N_inference=256`, eight input variables, RFF gamma `0.2`
- Chunk size: 8,192 features

Each point ran in a fresh subprocess. This makes peak process RAM and cold-start wall time
comparable, but it charges PyTorch import and CUDA initialization to every GPU point. A persistent
FreqAI process should amortize that startup cost.

## Float64 measurements

| P | P/N | CPU wall (s) | CUDA wall (s) | CUDA train (s) | CUDA inference (s) | CUDA peak VRAM (MiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 32.0 | 0.209 | 2.107 | 0.206 | 0.003 | 36.9 |
| 10,000 | 78.1 | 0.280 | 2.062 | 0.196 | 0.033 | 62.4 |
| 100,000 | 781.2 | 1.231 | 2.033 | 0.219 | 0.030 | 90.9 |
| 250,000 | 1,953.1 | 2.883 | 2.065 | 0.258 | 0.044 | 93.4 |
| 500,000 | 3,906.2 | 5.398 | 2.149 | 0.314 | 0.074 | 89.5 |
| 1,000,000 | 7,812.5 | 10.653 | 2.320 | 0.439 | 0.127 | 89.8 |

At one million features, CUDA reduced isolated wall time by `4.59x`. Excluding process import and
CUDA initialization, the measured train-plus-inference work fell from 10.473 seconds on CPU to
0.565 seconds on CUDA, an `18.5x` reduction. The wall-time crossover occurs between 100,000 and
250,000 features because cold-start overhead dominates smaller GPU jobs.

The one-million-feature CUDA worker reserved 132 MiB and reached 89.8 MiB of PyTorch-tracked tensor
allocation. The maximum tracked allocation anywhere in the float64 sweep was 93.4 MiB. PyTorch's
allocator metric does not include the complete CUDA driver context; peak host RSS for a CUDA worker
was 1,109.3 MiB. Both are reported so the small tensor footprint is not mistaken for zero runtime
overhead.

## Float32 control

The complete float32 grid also passed. At `P=1,000,000`:

| Solver | Wall (s) | Train + inference (s) | Peak tracked VRAM (MiB) | OOS MSE |
|:---|---:|---:|---:|---:|
| CPU streamed dual | 4.185 | 4.003 | n/a | 0.4391919971 |
| CUDA streamed dual | 2.264 | 0.447 | 49.0 | 0.4391919374 |

Float32 is faster and uses less memory, but float64 remains the primary benchmark and is required
for later numerical-stability repetitions around the interpolation threshold.

## Numerical and resource controls

- All 16 float64 subprocesses succeeded and returned finite predictions.
- Direct primal SVD and CPU streamed dual matched at `P=4,096`.
- CUDA matched CPU at `P=4,096`, 10,000, 100,000, 250,000, 500,000, and 1,000,000.
- At one million features, both float64 backends produced OOS MSE `0.43919189097`.
- The centered design reached rank 127, as expected for 128 observations with a separate intercept.
- At one million features the condition number was `13.405` and effective rank was `55.815` on
  both backends.
- The streamed CUDA tensor allocation remained below 94 MiB versus 2.861 GiB for the full
  train-plus-inference design matrix alone.
- Peak tracked VRAM stayed far below the predeclared 90% budget.

The synthetic OOS MSE values in this phase are implementation diagnostics only. They are not part
of the Phase 1 double-descent claim and must not be treated as market evidence.

## Operational recommendation

- Use direct primal SVD around and below the interpolation threshold where accurate singular-value
  diagnostics are required.
- Use PyTorch streamed dual on CUDA for large `P`; keep CPU streamed dual as an independent
  reproducibility reference.
- Use float32 for exploratory large-P throughput, but repeat critical interpolation-region points
  in float64 before interpreting a curve.
- Treat the current one-million-feature run as a short session only for `N=128`.
- Do not extrapolate these timings to FreqAI windows until Phase 3 measures effective `N`.
- After Phase 3, benchmark representative real `N` values before freezing the final P/N grid.

Generated float64 evidence is in `user_data/research_results/double_descent/phase2/`. The additional
float32 control is in `user_data/research_results/double_descent/phase2_cuda_float32/`. Each folder
contains `summary.json`, `benchmark.csv`, and `benchmark.html`.
