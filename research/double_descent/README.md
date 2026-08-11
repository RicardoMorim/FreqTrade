# Double Descent in Financial Return Prediction

This directory contains the reproducible experimental code for studying double descent and benign
overfitting in return prediction. Trading performance is downstream evidence; prediction quality is
the primary object of study.

## Phase 1: synthetic validation

Phase 1 is a gate. No market-data experiment should rely on this framework unless it can first
recover double descent under controlled conditions.

Two independent controls are run:

1. `linear_gaussian`: a high-dimensional Gaussian teacher/student regression problem.
2. `nonlinear_rff`: a nonlinear teacher fitted with nested Random Fourier Features.

Both use fixed train/test draws, repeated seeds, an unpenalized intercept, and the ridgeless
minimum-norm solution computed in float64. For each requested complexity the experiment records:

- nominal and actual `P/N`;
- training MSE and OOS MSE/MAE/R2;
- OOS information coefficient and directional accuracy;
- matrix rank, effective rank, condition number, and coefficient norm;
- whether training interpolation occurred;
- fit time;
- RFF-to-RBF-kernel approximation error.

The RFF mapping is prefix-stable: increasing `P` appends projections without changing any earlier
projection. `P` is the number of fitted feature coefficients plus the intercept. The empirical
interpolation threshold is validated from training error and matrix rank rather than assumed from
the requested grid.

### Predeclared gate

Each synthetic control must satisfy all of the following:

1. interpolation starts inside `0.90 <= P/N <= 1.10`;
2. the worst mean OOS MSE in that band is the interpolation peak;
3. OOS MSE at the largest tested ratio recovers at least 50% of the peak deterioration;
4. the largest model matches the best underparameterized OOS MSE within 10%;
5. for RFF, kernel approximation error falls as `P` grows.

These rules are evaluated automatically and the process exits non-zero on failure.

### Run

From the repository root:

```powershell
python scripts/run_double_descent_phase1.py
```

The default artifacts are written to:

```text
user_data/research_results/double_descent/phase1/
```

That generated directory is intentionally ignored by Git. The versioned source, configuration,
tests, and gate definition are sufficient to reproduce it.

### Scope

Passing Phase 1 establishes only that the machinery can detect a known synthetic effect. It is not
evidence that financial returns exhibit double descent, predictability, or profitability. Phase 2
benchmarks computational limits; Phase 3 measures the real rolling-window sample size in FreqAI.

## Phase 2: computational benchmark

Phase 2 benchmarks nested RFF generation, minimum-norm fitting, inference, and peak process RAM on
the available machine. Every case runs in a fresh subprocess so the parent can sample actual
process-tree resident memory rather than relying only on theoretical array sizes.

Three solver paths are compared:

1. `primal_svd` materializes the design matrix and directly solves the least-squares problem;
2. `streamed_dual` builds the sample-space Gram matrix in deterministic feature chunks, then makes
   a second streamed pass for inference on CPU;
3. `torch_cuda_dual` performs the same streamed algorithm on a CUDA GPU while preserving the exact
   NumPy-generated nested RFF parameters used by the CPU reference.

The dual path has bounded feature memory and is intended for `P >> N`. A shared case at `P=4096`
checks that both implementations produce the same predictions. The default grid then extends the
streamed path to one million features. This benchmark uses a fixed synthetic workload with
`N_train=128`; Phase 3 must measure real FreqAI `N` before these timings are extrapolated to market
experiments, because dual computation grows quadratically with `N`.

### Predeclared gate

The benchmark passes only if:

1. every isolated subprocess succeeds and returns finite predictions;
2. the largest requested feature count completes;
3. peak process RAM stays below 50% of installed memory;
4. primal and streamed-dual predictions agree in their overlap case;
5. streamed peak RAM is below the memory required to materialize the largest full design matrix.

When CUDA cases are enabled, it additionally requires a verified CUDA backend, CPU-equivalent
predictions, measured VRAM below 90% of the device, and a streamed CUDA footprint below the full
design-matrix size.

### Run

From the repository root:

```powershell
python scripts/run_double_descent_phase2.py
```

Artifacts are written to `user_data/research_results/double_descent/phase2/`. The command exits
non-zero when the gate fails. The runner discovers a CUDA-enabled `python3` installation and adds
the GPU grid automatically. Use `--cuda-python PATH` to select one explicitly or `--no-cuda` for a
CPU-only reproduction. A physically present GPU is never reported as benchmarked unless a CUDA
worker actually completes and exposes tracked VRAM.

## Phase 3: effective rolling-window sample size

Phase 3 runs a measurement-only strategy through the genuine FreqAI rolling-training path on
Binance BTC/USDT perpetual futures. It freezes a causal 25-variable market-state vector and records
the raw and final matrix dimensions passed to a dummy mean regressor. No trading signals are
generated and no predictive-performance claim is made.

The completed 2025 development run observed 13 windows at each requested history length:

| Training period | Effective N |
| ---: | ---: |
| 30 days | 719 |
| 60 days | 1,439 |
| 90 days | 2,159 |
| 180 days | 4,319 |
| 365 days | 8,759 |

All Phase 3 gates passed, including real-data coverage, subprocess success, finite matrices,
stable feature dimension, and unique chronological windows. See [PHASE3_RESULTS.md](PHASE3_RESULTS.md)
for the audit and the P/N grids derived from these measurements.

### Run

```powershell
python scripts/run_double_descent_phase3.py `
  --data-dir user_data\data\binance
```

Artifacts are written to `user_data/research_results/double_descent/phase3/`. The command exits
non-zero if data are absent, a Freqtrade subprocess does not produce measurements, or any sample
integrity gate fails.

## Phase 4: financial P/N sweep

Phase 4 fits deterministic nested RFF models through the real rolling FreqAI path using the
90-day effective sample size measured in Phase 3 (`N=2,159`). A persistent external CUDA worker
streams the RFF representation, solves the centered minimum-norm problem in sample space, and
returns predictions to Freqtrade for chronological OOS evaluation and a costed sign strategy.

The first development sweep uses one frozen seed, gamma `0.2`, ridge `0`, float64, and 19 P/N
points from 0.10 through 50. The integrity gate passed. MSE peaked at exactly `P/N=1` in all 13
rolling windows and recovered by 99.94% at the largest model. However, every RFF model had negative
OOS R2, the largest model was still 14.08 times worse than the zero-return forecast, and every
costed backtest lost heavily. This is a preliminary double-descent shape, not evidence of useful
benign overfitting, alpha, or profitability. See [PHASE4_RESULTS.md](PHASE4_RESULTS.md).

### Run

```powershell
python scripts/run_double_descent_phase4.py `
  --data-dir user_data\data\binance
```

The runner checkpoints after every ratio and validates complete existing artifacts before
resuming. Generated results are written to
`user_data/research_results/double_descent/phase4/`.

## Phase 5: multiple random seeds

Phase 5 repeats the frozen Phase 4 experiment across five predeclared RFF projections. The Phase 4
seed is reused only after its complete summary and configuration pass a compatibility audit. Four
additional independent seeds are derived deterministically from `numpy.random.SeedSequence(20260810)`;
no result is inspected when choosing them.

For every P/N point, Phase 5 reports the mean, median, sample standard deviation, range, and a 95%
Student-t interval across seeds. It separately evaluates whether the interpolation spike and second
descent are reproducible, whether extreme models beat the zero-return and underparameterized
baselines, and whether prediction or trading evidence is present. A repeatable curve shape alone is
not labelled benign overfitting.

### Run

```powershell
python scripts/run_double_descent_phase5.py `
  --data-dir user_data\data\binance
```

The default run performs 76 new rolling FreqAI backtests and audits the 19 Phase 4 reference cases.
It checkpoints at seed and P/N level. Generated artifacts are written to
`user_data/research_results/double_descent/phase5/`.

The completed initial run passed all gates. The interpolation peak occurred at `P/N=1` in all five
seeds and mean recovery by `P/N=50` was 99.929%. This robust shape is not useful benign overfitting:
the extreme model remained 13.80 times worse than the zero-return forecast, no RFF model beat zero
in any seed, and all costed strategies failed. See [PHASE5_RESULTS.md](PHASE5_RESULTS.md).

## Phase 6: exact RBF-kernel limit

Phase 6 computes the exact centered RBF kernel corresponding to the frozen RFF representation and
runs it through the same rolling FreqAI path. It compares the exact `P=infinity` prediction vector
with all five RFF seeds at approximately 10K, 54K, 108K, 250K, 500K, and one million features.

Convergence is evaluated directly from matched OOS predictions rather than inferred from similar
headline MSE values. The predeclared practical criterion requires both mean relative prediction
error and relative OOS-MSE distance to the exact kernel to be at most 10% at one million features.
Passing that numerical criterion is kept separate from beating simple predictive and economic
baselines.

### Run

```powershell
python scripts/run_double_descent_phase6.py `
  --data-dir user_data\data\binance
```

The large finite-RFF cases are streamed on CUDA and checkpointed after every feature count and seed.
Generated artifacts are written to `user_data/research_results/double_descent/phase6/`.

The completed run passed every gate. Prediction distance to the exact kernel fell monotonically at
an estimated `P^-0.509` rate. One million features passed the practical convergence criterion with
7.33% relative prediction error and an OOS-MSE gap of 0.84%. The exact kernel nevertheless remained
12.99 times worse than the zero-return forecast and lost 93.44% in the costed sign strategy. The
finite RFF results therefore converge correctly toward a bad predictive limit. See
[PHASE6_RESULTS.md](PHASE6_RESULTS.md).
