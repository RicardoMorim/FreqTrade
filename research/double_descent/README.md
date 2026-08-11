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

## Phase 7: explicit Ridge regularization map

Phase 7 tests whether explicit L2 regularization suppresses the interpolation catastrophe and
whether any apparent improvement represents genuine predictive information or merely shrinkage
toward the zero-return forecast. Ridge is applied to the centered RFF coefficients while leaving
the rolling-window intercept unregularized:

```text
(K + lambda I) alpha = y
```

The grid is frozen before execution at `lambda = 0, 1e-8, 1e-6, 1e-4, 1e-2, 1, 100`. The audited
reference seed runs the complete 19-point P/N grid. Two additional predeclared seeds repeat the
critical ratios `0.10`, `1.00`, `1.02`, and `50`, producing an initial three-seed robustness check.
The `lambda=0` cases are rerun and must reproduce Phase 5 within numerical tolerance.

Implementation gates additionally require training MSE to be non-decreasing and effective degrees
of freedom, normalized coefficient norm, and regularized system condition number to be
non-increasing as lambda grows for every matched seed/P/N cell. These are solver-integrity checks,
not desired financial outcomes. The 2026 holdout remains untouched.

### Run

```powershell
python scripts/run_double_descent_phase7.py `
  --data-dir user_data\data\binance
```

The map is checkpointed after every seed/lambda substudy under
`user_data/research_results/double_descent/phase7/`.

The completed 189-case run passed every gate. `lambda=1` removed 99.995% of the interpolation-peak
MSE in all three tested seeds, and strong Ridge flattened the P/N curve. The best reference-map
cell nevertheless remained 0.096% worse than the zero-return forecast; the best three-seed critical
cell remained 0.082% worse, with zero of three seeds beating zero. Ridge therefore fixes numerical
variance by shrinking the model toward a nearly constant forecast, but does not reveal predictive
alpha or economic value. See [PHASE7_RESULTS.md](PHASE7_RESULTS.md).

## Phase 8: training-window sensitivity

Phase 8 changes the rolling training history while keeping the asset, timeframe, chronological OOS
period, 25 causal inputs, RFF construction, gamma, ridgeless float64 solver, fee, and trading rule
fixed. It uses the exact effective sample sizes measured by Phase 3:

| Training history | Effective N |
| ---: | ---: |
| 30 days | 719 |
| 60 days | 1,439 |
| 90 days | 2,159 |
| 180 days | 4,319 |
| 365 days | 8,759 |

The reference seed runs the predeclared compact grid `P/N = 0.10, 0.50, 0.90, 0.98, 1.00, 1.02,
1.10, 2, 5` for every window. The smallest and largest windows additionally repeat the critical
points `0.10, 1.00, 1.02, 5` across three seeds. `P/N=5` is already materially
overparameterized while keeping the exact 365-day dual calculation feasible; Phase 6 separately
established convergence at much larger P and at the exact kernel limit.

The predeclared shape rule requires the global OOS-MSE peak to be near `P/N=1` and at least 50%
recovery by `P/N=5`. Persistence at 365 days requires the rule in a majority of the three endpoint
seeds. This is kept separate from useful benign overfitting: the largest model must also beat both
the zero-return forecast and the best underparameterized model. All comparisons remain development
only and the 2026 holdout is untouched.

Longer histories alter both N and the mix/recency of market regimes. Phase 8 is therefore a rolling
window sensitivity test, not a pure causal intervention on sample count. The five-window trend test
is descriptive and unadjusted for multiple comparisons; it is not used as a discovery claim.

### Run

```powershell
python scripts/run_double_descent_phase8.py `
  --data-dir user_data\data\binance
```

The 61 cases are checkpointed inside each rolling FreqAI sweep and again after every window/seed
substudy. Generated artifacts are written to
`user_data/research_results/double_descent/phase8/`.

The completed run passed all gates. A near-threshold OOS-MSE peak followed by at least 98.51%
recovery at `P/N=5` appeared in every reference-seed window and in all three endpoint seeds at both
30 and 365 days. The curve shape therefore did not disappear with more training history. It also
did not become useful: at `P/N=5`, OOS MSE deteriorated from 12.26 times the zero-return baseline
at 30 days to 38.00 times at 365 days, and no tested cell beat zero. All 61 costed sign strategies
had negative return and Sharpe with profit factor below one. Strict all-window interpolation moved
from `P/N=1` at 30 days to `P/N=1.1` at 365 days, underlining the distinction between nominal P,
numerical rank, and effective dimension. See [PHASE8_RESULTS.md](PHASE8_RESULTS.md).

## Phase 9: pre-holdout gamma calibration

Phase 9 selects a reasonable RFF/RBF kernel scale without reusing the already inspected 2025
experiment or touching the sealed 2026 holdout. The calibration interval is frozen at
`2024-07-01` through `2025-01-01`; every rolling model uses the same 90-day history, 30-day
evaluation step, 25 causal inputs, ridgeless float64 CUDA solver, fee, and sign strategy as the
earlier financial phases.

The frozen candidate grid is `gamma = 0.05, 0.1, 0.2, 0.5`. Seed `20260810` maps
`P/N = 0.10, 0.50, 0.90, 0.98, 1.00, 1.02, 1.10, 2, 5, 10, 50`; seeds `1898170439` and
`3432960257` repeat `0.10, 1.00, 1.02, 5, 50`. Because each gamma rescales the same seeded
Gaussian projection draws, the comparisons remain paired rather than introducing unrelated random
feature spaces. The design contains 84 cases and at least six rolling training windows per case.

Selection is prediction-only. For each gamma, the score is the geometric mean of OOS MSE divided
by the zero-return MSE at `P/N = 0.10, 5, 50` across all three seeds. A candidate is eligible only
if the double-descent shape rule passes in at least two seeds. A challenger replaces the existing
`gamma=0.2` baseline only if it lowers the score by at least 5% and wins at least six of the nine
matched seed/ratio comparisons. Trading results are recorded as downstream diagnostics but cannot
influence gamma selection. If no challenger clears both hurdles, `gamma=0.2` remains frozen.

This is a development calibration with four candidate scales and is not confirmatory evidence of
alpha. The selected gamma is frozen for subsequent development phases; neither the 2025 main
experiment nor the final 2026 holdout is part of the selection.

### Run

```powershell
python scripts/run_double_descent_phase9.py `
  --data-dir user_data\data\binance
```

The run checkpoints after each gamma/seed substudy and writes its generated evidence under
`user_data/research_results/double_descent/phase9/`.

The completed 84-case run passed every gate and selected `gamma=0.5`. It reduced the predeclared
prediction-error score by 48.04% relative to `gamma=0.2` and won all nine matched comparisons, so
it is now frozen for subsequent development phases. This is only a relative calibration success:
0/84 models beat zero-return MSE, 0/84 produced positive OOS R2, and every costed sign strategy lost
money. All gamma values retained a robust interpolation peak and second descent across three seeds,
but none showed useful benign overfitting. See [PHASE9_RESULTS.md](PHASE9_RESULTS.md).

## Phase 10: market information versus noise features

Phase 10 tests whether the observed high-dimensional recovery requires market information or is a
generic property of fitting random designs. The experiment compares four frozen representations:

1. `market_linear`: the 25 standardized causal market inputs, used once as a low-dimensional
   anchor;
2. `market_rff`: P nested RFF of those market inputs using the Phase 9-frozen `gamma=0.5`;
3. `pure_noise`: P deterministic iid-N(0,1) predictors keyed only by timestamp, feature id, and
   seed;
4. `market_plus_noise`: the first 25 pure-noise columns are replaced by the 25 market inputs while
   columns 26 through P remain identical, keeping total predictor count P exactly matched.

Timestamp-keyed counter-based noise makes every row reproducible without using market values or
targets. Its prefixes are nested across P and independent of chunk size, so increasing complexity
adds predictors to the same random space. The noise is generated inside the streamed CUDA solver
rather than materializing hundreds of thousands of DataFrame columns.

The reference seed maps `P/N = 0.10, 0.50, 0.90, 0.98, 1.00, 1.02, 1.10, 2, 5, 10, 50` for the
three high-dimensional representations. Two further seeds repeat `0.10, 1.00, 1.02, 5, 50`.
Together with the single 25-feature market anchor, this produces 64 cases. All cases use the same
2025 rolling OOS period, 90-day training window, measured `N=2,159`, ridgeless float64 solver,
0.1% fee per side, and sign strategy. The 2026 holdout remains sealed.

The primary evidence is chronological OOS prediction error, not trading. Matched comparisons ask:

- whether market RFF beats pure noise at the same P and seed;
- whether adding 25 market inputs improves on pure noise;
- whether pure noise itself produces an interpolation peak and second descent without predictive
  alpha.

A double-descent curve in pure noise would show that curve shape alone is not evidence of market
information. Conversely, systematic pure-noise performance better than the zero-return forecast
would trigger a leakage/reproducibility investigation rather than an alpha claim. Pairwise win
counts and effect ratios are descriptive; three seeds are insufficient for a strong significance
claim, and no parameters are selected in this phase.

### Run

```powershell
python scripts/run_double_descent_phase10.py `
  --data-dir user_data\data\binance
```

The run checkpoints every case and substudy under
`user_data/research_results/double_descent/phase10/`.

The completed run passed every integrity gate: 64/64 cases and 832/832 rolling fits succeeded while
the 2026 holdout remained sealed. All three high-dimensional representations showed an
interpolation peak and second descent in every seed, but pure noise reproduced the same shape and
never beat the zero-return forecast. Market RFF beat matched pure noise in only 1/15 robust cells,
and no model among all 64 cases achieved positive OOS R2, positive net return, positive Sharpe, or
profit factor above one. Phase 10 therefore supports double-descent geometry, not useful benign
overfitting or financial alpha. See [PHASE10_RESULTS.md](PHASE10_RESULTS.md).
