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
