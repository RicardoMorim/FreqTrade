# Phase 1 Results — Synthetic Double Descent

## Decision

**PASS.** The framework reproduced an interpolation peak followed by a second descent in both
predeclared synthetic controls. This authorizes the computational benchmark phase, but it is not
evidence of financial predictability or profitability.

## Frozen run

- Date: 2026-08-10
- Git base: `develop` at `e7a9abbb8f1f0b3ad7e4c120b5180eca6c9b123f`
- Training observations: 128
- Independent test observations: 1,024
- Repetitions: 8 per dataset
- Noise standard deviation: 0.5
- RFF gamma: 0.2
- Precision: float64
- Solver: NumPy SVD-based minimum-norm least squares, `rcond=1e-12`
- Grid: 17 points from `P/N=0.10` to `P/N=8.00`, dense around 1
- Seed family: `20260810`

## Gate results

| Control | Global peak P/N | Peak OOS MSE | OOS MSE at P/N=8 | Peak recovery | Gate |
|---|---:|---:|---:|---:|---|
| Linear Gaussian | 1.00 | 811.499 | 1.286 | 99.84% | PASS |
| Nonlinear RFF | 1.00 | 626.059 | 0.663 | 99.89% | PASS |

For both controls:

- interpolation first occurred at `P/N=1.00`;
- the global OOS error maximum occurred at `P/N=1.00`;
- the far-overparameterized model recovered more than the required 50% from the peak;
- the `P/N=8` model matched or beat the best underparameterized model within the allowed 10%.

The RFF-specific kernel convergence check also passed. Mean relative Frobenius error fell from
1.663 at the smallest model to 0.179 at `P/N=8`.

## Numerical signature

The interpolation spike coincided with severe ill-conditioning rather than a hidden training-error
change:

| Control | Condition number at P/N=1 | Condition number at P/N=8 |
|---|---:|---:|
| Linear Gaussian | 411.28 | 2.05 |
| Nonlinear RFF | 2,352.31 | 14.00 |

Training MSE was effectively zero from the interpolation threshold onward. In the nonlinear RFF
control, mean OOS results changed as follows:

| P/N | OOS MSE | OOS R2 | IC | Directional accuracy |
|---:|---:|---:|---:|---:|
| 0.10 | 0.770 | 0.093 | 0.352 | 65.19% |
| 1.00 | 626.059 | -715.656 | 0.073 | 54.68% |
| 8.00 | 0.663 | 0.217 | 0.552 | 72.29% |

The direct ridgeless RBF kernel limit achieved mean OOS MSE 0.569, mean OOS R2 0.329, and mean IC
0.599. Its position beyond the finite `P/N=8` result is consistent with convergence toward the
associated kernel.

## Validation performed

- focused Pytest: 3 passed;
- Ruff: all checks passed;
- `git diff --check`: passed;
- nested-feature invariant tested directly;
- overparameterized interpolation tested directly;
- full experiment rerun after correcting the gate to inspect the global OOS maximum.

## Reproduce

```powershell
python scripts/run_double_descent_phase1.py
```

Generated artifacts are under
`user_data/research_results/double_descent/phase1/` and include detailed CSV, aggregate CSV,
machine-readable JSON, and an interactive Plotly chart. This path is ignored by Git by design.

## Interpretation boundary

This phase validates the experimental machinery only. It shows that the implementation can detect
double descent where the data-generating process permits it. It does not show that crypto returns
contain an exploitable second descent, and it does not justify selecting gamma, P, a trading rule,
or any market-data result. The next authorized step is Phase 2: benchmark time and peak memory over
increasing P before integrating the model with rolling FreqAI windows.
