# Phase 19 plan: numerical stability at interpolation

## Objective

Phase 19 tests whether the real-data interpolation peak is an unstable numerical artifact. It uses
the same frozen 25 market features, target, `gamma=0.5`, ridgeless objective, 90-day training
window, and three Phase 10 seeds. It does not select a model, alter trading, or enter the sealed
2026 holdout.

## Genuine FreqAI anchor

The model input is captured from the actual FreqAI preprocessing path rather than reconstructed
offline. The predeclared April 2025 evaluation window supplies:

- exactly 2,159 processed training observations;
- the 25 scaled causal market features;
- scaled training targets;
- processed inference features, raw realized targets, timestamps, and `do_predict` state;
- the fitted affine label inverse needed to express predictions in return units.

The capture model deliberately predicts zero and is not evaluated as a financial model. Its sole
purpose is to persist the genuine numerical inputs.

## Frozen critical matrix

The P/N grid concentrates around interpolation:

```text
0.90, 0.98, 1.00, 1.02, 1.10
```

Each point runs for all three frozen seeds. For every one of the 15 cases:

1. CUDA float64 is fitted and predicted twice;
2. CUDA float32 is fitted once as a precision stress control;
3. rank, effective rank, condition number, train error, OOS metrics, coefficient norm, runtime,
   and prediction agreement are recorded.

At `P/N=1`, every seed also runs an independent full-N CPU primal SVD using SciPy's `gesdd` driver.
The CPU and CUDA solvers use the same centered nested RFF matrix and the same matched spectral
floor. This distinguishes backend disagreement from a deliberate rank-threshold convention.

## Frozen gates

- every CUDA prediction must be finite;
- repeated CUDA float64 predictions must agree within `5e-12` maximum absolute error and retain
  identical rank;
- each CPU reference may differ by at most one retained direction from CUDA float64;
- CPU versus CUDA OOS relative L2 error must be at most `1e-5`;
- CPU versus CUDA OOS maximum absolute scaled error must be at most `1e-5`;
- all 15 critical cases and all three CPU references must complete;
- the holdout remains unused.

Float32 is not required to reproduce float64. Its machine-epsilon spectral floor is much larger,
so divergence is itself a numerical-sensitivity result. Float32 cannot replace or redefine the
frozen float64 primary experiment.

## Interpretation

The integrity gate is separate from the curve assessment. A valid phase can conclude that:

- float64 is deterministic and agrees with an independent solver;
- the local interpolation peak and recovery are or are not reproduced in the anchor window;
- float32 materially changes rank or predictions near the threshold.

Agreement between float64 backends would rule out GPU nondeterminism as the explanation for the
curve. It would not turn a statistically poor or economically losing model into useful benign
overfitting.

## Execution

```powershell
& 'C:\Users\Ricar\Desktop\projects\freqtrade\.venv\Scripts\python.exe' `
  scripts\run_double_descent_phase19.py --stage all
```

The run is resumable at case level. A filtered `--case-id` invocation remains deliberately unable
to pass the complete matrix gate.
