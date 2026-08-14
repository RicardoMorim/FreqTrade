# Phase 20 — Effective-dimension results

## Outcome

Phase 20 completed and passed every integrity gate. It consumed 173 frozen diagnostic rows from
Phases 6, 10, and 15, performed no new model fit, did not inspect the 2026 holdout, and did not use
prediction or trading performance to determine success.

The main result is strong: nominal predictor count is a poor proxy for the statistical dimension
of the market RFF representations. Hundreds of thousands, and even one million, nominal RFF
collapse to only tens of effective spectral dimensions.

## What effective rank means here

For retained eigenvalues of each centered training Gram matrix,

```text
d_eff = (sum(lambda))^2 / sum(lambda^2)
```

This is a spectral participation ratio. Algebraic rank asks how many directions are retained;
effective rank asks how evenly the spectrum is distributed over those directions. A model may
therefore have enough algebraic rank to interpolate while most of its spectral mass remains
concentrated in a much smaller subspace.

It is not an estimate of the number of causal market factors and it is not evidence of alpha.

## Same P and N, radically different representations

The Phase 10 control is the cleanest comparison because market RFF, pure noise, and market plus
noise use the same `N=2,159`, the same nominal `P`, the same seeds, and the same float64 ridgeless
solver.

At `P/N=50` (`P=107,950`):

| Representation | Effective rank | `d_eff/N` | `P/d_eff` |
|---|---:|---:|---:|
| Market RFF | 75.498 | 3.497% | 1,429.8x |
| Pure noise | 2,115.698 | 97.994% | 51.0x |
| Market plus noise | 2,115.690 | 97.994% | 51.0x |

Market RFF use only 3.57% of the effective dimension reached by matched pure noise. Once noise is
added to the market vector, the representation becomes almost indistinguishable from pure noise
in effective-rank terms.

Across the full Phase 10 width increase, nominal `P` grows about 500 times. Market-RFF effective
rank grows only 1.38 times, from 54.702 to 75.498, while pure-noise effective rank grows 10.78
times and nearly fills the centered sample space.

This shows that the contrast is caused by representation structure, not nominal width alone.

## Cross-asset and frequency replication

Phase 15 increases effective training size to approximately `N=8,639` and reaches `P=431,950`.
The primary next-15m return experiment gives:

| Asset | Effective rank | `d_eff/N` | `P/d_eff` |
|---|---:|---:|---:|
| BTC | 64.682 | 0.749% | 6,678.1x |
| ETH | 59.344 | 0.687% | 7,278.8x |
| PAXG gold proxy | 64.225 | 0.743% | 6,725.6x |

From `P=864` to `P=431,950`, nominal width grows about 500 times. Effective rank grows only 6.4%
to 7.5% across the three assets. The log elasticity of effective rank with respect to nominal `P`
is only 0.010-0.012. The matched one-hour target controls produce almost identical dimensional
results, so this saturation is not specific to the 15-minute prediction horizon.

## One million predictors and the kernel limit

The Phase 6 `gamma=0.2` family provides the largest explicit model and its matched exact RBF
kernel. This family is kept separate from the later frozen `gamma=0.5` experiments.

- `P=1,000,000` RFF: effective rank 23.761;
- exact RBF kernel: effective rank 23.748;
- relative effective-rank gap: 0.052%;
- nominal predictors per effective dimension at one million RFF: 42,086x.

Nominal `P` grows 92.6 times from 10,795 to one million within this convergence extension, while
effective rank grows just 0.16%. The finite representation has converged to the kernel's spectral
dimension even though its nominal predictor count continues to grow.

## Scientific interpretation

The experiment distinguishes three notions that must not be conflated:

```text
nominal width P != retained algebraic rank != spectral effective rank
```

Large nominal `P/N` correctly describes how the ridgeless model crosses an interpolation
boundary, but it substantially overstates the number of statistically important directions in the
market representation. The market RFF spectrum is highly redundant and saturates rapidly. In
contrast, independent noise spreads energy across nearly the whole sample space.

This helps explain why a visible interpolation peak and second descent do not by themselves imply
useful benign overfitting. Earlier phases already found that these curves did not beat the
zero-return forecast and did not produce profitable trading after costs. Phase 20 adds that the
extreme nominal widths are not extreme in effective spectral dimension.

## Limitations

- Source artifacts expose mean effective rank across rolling windows, not every window-level
  spectrum, so Phase 20 cannot reconstruct within-window distributions.
- Effective rank is descriptive and depends on the feature map, gamma, preprocessing, centering,
  and numerical retention threshold.
- The exact-kernel anchor is valid only inside the Phase 6 `gamma=0.2` family. It is not an exact
  kernel substitute for Phase 10 or 15 at `gamma=0.5`.
- PAXG is a tokenized-gold proxy and has shorter coverage than BTC and ETH.
- Phase 19 failed one strict CPU maximum-absolute tolerance near `P/N=1`. Phase 20 does not erase
  that result or use numerical diagnostics to claim alpha.

## Decision

Phase 20 supports the statement that nominal overparameterization in this experiment is largely
spectrally redundant. It does not support financial predictability, profitability, or deployment.
The 2026 holdout remains sealed.

The complete machine-readable outputs and interactive chart are in
`user_data/research_results/double_descent/phase20/`.
