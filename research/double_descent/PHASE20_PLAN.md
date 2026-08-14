# Phase 20 — Effective dimension

## Question

Does nominal RFF width `P` represent genuine statistical complexity, or do very large market
representations occupy a much smaller spectral subspace?

Phase 20 is a frozen post-processing audit. It does not fit a model, choose a parameter, inspect
the 2026 holdout, or use prediction/trading performance to pass its gate.

## Frozen sources

- Phase 10: BTC 1h at `N=2,159`, comparing market RFF, timestamp-keyed pure noise, and market plus
  noise at matched `P` and seeds (`gamma=0.5`).
- Phase 15: market RFF on BTC, ETH, and the PAXG tokenized-gold proxy at 15m, with a matched 1h
  target control, reaching `P=431,950` (`gamma=0.5`).
- Phase 6: five-seed finite-RFF convergence through `P=1,000,000` and the matched exact RBF-kernel
  limit (`gamma=0.2`). This family is never pooled with the later `gamma=0.5` family.

Only source phases whose complete integrity gates passed are admitted. Phase 19 is not a source
gate because its strict numerical audit failed one predeclared maximum-absolute CPU tolerance;
that limitation remains explicit in the final interpretation.

## Definition

For retained eigenvalues of the centered training Gram matrix,

```text
d_eff = (sum(lambda))^2 / sum(lambda^2)
```

This spectral participation ratio is compared with:

- nominal predictors `P`;
- effective training observations `N`;
- the centered rank ceiling `min(P, N - 1)`;
- retained algebraic rank where the source artifact exposes it.

The principal descriptive quantities are `d_eff/N`, `d_eff/min(P,N-1)`, `P/d_eff`, growth in
`d_eff` relative to growth in `P`, and matched representation contrasts. Confidence intervals
describe independent seeds only; one-seed cells are labelled as such.

## Integrity gate

The phase passes only if:

1. Phase 6, 10, and 15 source gates passed and their holdout flags remain sealed;
2. all sources preserve ridgeless float64 and the recorded gamma family;
3. all 173 expected diagnostic rows are present: 30 Phase 6 RFF, one exact kernel, 64 Phase 10,
   and 78 Phase 15;
4. every effective rank is finite, positive, and bounded by the centered available rank;
5. all 11 matched Phase 10 representation contrasts are present;
6. the million-feature point and BTC/ETH/Gold coverage are present;
7. no prediction, trading result, new fit, or holdout observation affects the gate.

Passing this gate establishes only that the dimensional audit is complete and internally
consistent. It does not establish return predictability, benign overfitting, or profitability.
