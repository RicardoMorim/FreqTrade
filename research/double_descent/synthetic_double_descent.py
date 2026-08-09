from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


def min_norm_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    ridge_lambda: float,
    rcond: float,
) -> tuple[np.ndarray, float]:
    """Fit linear least-squares/minimum-norm regression and return predictions + train MSE."""
    n, p = x_train.shape
    if p <= n:
        if ridge_lambda > 0:
            a = x_train.T @ x_train + ridge_lambda * np.eye(p)
            beta = np.linalg.solve(a, x_train.T @ y_train)
        else:
            beta = np.linalg.pinv(x_train, rcond=rcond) @ y_train
    else:
        gram = x_train @ x_train.T
        if ridge_lambda > 0:
            alpha = np.linalg.solve(gram + ridge_lambda * np.eye(n), y_train)
        else:
            alpha = np.linalg.pinv(gram, rcond=rcond, hermitian=True) @ y_train
        beta = x_train.T @ alpha

    train_pred = x_train @ beta
    return x_test @ beta, float(np.mean((train_pred - y_train) ** 2))


def build_nested_designs(
    n_train: int,
    n_test: int,
    p_max: int,
    n_informative: int,
    noise_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_train = rng.normal(size=(n_train, p_max)).astype(np.float64)
    x_test = rng.normal(size=(n_test, p_max)).astype(np.float64)
    beta = np.zeros(p_max, dtype=np.float64)
    beta[:n_informative] = rng.normal(size=n_informative) / math.sqrt(n_informative)
    y_train = x_train @ beta + rng.normal(scale=noise_std, size=n_train)
    y_test = x_test @ beta + rng.normal(scale=noise_std, size=n_test)
    return x_train, y_train, x_test, y_test


def parse_grid(value: str) -> list[int]:
    return sorted({int(item.strip()) for item in value.split(",") if item.strip()})


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic double-descent sanity check")
    parser.add_argument("--n-train", type=int, default=600)
    parser.add_argument("--n-test", type=int, default=3000)
    parser.add_argument("--n-informative", type=int, default=30)
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--rcond", type=float, default=1e-10)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--p-grid",
        default="30,60,120,240,420,540,570,590,600,610,630,660,720,900,1200,2400,4800",
    )
    parser.add_argument("--output", type=Path, default=Path("results/synthetic.csv"))
    args = parser.parse_args()

    p_grid = parse_grid(args.p_grid)
    if not p_grid:
        raise SystemExit("p-grid cannot be empty")
    if max(p_grid) < args.n_informative:
        raise SystemExit("p-grid must contain at least n_informative predictors")

    rows: list[dict[str, float | int]] = []
    for seed in range(args.seeds):
        x_train, y_train, x_test, y_test = build_nested_designs(
            n_train=args.n_train,
            n_test=args.n_test,
            p_max=max(p_grid),
            n_informative=args.n_informative,
            noise_std=args.noise_std,
            seed=seed,
        )
        for p in p_grid:
            pred, train_mse = min_norm_predict(
                x_train[:, :p], y_train, x_test[:, :p], args.ridge, args.rcond
            )
            test_mse = float(np.mean((pred - y_test) ** 2))
            rows.append(
                {
                    "seed": seed,
                    "n_train": args.n_train,
                    "p": p,
                    "p_over_n": p / args.n_train,
                    "train_mse": train_mse,
                    "test_mse": test_mse,
                }
            )
            print(
                f"seed={seed:02d} p={p:6d} p/n={p / args.n_train:7.3f} "
                f"train_mse={train_mse:.5f} test_mse={test_mse:.5f}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
