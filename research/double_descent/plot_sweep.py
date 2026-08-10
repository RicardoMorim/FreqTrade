from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a synthetic/market double-descent sweep CSV")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--metric", default="test_mse")
    parser.add_argument("--x-column", default=None)
    parser.add_argument("--output", type=Path, default=Path("results/double_descent.png"))
    parser.add_argument("--title", default="Double-descent sweep")
    parser.add_argument("--band", choices=["std", "ci95", "none"], default="ci95")
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    x_column = args.x_column
    if x_column is None:
        if "p_over_n" in frame:
            x_column = "p_over_n"
        elif "p_over_n_median" in frame:
            x_column = "p_over_n_median"
        else:
            raise SystemExit("CSV must contain p_over_n or p_over_n_median")
    if x_column not in frame or args.metric not in frame:
        raise SystemExit(f"CSV must contain {x_column} and {args.metric}")

    data = frame[[x_column, args.metric]].replace([np.inf, -np.inf], np.nan).dropna()
    grouped = data.groupby(x_column)[args.metric]
    mean = grouped.mean().sort_index()
    std = grouped.std().reindex(mean.index).fillna(0.0)
    count = grouped.count().reindex(mean.index).clip(lower=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(mean.index, mean.values, marker="o")
    if args.band != "none":
        spread = std if args.band == "std" else 1.96 * std / np.sqrt(count)
        ax.fill_between(mean.index, mean - spread, mean + spread, alpha=0.2)
    ax.axvline(1.0, linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("P / N")
    ax.set_ylabel(args.metric)
    ax.set_title(args.title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
