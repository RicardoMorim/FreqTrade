from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a synthetic/market double-descent sweep CSV")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--metric", default="test_mse")
    parser.add_argument("--output", type=Path, default=Path("results/double_descent.png"))
    parser.add_argument("--title", default="Double-descent sweep")
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    if "p_over_n" not in frame or args.metric not in frame:
        raise SystemExit(f"CSV must contain p_over_n and {args.metric}")

    grouped = frame.groupby("p_over_n")[args.metric]
    mean = grouped.mean().sort_index()
    std = grouped.std().reindex(mean.index).fillna(0.0)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(mean.index, mean.values, marker="o")
    ax.fill_between(mean.index, mean - std, mean + std, alpha=0.2)
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
