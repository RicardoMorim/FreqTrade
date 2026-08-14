from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from rff_core import RFFConfig, StreamingRFFRegressor


def parse_grid(value: str) -> list[int]:
    return sorted({int(item.strip()) for item in value.split(",") if item.strip()})


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the RFF core on representative N/d/P")
    parser.add_argument("--n-train", type=int, default=1400)
    parser.add_argument("--n-predict", type=int, default=168)
    parser.add_argument("--input-dim", type=int, default=30)
    parser.add_argument("--p-grid", default="256,1024,1400,4096,16384,65536")
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cuda")
    parser.add_argument(
        "--matmul-precision", choices=["highest", "high", "medium"], default="highest"
    )
    parser.add_argument("--output", type=Path, default=Path("results/rff_benchmark.csv"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    x_train = rng.normal(size=(args.n_train, args.input_dim)).astype(np.float32)
    y_train = rng.normal(size=args.n_train).astype(np.float32)
    x_predict = rng.normal(size=(args.n_predict, args.input_dim)).astype(np.float32)

    rows: list[dict[str, float | int | str]] = []
    for p in parse_grid(args.p_grid):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        config = RFFConfig(
            n_random_features=p,
            gamma=args.gamma,
            ridge_lambda=args.ridge,
            seed=args.seed,
            chunk_size=args.chunk_size,
            device=args.device,
            matmul_precision=args.matmul_precision,
        )
        model = StreamingRFFRegressor(config)

        started = time.perf_counter()
        model.fit(x_train, y_train)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        fit_seconds = time.perf_counter() - started

        started = time.perf_counter()
        model.predict(x_predict)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        predict_seconds = time.perf_counter() - started

        peak_mb = (
            torch.cuda.max_memory_allocated() / (1024**2)
            if torch.cuda.is_available()
            else 0.0
        )
        row = {
            "n_train": args.n_train,
            "n_predict": args.n_predict,
            "input_dim": args.input_dim,
            "p": p,
            "p_over_n": p / args.n_train,
            "solver_space": model.diagnostics_["solver_space"],
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "peak_cuda_mb": peak_mb,
            "matmul_precision": args.matmul_precision,
        }
        rows.append(row)
        print(
            f"P={p:8d} P/N={p / args.n_train:8.3f} "
            f"solver={row['solver_space']:6s} fit={fit_seconds:8.3f}s "
            f"predict={predict_seconds:8.3f}s peak={peak_mb:8.1f} MiB"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
