from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


def parse_int_grid(value: str) -> list[int]:
    return sorted({int(item.strip()) for item in value.split(",") if item.strip()})


def parse_seed_grid(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def deep_copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Freqtrade/FreqAI RFF complexity sweep")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--timerange", required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("results/freqtrade"))
    parser.add_argument(
        "--p-grid",
        default="64,128,256,512,1024,2048,4096,8192,16384,32768,65536",
    )
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--timeframe-detail", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = json.loads(args.config.read_text(encoding="utf-8"))
    p_grid = parse_int_grid(args.p_grid)
    seeds = parse_seed_grid(args.seeds)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = args.results_dir / "configs"
    temp_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.results_dir / "manifest.jsonl"

    for p in p_grid:
        for seed in seeds:
            run_name = f"rff-p{p}-seed{seed}"
            cfg = deep_copy_json(base)
            freqai = cfg.setdefault("freqai", {})
            freqai["identifier"] = f"double-descent-{run_name}"
            params = freqai.setdefault("model_training_parameters", {})
            params.update(
                {
                    "mode": "rff",
                    "n_random_features": p,
                    "gamma": args.gamma,
                    "ridge_lambda": args.ridge,
                    "seed": seed,
                    "chunk_size": args.chunk_size,
                    "device": args.device,
                    "compute_diagnostics": False,
                }
            )

            config_path = temp_dir / f"{run_name}.json"
            config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            export_path = args.results_dir / f"{run_name}.json"
            log_path = args.results_dir / f"{run_name}.log"

            cmd = [
                "freqtrade",
                "backtesting",
                "--config",
                str(config_path),
                "--strategy",
                "DoubleDescentFreqaiStrategy",
                "--freqaimodel",
                "DoubleDescentRFFRegressor",
                "--timerange",
                args.timerange,
                "--export",
                "trades",
                "--export-filename",
                str(export_path),
            ]
            if args.timeframe_detail:
                cmd.extend(["--timeframe-detail", args.timeframe_detail])

            print(" ".join(cmd))
            if args.dry_run:
                continue

            started = time.time()
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
            elapsed = time.time() - started
            manifest = {
                "run_name": run_name,
                "p": p,
                "seed": seed,
                "gamma": args.gamma,
                "ridge_lambda": args.ridge,
                "timerange": args.timerange,
                "returncode": completed.returncode,
                "elapsed_seconds": elapsed,
                "config": str(config_path),
                "export": str(export_path),
                "log": str(log_path),
            }
            with manifest_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(manifest) + "\n")
            if completed.returncode != 0:
                print(f"FAILED {run_name}; inspect {log_path}")
            else:
                print(f"DONE {run_name} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
