from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def parse_int_grid(value: str) -> list[int]:
    return sorted({int(item.strip()) for item in value.split(",") if item.strip()})


def parse_seed_grid(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def deep_copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in value).strip("-")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def collect_oos_metrics(model_root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    if model_root.exists():
        for path in model_root.rglob("double_descent_oos.jsonl"):
            records.extend(read_jsonl(path))

    records = [row for row in records if int(row.get("n_oos", 0)) > 0]
    if not records:
        return {"oos_windows": 0, "oos_n": 0}

    n = sum(int(row["n_oos"]) for row in records)
    sse = sum(float(row["sse"]) for row in records)
    sae = sum(float(row["sae"]) for row in records)
    sum_y = sum(float(row["sum_y"]) for row in records)
    sum_y2 = sum(float(row["sum_y2"]) for row in records)
    sum_p = sum(float(row["sum_pred"]) for row in records)
    sum_p2 = sum(float(row["sum_pred2"]) for row in records)
    sum_py = sum(float(row["sum_pred_y"]) for row in records)
    direction_correct = sum(int(row["direction_correct"]) for row in records)

    sst = sum_y2 - sum_y * sum_y / n
    r2 = 1.0 - sse / sst if sst > 0 else math.nan
    cov = sum_py - sum_p * sum_y / n
    var_p = sum_p2 - sum_p * sum_p / n
    var_y = sst
    ic = cov / math.sqrt(var_p * var_y) if var_p > 0 and var_y > 0 else math.nan

    n_train = [int(row["n_samples"]) for row in records if row.get("n_samples") is not None]
    p_over_n = [
        float(row["p_over_n"])
        for row in records
        if isinstance(row.get("p_over_n"), int | float)
    ]
    n_train_sorted = sorted(n_train)
    pon_sorted = sorted(p_over_n)

    def median(values: list[float | int]) -> float:
        if not values:
            return math.nan
        mid = len(values) // 2
        if len(values) % 2:
            return float(values[mid])
        return (float(values[mid - 1]) + float(values[mid])) / 2.0

    return {
        "oos_windows": len(records),
        "oos_n": n,
        "oos_mse": sse / n,
        "oos_mae": sae / n,
        "oos_r2": r2,
        "oos_ic": ic,
        "oos_directional_accuracy": direction_correct / n,
        "n_train_min": min(n_train) if n_train else None,
        "n_train_median": median(n_train_sorted),
        "n_train_max": max(n_train) if n_train else None,
        "p_over_n_median": median(pon_sorted),
    }


def load_successful_runs(summary_path: Path) -> set[str]:
    if not summary_path.exists():
        return set()
    with summary_path.open("r", newline="", encoding="utf-8") as fh:
        return {
            row["run_name"]
            for row in csv.DictReader(fh)
            if row.get("returncode") == "0" and row.get("run_name")
        }


def append_summary(path: Path, row: dict[str, Any]) -> None:
    fields = [
        "experiment_id",
        "run_name",
        "p",
        "seed",
        "gamma",
        "ridge_lambda",
        "train_period_days",
        "backtest_period_days",
        "timerange",
        "returncode",
        "elapsed_seconds",
        "oos_windows",
        "oos_n",
        "oos_mse",
        "oos_mae",
        "oos_r2",
        "oos_ic",
        "oos_directional_accuracy",
        "n_train_min",
        "n_train_median",
        "n_train_max",
        "p_over_n_median",
        "identifier",
        "config",
        "export",
        "log",
    ]
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


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
    parser.add_argument("--feature-dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--accumulator-dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument(
        "--matmul-precision", choices=["highest", "high", "medium"], default="highest"
    )
    parser.add_argument("--pinv-rtol", type=float, default=1e-6)
    parser.add_argument("--train-period-days", type=int, default=None)
    parser.add_argument("--backtest-period-days", type=int, default=None)
    parser.add_argument("--timeframe-detail", default=None)
    parser.add_argument("--fee", type=float, default=None)
    parser.add_argument("--user-data-dir", type=Path, default=Path("user_data"))
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Stable run id. Reuse the same value together with --resume after interruption.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    experiment_id = safe_id(
        args.experiment_id or datetime.now(UTC).strftime("dd-%Y%m%d-%H%M%S")
    )
    base = json.loads(args.config.read_text(encoding="utf-8"))
    p_grid = parse_int_grid(args.p_grid)
    seeds = parse_seed_grid(args.seeds)
    if not p_grid or not seeds:
        raise SystemExit("p-grid and seeds must not be empty")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = args.results_dir / "configs"
    temp_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.results_dir / "manifest.jsonl"
    summary_path = args.results_dir / "summary.csv"
    (args.results_dir / "experiment_id.txt").write_text(experiment_id + "\n", encoding="utf-8")
    successful = load_successful_runs(summary_path) if args.resume else set()

    print(f"experiment_id={experiment_id}")
    for p in p_grid:
        for seed in seeds:
            run_name = f"{experiment_id}-rff-p{p}-seed{seed}"
            if run_name in successful:
                print(f"SKIP {run_name} (already successful)")
                continue

            cfg = deep_copy_json(base)
            freqai = cfg.setdefault("freqai", {})
            identifier = f"double-descent-{run_name}"
            freqai["identifier"] = identifier
            if args.train_period_days is not None:
                freqai["train_period_days"] = args.train_period_days
            if args.backtest_period_days is not None:
                freqai["backtest_period_days"] = args.backtest_period_days

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
                    "feature_dtype": args.feature_dtype,
                    "accumulator_dtype": args.accumulator_dtype,
                    "matmul_precision": args.matmul_precision,
                    "pinv_rtol": args.pinv_rtol,
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
            if args.fee is not None:
                cmd.extend(["--fee", str(args.fee)])

            print(" ".join(cmd))
            if args.dry_run:
                continue

            started = time.perf_counter()
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
            elapsed = time.perf_counter() - started
            model_root = args.user_data_dir / "models" / identifier
            oos = collect_oos_metrics(model_root)

            manifest = {
                "experiment_id": experiment_id,
                "run_name": run_name,
                "p": p,
                "seed": seed,
                "gamma": args.gamma,
                "ridge_lambda": args.ridge,
                "train_period_days": freqai.get("train_period_days"),
                "backtest_period_days": freqai.get("backtest_period_days"),
                "timerange": args.timerange,
                "returncode": completed.returncode,
                "elapsed_seconds": elapsed,
                "identifier": identifier,
                "config": str(config_path),
                "export": str(export_path),
                "log": str(log_path),
                **oos,
            }
            with manifest_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(manifest) + "\n")
            append_summary(summary_path, manifest)

            if completed.returncode != 0:
                print(f"FAILED {run_name}; inspect {log_path}")
                if not args.continue_on_error:
                    raise SystemExit(completed.returncode)
            else:
                extra = ""
                if oos.get("oos_n"):
                    extra = (
                        f" OOS mse={oos.get('oos_mse', math.nan):.6g}"
                        f" r2={oos.get('oos_r2', math.nan):.4g}"
                        f" ic={oos.get('oos_ic', math.nan):.4g}"
                        f" median(P/N)={oos.get('p_over_n_median', math.nan):.3f}"
                    )
                print(f"DONE {run_name} in {elapsed:.1f}s.{extra}")

    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
