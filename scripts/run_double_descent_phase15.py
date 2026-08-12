#!/usr/bin/env python3
"""Prepare, benchmark, and run the frozen Phase 15 frequency study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.double_descent.benchmark import discover_cuda_python  # noqa: E402
from research.double_descent.phase15 import (  # noqa: E402
    Phase15Config,
    download_phase15_data,
    prepare_phase15,
    run_phase15,
    run_phase15_benchmark,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 15m frequency robustness on BTC, ETH, and PAXG gold proxy."
    )
    parser.add_argument(
        "--stage",
        choices=("prepare", "benchmark", "run", "all"),
        default="prepare",
        help="Preparation measures N; benchmark gates the expensive full run.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("user_data/data/binance"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("user_data/research_results/double_descent/phase15"),
    )
    parser.add_argument("--cuda-python", type=Path)
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--timeout", type=int, default=3_600)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def _outcome(summary: dict) -> dict:
    return {
        "stage": summary.get("stage"),
        "gate": summary.get("gate"),
        "effective_n": summary.get("effective_n")
        or summary.get("preparation", {}).get("effective_n"),
        "comparisons": summary.get("comparisons"),
    }


def main() -> int:
    arguments = parse_arguments()
    cuda_python = (
        str(arguments.cuda_python.resolve()) if arguments.cuda_python else discover_cuda_python()
    )
    if not cuda_python:
        raise SystemExit("No CUDA-enabled Python interpreter was found")
    config = Phase15Config(
        data_directory=arguments.data_dir.resolve(),
        output_directory=arguments.output_dir,
        python_executable=sys.executable,
        cuda_python_executable=cuda_python,
        subprocess_timeout_seconds=arguments.timeout,
        benchmark_max_case_seconds=float(arguments.timeout),
        resume=not arguments.no_resume,
        smoke_test=arguments.smoke_test,
    )
    if arguments.download_data or arguments.download_only:
        download_phase15_data(config)
    if arguments.download_only:
        print(f"Phase 15 data downloaded to: {config.data_directory}")
        return 0

    summaries = []
    if arguments.stage in {"prepare", "all"}:
        summaries.append(prepare_phase15(config))
    if arguments.stage in {"benchmark", "all"} and not arguments.smoke_test:
        summaries.append(run_phase15_benchmark(config))
    if arguments.stage in {"run", "all"}:
        summaries.append(run_phase15(config))
    final = summaries[-1]
    print(json.dumps(_outcome(final), indent=2, default=str, sort_keys=True))
    print(f"Artifacts: {arguments.output_dir.resolve()}")
    return 0 if final.get("gate", {}).get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
