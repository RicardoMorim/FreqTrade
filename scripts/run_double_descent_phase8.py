#!/usr/bin/env python3
"""Run the predeclared training-window sensitivity experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.double_descent.benchmark import discover_cuda_python  # noqa: E402
from research.double_descent.phase8 import (  # noqa: E402
    PHASE8_MAIN_RATIOS,
    PHASE8_ROBUSTNESS_RATIOS,
    PHASE8_SEEDS,
    PHASE8_TRAINING_WINDOWS,
    Phase8Config,
    run_phase8,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test financial double descent across rolling training-window sizes."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("user_data/data/binance"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("user_data/research_results/double_descent/phase8"),
    )
    parser.add_argument(
        "--phase3-summary",
        type=Path,
        default=Path("user_data/research_results/double_descent/phase3/summary.json"),
    )
    parser.add_argument(
        "--phase5-summary",
        type=Path,
        default=Path("user_data/research_results/double_descent/phase5/summary.json"),
    )
    parser.add_argument(
        "--phase5-seed-results",
        type=Path,
        default=Path("user_data/research_results/double_descent/phase5/seed_results.csv"),
    )
    parser.add_argument("--cuda-python", type=Path)
    parser.add_argument("--timerange", default="20250101-20260101")
    parser.add_argument("--training-window", type=int, action="append", dest="training_windows")
    parser.add_argument("--ratio", type=float, action="append", dest="main_ratios")
    parser.add_argument("--robustness-ratio", type=float, action="append", dest="robustness_ratios")
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--minimum-windows", type=int, default=10)
    parser.add_argument("--minimum-seeds", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    cuda_python = (
        str(arguments.cuda_python.resolve()) if arguments.cuda_python else discover_cuda_python()
    )
    if not cuda_python:
        raise SystemExit("No CUDA-enabled Python interpreter was found")
    config = Phase8Config(
        data_directory=arguments.data_dir.resolve(),
        output_directory=arguments.output_dir,
        phase3_summary=arguments.phase3_summary.resolve(),
        phase5_summary=arguments.phase5_summary.resolve(),
        phase5_seed_results=arguments.phase5_seed_results.resolve(),
        python_executable=sys.executable,
        cuda_python_executable=cuda_python,
        timerange=arguments.timerange,
        training_windows=(
            tuple(arguments.training_windows)
            if arguments.training_windows
            else PHASE8_TRAINING_WINDOWS
        ),
        main_ratios=(tuple(arguments.main_ratios) if arguments.main_ratios else PHASE8_MAIN_RATIOS),
        robustness_ratios=(
            tuple(arguments.robustness_ratios)
            if arguments.robustness_ratios
            else PHASE8_ROBUSTNESS_RATIOS
        ),
        seeds=tuple(arguments.seeds) if arguments.seeds else PHASE8_SEEDS,
        chunk_size=arguments.chunk_size,
        fee=arguments.fee,
        minimum_training_windows=arguments.minimum_windows,
        minimum_seed_count=arguments.minimum_seeds,
        subprocess_timeout_seconds=arguments.timeout,
        resume=not arguments.no_resume,
        smoke_test=arguments.smoke_test,
    )
    summary = run_phase8(config)
    print(json.dumps(summary["training_window_assessment"], indent=2, sort_keys=True))
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    print(f"Artifacts: {arguments.output_dir.resolve()}")
    return 0 if summary["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
