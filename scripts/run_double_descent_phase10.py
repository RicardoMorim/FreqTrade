#!/usr/bin/env python3
"""Run the frozen Phase 10 market/noise feature controls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.double_descent.benchmark import discover_cuda_python  # noqa: E402
from research.double_descent.phase10 import (  # noqa: E402
    PHASE10_GAMMA,
    PHASE10_MAP_RATIOS,
    PHASE10_REPRESENTATIONS,
    PHASE10_ROBUSTNESS_RATIOS,
    PHASE10_SEEDS,
    Phase10Config,
    run_phase10,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare market, RFF, pure-noise, and market-plus-noise predictors."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("user_data/data/binance"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("user_data/research_results/double_descent/phase10"),
    )
    parser.add_argument(
        "--phase9-summary",
        type=Path,
        default=Path("user_data/research_results/double_descent/phase9/summary.json"),
    )
    parser.add_argument(
        "--frozen-gamma",
        type=Path,
        default=Path("user_data/research_results/double_descent/phase9/frozen_gamma.json"),
    )
    parser.add_argument("--cuda-python", type=Path)
    parser.add_argument("--timerange", default="20250101-20260101")
    parser.add_argument("--representation", action="append", dest="representations")
    parser.add_argument("--ratio", type=float, action="append", dest="map_ratios")
    parser.add_argument("--robustness-ratio", type=float, action="append", dest="robustness_ratios")
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--gamma", type=float, default=PHASE10_GAMMA)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--minimum-windows", type=int, default=10)
    parser.add_argument("--minimum-seeds", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=1800)
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
    config = Phase10Config(
        data_directory=arguments.data_dir.resolve(),
        output_directory=arguments.output_dir,
        phase9_summary=arguments.phase9_summary.resolve(),
        frozen_gamma_file=arguments.frozen_gamma.resolve(),
        python_executable=sys.executable,
        cuda_python_executable=cuda_python,
        timerange=arguments.timerange,
        representations=(
            tuple(arguments.representations)
            if arguments.representations
            else PHASE10_REPRESENTATIONS
        ),
        map_ratios=tuple(arguments.map_ratios) if arguments.map_ratios else PHASE10_MAP_RATIOS,
        robustness_ratios=(
            tuple(arguments.robustness_ratios)
            if arguments.robustness_ratios
            else PHASE10_ROBUSTNESS_RATIOS
        ),
        seeds=tuple(arguments.seeds) if arguments.seeds else PHASE10_SEEDS,
        gamma=arguments.gamma,
        chunk_size=arguments.chunk_size,
        fee=arguments.fee,
        minimum_training_windows=arguments.minimum_windows,
        minimum_seed_count=arguments.minimum_seeds,
        subprocess_timeout_seconds=arguments.timeout,
        resume=not arguments.no_resume,
        smoke_test=arguments.smoke_test,
    )
    summary = run_phase10(config)
    print(json.dumps(summary["representation_comparisons"], indent=2, sort_keys=True))
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    print(f"Artifacts: {arguments.output_dir.resolve()}")
    return 0 if summary["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
