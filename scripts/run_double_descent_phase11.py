#!/usr/bin/env python3
"""Run the frozen Phase 11 shuffled-label negative control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.double_descent.benchmark import discover_cuda_python  # noqa: E402
from research.double_descent.phase11 import (  # noqa: E402
    MAXIMUM_ABSOLUTE_TRAIN_LABEL_CORRELATION,
    PHASE11_FEATURE_SEEDS,
    PHASE11_GAMMA,
    PHASE11_LABEL_SHUFFLE_SEEDS,
    PHASE11_MAP_RATIOS,
    PHASE11_ROBUSTNESS_RATIOS,
    Phase11Config,
    run_phase11,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a timestamp-keyed shuffled-label control for the frozen market RFF."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("user_data/data/binance"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("user_data/research_results/double_descent/phase11"),
    )
    parser.add_argument(
        "--phase10-summary",
        type=Path,
        default=Path("user_data/research_results/double_descent/phase10/summary.json"),
    )
    parser.add_argument(
        "--phase10-map",
        type=Path,
        default=Path("user_data/research_results/double_descent/phase10/representation_map.csv"),
    )
    parser.add_argument("--cuda-python", type=Path)
    parser.add_argument("--timerange", default="20250101-20260101")
    parser.add_argument("--ratio", type=float, action="append", dest="map_ratios")
    parser.add_argument("--robustness-ratio", type=float, action="append", dest="robustness_ratios")
    parser.add_argument("--feature-seed", type=int, action="append", dest="feature_seeds")
    parser.add_argument("--label-seed", type=int, action="append", dest="label_shuffle_seeds")
    parser.add_argument("--gamma", type=float, default=PHASE11_GAMMA)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--minimum-windows", type=int, default=10)
    parser.add_argument("--minimum-seeds", type=int, default=3)
    parser.add_argument(
        "--maximum-label-correlation",
        type=float,
        default=MAXIMUM_ABSOLUTE_TRAIN_LABEL_CORRELATION,
    )
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
    config = Phase11Config(
        data_directory=arguments.data_dir.resolve(),
        output_directory=arguments.output_dir,
        phase10_summary=arguments.phase10_summary.resolve(),
        phase10_map=arguments.phase10_map.resolve(),
        python_executable=sys.executable,
        cuda_python_executable=cuda_python,
        timerange=arguments.timerange,
        map_ratios=tuple(arguments.map_ratios) if arguments.map_ratios else PHASE11_MAP_RATIOS,
        robustness_ratios=(
            tuple(arguments.robustness_ratios)
            if arguments.robustness_ratios
            else PHASE11_ROBUSTNESS_RATIOS
        ),
        feature_seeds=(
            tuple(arguments.feature_seeds) if arguments.feature_seeds else PHASE11_FEATURE_SEEDS
        ),
        label_shuffle_seeds=(
            tuple(arguments.label_shuffle_seeds)
            if arguments.label_shuffle_seeds
            else PHASE11_LABEL_SHUFFLE_SEEDS
        ),
        gamma=arguments.gamma,
        chunk_size=arguments.chunk_size,
        fee=arguments.fee,
        minimum_training_windows=arguments.minimum_windows,
        minimum_seed_count=arguments.minimum_seeds,
        maximum_absolute_train_label_correlation=arguments.maximum_label_correlation,
        subprocess_timeout_seconds=arguments.timeout,
        resume=not arguments.no_resume,
        smoke_test=arguments.smoke_test,
    )
    summary = run_phase11(config)
    negative = {
        key: value for key, value in summary["negative_control"].items() if key != "comparisons"
    }
    print(json.dumps(negative, indent=2, sort_keys=True))
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    print(f"Artifacts: {arguments.output_dir.resolve()}")
    return 0 if summary["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
