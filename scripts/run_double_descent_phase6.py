#!/usr/bin/env python3
"""Run the exact RBF-kernel limit and finite-RFF convergence study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.double_descent.benchmark import discover_cuda_python  # noqa: E402
from research.double_descent.phase6 import (  # noqa: E402
    EXTENSION_FEATURE_COUNTS,
    Phase6Config,
    run_phase6,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare nested RFF models with their exact centered RBF-kernel limit."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("user_data/data/binance"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("user_data/research_results/double_descent/phase6"),
    )
    parser.add_argument(
        "--phase5-summary",
        type=Path,
        default=Path("user_data/research_results/double_descent/phase5/summary.json"),
    )
    parser.add_argument("--cuda-python", type=Path)
    parser.add_argument("--timerange", default="20250101-20260101")
    parser.add_argument(
        "--extension-feature-count",
        type=int,
        action="append",
        dest="extension_feature_counts",
    )
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    cuda_python = (
        str(arguments.cuda_python.resolve()) if arguments.cuda_python else discover_cuda_python()
    )
    if not cuda_python:
        raise SystemExit("No CUDA-enabled Python interpreter was found")
    extension_counts = (
        tuple(arguments.extension_feature_counts)
        if arguments.extension_feature_counts
        else EXTENSION_FEATURE_COUNTS
    )
    config = Phase6Config(
        data_directory=arguments.data_dir.resolve(),
        output_directory=arguments.output_dir,
        phase5_summary=arguments.phase5_summary.resolve(),
        python_executable=sys.executable,
        cuda_python_executable=cuda_python,
        timerange=arguments.timerange,
        extension_feature_counts=extension_counts,
        chunk_size=arguments.chunk_size,
        subprocess_timeout_seconds=arguments.timeout,
        resume=not arguments.no_resume,
    )
    summary = run_phase6(config)
    print(json.dumps(summary["kernel_limit_assessment"], indent=2, sort_keys=True))
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    print(f"Artifacts: {arguments.output_dir.resolve()}")
    return 0 if summary["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
