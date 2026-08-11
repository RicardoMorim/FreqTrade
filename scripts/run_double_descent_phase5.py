#!/usr/bin/env python3
"""Run the initial multi-seed financial double-descent robustness study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.double_descent.benchmark import discover_cuda_python  # noqa: E402
from research.double_descent.phase3 import PN_RATIOS  # noqa: E402
from research.double_descent.phase5 import PHASE5_SEEDS, Phase5Config, run_phase5  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeat the frozen financial P/N sweep across predeclared RFF seeds."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("user_data/data/binance"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("user_data/research_results/double_descent/phase5"),
    )
    parser.add_argument(
        "--phase4-reference",
        type=Path,
        default=Path("user_data/research_results/double_descent/phase4/summary.json"),
    )
    parser.add_argument("--no-phase4-reference", action="store_true")
    parser.add_argument("--cuda-python", type=Path)
    parser.add_argument("--timerange", default="20250101-20260101")
    parser.add_argument("--ratio", type=float, action="append", dest="ratios")
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--minimum-windows", type=int, default=10)
    parser.add_argument("--minimum-seeds", type=int, default=5)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    cuda_python = (
        str(arguments.cuda_python.resolve()) if arguments.cuda_python else discover_cuda_python()
    )
    if not cuda_python:
        raise SystemExit("No CUDA-enabled Python interpreter was found")
    reference = None if arguments.no_phase4_reference else arguments.phase4_reference.resolve()
    config = Phase5Config(
        data_directory=arguments.data_dir.resolve(),
        output_directory=arguments.output_dir,
        phase4_reference_summary=reference,
        python_executable=sys.executable,
        cuda_python_executable=cuda_python,
        timerange=arguments.timerange,
        ratios=tuple(arguments.ratios) if arguments.ratios else PN_RATIOS,
        seeds=tuple(arguments.seeds) if arguments.seeds else PHASE5_SEEDS,
        gamma=arguments.gamma,
        chunk_size=arguments.chunk_size,
        fee=arguments.fee,
        minimum_training_windows=arguments.minimum_windows,
        minimum_seed_count=arguments.minimum_seeds,
        resume=not arguments.no_resume,
        subprocess_timeout_seconds=arguments.timeout,
    )
    summary = run_phase5(config)
    print(json.dumps(summary["multi_seed_assessment"], indent=2, sort_keys=True))
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    print(f"Artifacts: {arguments.output_dir.resolve()}")
    return 0 if summary["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
