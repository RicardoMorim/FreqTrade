#!/usr/bin/env python3
"""Run the single-seed financial P/N sweep for Phase 4."""

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
from research.double_descent.phase4 import Phase4Config, run_phase4  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a ridgeless nested-RFF sweep through rolling FreqAI windows."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("user_data/data/binance"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("user_data/research_results/double_descent/phase4"),
    )
    parser.add_argument("--cuda-python", type=Path)
    parser.add_argument("--timerange", default="20250101-20260101")
    parser.add_argument("--ratio", type=float, action="append", dest="ratios")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--minimum-windows", type=int, default=10)
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
    ratios = tuple(arguments.ratios) if arguments.ratios else PN_RATIOS
    config = Phase4Config(
        data_directory=arguments.data_dir.resolve(),
        output_directory=arguments.output_dir,
        python_executable=sys.executable,
        cuda_python_executable=cuda_python,
        timerange=arguments.timerange,
        ratios=ratios,
        seed=arguments.seed,
        gamma=arguments.gamma,
        chunk_size=arguments.chunk_size,
        fee=arguments.fee,
        minimum_training_windows=arguments.minimum_windows,
        resume=not arguments.no_resume,
        subprocess_timeout_seconds=arguments.timeout,
    )
    summary = run_phase4(config)
    print(json.dumps(summary["curve_assessment"], indent=2, sort_keys=True))
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    print(f"Artifacts: {arguments.output_dir.resolve()}")
    return 0 if summary["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
