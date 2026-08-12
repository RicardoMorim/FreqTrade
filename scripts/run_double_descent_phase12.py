#!/usr/bin/env python3
"""Run the frozen Phase 12 simple-baseline comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.double_descent.phase12 import (  # noqa: E402
    PHASE12_BASELINES,
    PHASE12_RIDGE_ALPHA,
    Phase12Config,
    run_phase12,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare frozen simple baselines with the Phase 10 market RFF models."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("user_data/data/binance"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("user_data/research_results/double_descent/phase12"),
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
    parser.add_argument(
        "--phase11-summary",
        type=Path,
        default=Path("user_data/research_results/double_descent/phase11/summary.json"),
    )
    parser.add_argument("--timerange", default="20250101-20260101")
    parser.add_argument("--baseline", action="append", choices=PHASE12_BASELINES)
    parser.add_argument("--ridge-alpha", type=float, default=PHASE12_RIDGE_ALPHA)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--minimum-windows", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    config = Phase12Config(
        data_directory=arguments.data_dir.resolve(),
        output_directory=arguments.output_dir,
        phase10_summary=arguments.phase10_summary.resolve(),
        phase10_map=arguments.phase10_map.resolve(),
        phase11_summary=arguments.phase11_summary.resolve(),
        python_executable=sys.executable,
        timerange=arguments.timerange,
        baselines=tuple(arguments.baseline) if arguments.baseline else PHASE12_BASELINES,
        ridge_alpha=arguments.ridge_alpha,
        fee=arguments.fee,
        minimum_training_windows=arguments.minimum_windows,
        subprocess_timeout_seconds=arguments.timeout,
        resume=not arguments.no_resume,
        smoke_test=arguments.smoke_test,
    )
    summary = run_phase12(config)
    print(json.dumps(summary["comparisons"], indent=2, sort_keys=True))
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    print(f"Artifacts: {arguments.output_dir.resolve()}")
    return 0 if summary["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
