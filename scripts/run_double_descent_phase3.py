#!/usr/bin/env python3
"""Run Phase 3 of the double-descent research programme."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.double_descent.phase3 import Phase3Config, run_phase3  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure the effective N produced by real rolling FreqAI training windows."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("user_data/data/binance"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("user_data/research_results/double_descent/phase3"),
    )
    parser.add_argument("--timerange", default="20250101-20260101")
    parser.add_argument("--train-period-days", type=int, action="append", dest="periods")
    parser.add_argument("--backtest-period-days", type=int, default=30)
    parser.add_argument("--minimum-windows", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=900)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    config = Phase3Config(
        data_directory=arguments.data_dir.resolve(),
        output_directory=arguments.output_dir,
        python_executable=sys.executable,
        timerange=arguments.timerange,
        train_periods_days=(
            tuple(arguments.periods) if arguments.periods else (30, 60, 90, 180, 365)
        ),
        backtest_period_days=arguments.backtest_period_days,
        minimum_windows_per_period=arguments.minimum_windows,
        subprocess_timeout_seconds=arguments.timeout,
    )
    summary = run_phase3(config)
    print(json.dumps(summary["data_audit"], indent=2, sort_keys=True))
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    print(f"Artifacts: {arguments.output_dir.resolve()}")
    return 0 if summary["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
