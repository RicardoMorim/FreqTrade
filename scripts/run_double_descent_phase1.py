#!/usr/bin/env python3
"""Run Phase 1 of the double-descent research programme."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.double_descent.synthetic import (  # noqa: E402
    DEFAULT_RATIOS,
    Phase1Config,
    run_phase1,
)


def _parse_ratios(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ratios must be comma-separated numbers") from exc


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate double-descent detection on synthetic linear and RFF problems."
    )
    parser.add_argument("--n-train", type=int, default=128)
    parser.add_argument("--n-test", type=int, default=1024)
    parser.add_argument("--repetitions", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--noise-standard-deviation", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument(
        "--ratios",
        type=_parse_ratios,
        default=DEFAULT_RATIOS,
        help="Comma-separated P/N grid.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("user_data/research_results/double_descent/phase1"),
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    config = Phase1Config(
        n_train=arguments.n_train,
        n_test=arguments.n_test,
        repetitions=arguments.repetitions,
        base_seed=arguments.seed,
        noise_standard_deviation=arguments.noise_standard_deviation,
        gamma=arguments.gamma,
        ratios=arguments.ratios,
    )
    summary = run_phase1(config, arguments.output_dir)
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    print(f"Artifacts: {arguments.output_dir.resolve()}")
    return 0 if summary["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
