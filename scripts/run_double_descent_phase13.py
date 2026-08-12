#!/usr/bin/env python3
"""Run the frozen Phase 13 causal market-regime decomposition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.double_descent.phase13 import (  # noqa: E402
    FDR_ALPHA,
    HAC_LAG_HOURS,
    MINIMUM_JOINT_OBSERVATIONS,
    MINIMUM_MARGINAL_OBSERVATIONS,
    TREND_LOOKBACK_HOURS,
    TREND_SCORE_THRESHOLD,
    VOLATILITY_REFERENCE_HOURS,
    Phase13Config,
    run_phase13,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompose frozen Phase 10/12 OOS evidence by causal market regime."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("user_data/data/binance"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("user_data/research_results/double_descent/phase13"),
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
        "--phase12-summary",
        type=Path,
        default=Path("user_data/research_results/double_descent/phase12/summary.json"),
    )
    parser.add_argument("--timerange", default="20250101-20260101")
    parser.add_argument("--trend-lookback-hours", type=int, default=TREND_LOOKBACK_HOURS)
    parser.add_argument(
        "--volatility-reference-hours", type=int, default=VOLATILITY_REFERENCE_HOURS
    )
    parser.add_argument("--trend-threshold", type=float, default=TREND_SCORE_THRESHOLD)
    parser.add_argument("--hac-lag-hours", type=int, default=HAC_LAG_HOURS)
    parser.add_argument("--fdr-alpha", type=float, default=FDR_ALPHA)
    parser.add_argument(
        "--minimum-marginal-observations",
        type=int,
        default=MINIMUM_MARGINAL_OBSERVATIONS,
    )
    parser.add_argument(
        "--minimum-joint-observations", type=int, default=MINIMUM_JOINT_OBSERVATIONS
    )
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    config = Phase13Config(
        data_directory=arguments.data_dir.resolve(),
        output_directory=arguments.output_dir,
        phase10_summary=arguments.phase10_summary.resolve(),
        phase10_map=arguments.phase10_map.resolve(),
        phase12_summary=arguments.phase12_summary.resolve(),
        python_executable=sys.executable,
        timerange=arguments.timerange,
        trend_lookback_hours=arguments.trend_lookback_hours,
        volatility_reference_hours=arguments.volatility_reference_hours,
        trend_score_threshold=arguments.trend_threshold,
        hac_lag_hours=arguments.hac_lag_hours,
        fdr_alpha=arguments.fdr_alpha,
        minimum_marginal_observations=arguments.minimum_marginal_observations,
        minimum_joint_observations=arguments.minimum_joint_observations,
        smoke_test=arguments.smoke_test,
    )
    summary = run_phase13(config)
    print(json.dumps(summary["comparisons"], indent=2, sort_keys=True))
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    print(f"Artifacts: {arguments.output_dir.resolve()}")
    return 0 if summary["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
