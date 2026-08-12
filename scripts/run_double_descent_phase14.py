#!/usr/bin/env python3
"""Download ETH data and run the frozen Phase 14 cross-asset replication."""

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
    PHASE10_MAP_RATIOS,
    PHASE10_ROBUSTNESS_RATIOS,
    PHASE10_SEEDS,
)
from research.double_descent.phase12 import PHASE12_BASELINES  # noqa: E402
from research.double_descent.phase14 import (  # noqa: E402
    Phase14Config,
    download_eth_data,
    run_phase14,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replicate the frozen BTC double-descent experiment on ETH."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("user_data/data/binance"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("user_data/research_results/double_descent/phase14"),
    )
    parser.add_argument("--cuda-python", type=Path)
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    cuda_python = (
        str(arguments.cuda_python.resolve()) if arguments.cuda_python else discover_cuda_python()
    )
    if not cuda_python:
        raise SystemExit("No CUDA-enabled Python interpreter was found")
    smoke = arguments.smoke_test
    config = Phase14Config(
        data_directory=arguments.data_dir.resolve(),
        output_directory=arguments.output_dir,
        python_executable=sys.executable,
        cuda_python_executable=cuda_python,
        map_ratios=(0.1, 1.0, 5.0) if smoke else PHASE10_MAP_RATIOS,
        robustness_ratios=(0.1, 1.0, 5.0) if smoke else PHASE10_ROBUSTNESS_RATIOS,
        seeds=(PHASE10_SEEDS[0],) if smoke else PHASE10_SEEDS,
        baselines=("zero_return", "market_ols", "market_ridge") if smoke else PHASE12_BASELINES,
        minimum_seed_count=1 if smoke else 3,
        subprocess_timeout_seconds=arguments.timeout,
        resume=not arguments.no_resume,
        smoke_test=smoke,
    )
    if arguments.download_data or arguments.download_only:
        download_eth_data(config)
    if arguments.download_only:
        print(f"ETH data downloaded to: {config.data_directory}")
        return 0
    summary = run_phase14(config)
    outcome = {
        "shape_replication": summary["comparisons"]["shape_replication"],
        "useful_benign_overfitting_replication": summary["comparisons"][
            "useful_benign_overfitting_replication"
        ],
        "zero_baseline_comparison": summary["comparisons"]["zero_baseline_comparison"],
        "rff_complexity_justified_on_eth": summary["comparisons"][
            "rff_complexity_justified_on_eth"
        ],
    }
    print(json.dumps(outcome, indent=2, sort_keys=True))
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    print(f"Artifacts: {arguments.output_dir.resolve()}")
    return 0 if summary["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
