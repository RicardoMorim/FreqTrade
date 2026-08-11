#!/usr/bin/env python3
"""Run the predeclared pre-2025 gamma calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.double_descent.benchmark import discover_cuda_python  # noqa: E402
from research.double_descent.phase9 import (  # noqa: E402
    PHASE9_CALIBRATION_TIMERANGE,
    PHASE9_GAMMAS,
    PHASE9_MAP_RATIOS,
    PHASE9_ROBUSTNESS_RATIOS,
    PHASE9_SEEDS,
    PHASE9_SELECTION_RATIOS,
    Phase9Config,
    run_phase9,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate and freeze RFF/RBF gamma before the main financial period."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("user_data/data/binance"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("user_data/research_results/double_descent/phase9"),
    )
    parser.add_argument(
        "--phase5-summary",
        type=Path,
        default=Path("user_data/research_results/double_descent/phase5/summary.json"),
    )
    parser.add_argument("--cuda-python", type=Path)
    parser.add_argument("--timerange", default=PHASE9_CALIBRATION_TIMERANGE)
    parser.add_argument("--gamma", type=float, action="append", dest="gammas")
    parser.add_argument("--ratio", type=float, action="append", dest="map_ratios")
    parser.add_argument("--robustness-ratio", type=float, action="append", dest="robustness_ratios")
    parser.add_argument("--selection-ratio", type=float, action="append", dest="selection_ratios")
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--minimum-windows", type=int, default=6)
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
    config = Phase9Config(
        data_directory=arguments.data_dir.resolve(),
        output_directory=arguments.output_dir,
        phase5_summary=arguments.phase5_summary.resolve(),
        python_executable=sys.executable,
        cuda_python_executable=cuda_python,
        calibration_timerange=arguments.timerange,
        map_ratios=(tuple(arguments.map_ratios) if arguments.map_ratios else PHASE9_MAP_RATIOS),
        robustness_ratios=(
            tuple(arguments.robustness_ratios)
            if arguments.robustness_ratios
            else PHASE9_ROBUSTNESS_RATIOS
        ),
        selection_ratios=(
            tuple(arguments.selection_ratios)
            if arguments.selection_ratios
            else PHASE9_SELECTION_RATIOS
        ),
        gammas=tuple(arguments.gammas) if arguments.gammas else PHASE9_GAMMAS,
        seeds=tuple(arguments.seeds) if arguments.seeds else PHASE9_SEEDS,
        chunk_size=arguments.chunk_size,
        fee=arguments.fee,
        minimum_training_windows=arguments.minimum_windows,
        minimum_seed_count=arguments.minimum_seeds,
        subprocess_timeout_seconds=arguments.timeout,
        resume=not arguments.no_resume,
        smoke_test=arguments.smoke_test,
    )
    summary = run_phase9(config)
    print(json.dumps(summary["gamma_selection"], indent=2, sort_keys=True))
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    print(f"Artifacts: {arguments.output_dir.resolve()}")
    return 0 if summary["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
