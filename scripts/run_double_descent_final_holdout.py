#!/usr/bin/env python3
"""Freeze, prepare, and execute the one-shot supervised final holdout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.double_descent.benchmark import discover_cuda_python  # noqa: E402
from research.double_descent.final_holdout import (  # noqa: E402
    FinalHoldoutConfig,
    download_final_holdout_data,
    freeze_final_holdout_protocol,
    prepare_final_holdout,
    run_final_holdout,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen 2026 YTD final holdout.")
    parser.add_argument(
        "--stage",
        choices=("freeze", "prepare", "run", "all"),
        default="freeze",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("user_data/data/binance"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("user_data/research_results/double_descent/final_holdout_2026_ytd"),
    )
    parser.add_argument("--cuda-python", type=Path)
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--timeout", type=int, default=3_600)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    cuda_python = (
        str(arguments.cuda_python.resolve()) if arguments.cuda_python else discover_cuda_python()
    )
    if not cuda_python:
        raise SystemExit("No CUDA-enabled Python interpreter was found")
    config = FinalHoldoutConfig(
        data_directory=arguments.data_dir.resolve(),
        output_directory=arguments.output_dir,
        python_executable=sys.executable,
        cuda_python_executable=cuda_python,
        subprocess_timeout_seconds=arguments.timeout,
        resume=not arguments.no_resume,
    )
    if arguments.stage == "freeze":
        result = freeze_final_holdout_protocol(config)
    else:
        if arguments.download_data:
            download_final_holdout_data(config)
        if arguments.stage in {"prepare", "all"}:
            result = prepare_final_holdout(config)
        if arguments.stage in {"run", "all"}:
            result = run_final_holdout(config)
    print(json.dumps(result, indent=2, default=str, sort_keys=True))
    gate = result.get("gate")
    return 0 if gate is None or gate.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
