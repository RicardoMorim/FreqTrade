#!/usr/bin/env python3
"""Prepare, capture, and run the frozen Phase 19 numerical audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.double_descent.phase19 import (  # noqa: E402
    Phase19Config,
    capture_anchor_window,
    phase19_cases,
    prepare_phase19,
    run_phase19,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit numerical stability around the real interpolation threshold."
    )
    parser.add_argument("--stage", choices=("prepare", "capture", "run", "all"), default="all")
    parser.add_argument("--data-dir", type=Path, default=Path("user_data/data/binance"))
    parser.add_argument(
        "--phase10-summary",
        type=Path,
        default=Path("user_data/research_results/double_descent/phase10/summary.json"),
    )
    parser.add_argument(
        "--phase18-summary",
        type=Path,
        default=Path("user_data/research_results/double_descent/phase18/summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("user_data/research_results/double_descent/phase19"),
    )
    parser.add_argument("--cuda-python", default="")
    parser.add_argument("--timeout", type=int, default=3_600)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--case-id",
        action="append",
        choices=[case.case_id for case in phase19_cases()],
        default=[],
        help="Run only one named numerical case; the resulting full gate remains partial.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    config = Phase19Config(
        data_directory=arguments.data_dir.resolve(),
        phase10_summary=arguments.phase10_summary.resolve(),
        phase18_summary=arguments.phase18_summary.resolve(),
        output_directory=arguments.output_dir,
        project_root=PROJECT_ROOT,
        python_executable=sys.executable,
        cuda_python_executable=arguments.cuda_python,
        subprocess_timeout_seconds=arguments.timeout,
        resume=not arguments.no_resume,
    )
    if arguments.stage == "prepare":
        summary = prepare_phase19(config)
    elif arguments.stage == "capture":
        summary = capture_anchor_window(config)
    else:
        summary = run_phase19(config, tuple(arguments.case_id))
    print(
        json.dumps(
            {
                "phase": summary["phase"],
                "stage": summary["stage"],
                "gate": summary["gate"],
                "artifacts": summary["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
