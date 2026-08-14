#!/usr/bin/env python3
"""Prepare and run the frozen Phase 18 recursive-indicator audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.double_descent.phase17 import representative_case_ids  # noqa: E402
from research.double_descent.phase18 import (  # noqa: E402
    Phase18Config,
    prepare_phase18,
    run_phase18,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the frozen Phase 15 pipeline with recursive-analysis."
    )
    parser.add_argument("--stage", choices=("prepare", "run", "all"), default="prepare")
    parser.add_argument("--data-dir", type=Path, default=Path("user_data/data/binance"))
    parser.add_argument(
        "--phase15-summary",
        type=Path,
        default=Path("user_data/research_results/double_descent/phase15/summary.json"),
    )
    parser.add_argument(
        "--phase17-summary",
        type=Path,
        default=Path("user_data/research_results/double_descent/phase17/summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("user_data/research_results/double_descent/phase18"),
    )
    parser.add_argument("--timeout", type=int, default=3_600)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--case-id",
        action="append",
        choices=representative_case_ids(),
        default=[],
        help="Run only a named representative; the resulting gate remains partial.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    config = Phase18Config(
        data_directory=arguments.data_dir.resolve(),
        phase15_summary=arguments.phase15_summary.resolve(),
        phase17_summary=arguments.phase17_summary.resolve(),
        output_directory=arguments.output_dir,
        project_root=PROJECT_ROOT,
        python_executable=sys.executable,
        subprocess_timeout_seconds=arguments.timeout,
        resume=not arguments.no_resume,
    )
    if arguments.stage == "prepare":
        summary = prepare_phase18(config)
    else:
        summary = run_phase18(config, tuple(arguments.case_id))
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
