#!/usr/bin/env python3
"""Prepare and run the frozen Phase 17 lookahead-bias audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.double_descent.phase17 import (  # noqa: E402
    Phase17Config,
    prepare_phase17,
    representative_case_ids,
    run_phase17,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the frozen Phase 15 FreqAI pipeline with lookahead-analysis."
    )
    parser.add_argument("--stage", choices=("prepare", "run", "all"), default="prepare")
    parser.add_argument("--data-dir", type=Path, default=Path("user_data/data/binance"))
    parser.add_argument(
        "--phase15-summary",
        type=Path,
        default=Path("user_data/research_results/double_descent/phase15/summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("user_data/research_results/double_descent/phase17"),
    )
    parser.add_argument("--timeout", type=int, default=3_600)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--case-id",
        action="append",
        choices=representative_case_ids(),
        default=[],
        help="Run only a named representative; useful for a non-passing smoke test.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    config = Phase17Config(
        data_directory=arguments.data_dir.resolve(),
        phase15_summary=arguments.phase15_summary.resolve(),
        output_directory=arguments.output_dir,
        project_root=PROJECT_ROOT,
        python_executable=sys.executable,
        subprocess_timeout_seconds=arguments.timeout,
        resume=not arguments.no_resume,
    )
    if arguments.stage == "prepare":
        summary = prepare_phase17(config)
    else:
        summary = run_phase17(config, tuple(arguments.case_id))
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
