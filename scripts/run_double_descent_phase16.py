#!/usr/bin/env python3
"""Run frozen economic and execution-cost robustness for Phase 15 trades."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.double_descent.phase16 import Phase16Config, run_phase16  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reprice every primary Phase 15 case under frozen execution costs."
    )
    parser.add_argument(
        "--phase15-summary",
        type=Path,
        default=Path("user_data/research_results/double_descent/phase15/summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("user_data/research_results/double_descent/phase16"),
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    summary = run_phase16(
        Phase16Config(
            phase15_summary=arguments.phase15_summary.resolve(),
            output_directory=arguments.output_dir,
            project_root=PROJECT_ROOT,
        )
    )
    print(
        json.dumps(
            {
                "phase": summary["phase"],
                "gate": summary["gate"],
                "economic_evidence": summary["economic_evidence"],
                "artifacts": summary["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
