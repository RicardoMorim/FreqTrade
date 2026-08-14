#!/usr/bin/env python3
"""Run the frozen Phase 20 effective-dimension audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.double_descent.phase20 import Phase20Config, run_phase20  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare nominal RFF predictor count with spectral effective dimension."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("user_data/research_results/double_descent/phase20"),
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    summary = run_phase20(Phase20Config(output_directory=arguments.output_dir))
    print(
        json.dumps(
            {
                "phase": summary["phase"],
                "stage": summary["stage"],
                "gate": summary["gate"],
                "assessment": summary["assessment"],
                "artifacts": summary["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
