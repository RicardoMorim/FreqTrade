#!/usr/bin/env python3
"""Run Phase 2 of the double-descent research programme."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.double_descent.benchmark import (  # noqa: E402
    DEFAULT_CASES,
    DEFAULT_CUDA_CASES,
    BenchmarkCase,
    Phase2Config,
    discover_cuda_python,
    run_phase2,
)


def _parse_case(value: str) -> BenchmarkCase:
    try:
        feature_count, solver = value.split(":", 1)
        if solver not in {"primal_svd", "streamed_dual", "torch_cuda_dual"}:
            raise ValueError
        return BenchmarkCase(int(feature_count), solver)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            "case must use P:primal_svd, P:streamed_dual, or P:torch_cuda_dual"
        ) from exc


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark RFF generation and minimum-norm solvers in isolated processes."
    )
    parser.add_argument("--n-train", type=int, default=128)
    parser.add_argument("--n-inference", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--cuda-python",
        type=Path,
        help="Python interpreter containing a CUDA-enabled PyTorch installation.",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Run only the CPU benchmark even when a CUDA interpreter is discoverable.",
    )
    parser.add_argument(
        "--case",
        type=_parse_case,
        action="append",
        dest="cases",
        help="Repeat to override the default benchmark grid.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("user_data/research_results/double_descent/phase2"),
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    cuda_python = None
    if not arguments.no_cuda:
        cuda_python = (
            str(arguments.cuda_python.resolve())
            if arguments.cuda_python
            else discover_cuda_python()
        )
    if arguments.cases:
        cases = tuple(arguments.cases)
    else:
        cases = DEFAULT_CASES + (DEFAULT_CUDA_CASES if cuda_python else ())
    config = Phase2Config(
        n_train=arguments.n_train,
        n_inference=arguments.n_inference,
        cases=cases,
        chunk_size=arguments.chunk_size,
        dtype=arguments.dtype,
        base_seed=arguments.seed,
        overlap_relative_tolerance=1e-4 if arguments.dtype == "float32" else 1e-6,
        cuda_python_executable=cuda_python,
    )
    summary = run_phase2(config, arguments.output_dir)
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    print(json.dumps(summary["recommendation"], indent=2, sort_keys=True))
    print(f"Artifacts: {arguments.output_dir.resolve()}")
    return 0 if summary["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
