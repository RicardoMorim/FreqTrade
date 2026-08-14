from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ASSETS = {
    "DoubleDescentRFFRegressor.py": Path("user_data/freqaimodels/DoubleDescentRFFRegressor.py"),
    "DoubleDescentFreqaiStrategy.py": Path("user_data/strategies/DoubleDescentFreqaiStrategy.py"),
    "double_descent_freqai.example.json": Path("user_data/configs/double_descent_freqai.example.json"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Install the experiment assets into Freqtrade user_data")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Root of the Freqtrade checkout/fork",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing installed assets")
    args = parser.parse_args()

    source_dir = Path(__file__).resolve().parent / "freqtrade_assets"
    for source_name, relative_target in ASSETS.items():
        source = source_dir / source_name
        target = args.repo_root / relative_target
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not args.force:
            raise SystemExit(f"Refusing to overwrite {target}; rerun with --force")
        shutil.copy2(source, target)
        print(f"Installed {source_name} -> {target}")


if __name__ == "__main__":
    main()
