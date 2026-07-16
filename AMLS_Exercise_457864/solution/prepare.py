"""Prepare deterministic image caches for the labeled dataset splits."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.preparation import PreparationConfig, prepare_all_splits


IMAGE_SIZE = 128
BATCH_SIZE = 256
WORKERS = 8


def parse_args() -> argparse.Namespace:
    """Parse the evaluator-provided runtime limit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    """Prepare train, calibration, and validation arrays."""
    args = parse_args()
    solution_dir = Path(__file__).resolve().parent
    artifacts_dir = solution_dir / "artifacts"
    output_dir = artifacts_dir / "task02" / "prepared"

    summaries = prepare_all_splits(
        data_directory=solution_dir / "data",
        clean_index_path=artifacts_dir / "task01" / "clean_train_index.parquet",
        output_directory=output_dir,
        config=PreparationConfig(
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            workers=WORKERS,
            timeout_seconds=args.timeout_seconds,
        ),
    )
    print(f"prepared_samples={sum(summary.samples for summary in summaries)}")
    print(f"output_directory={output_dir}")


if __name__ == "__main__":
    main()
