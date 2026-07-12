"""Prepare deterministic uint8 NumPy caches for all labeled image splits."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.preparation import PreparationConfig, prepare_all_splits


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the preparation pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=float, required=True)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
    )
    return parser.parse_args()


def main() -> None:
    """Run deterministic preparation for train and labeled evaluation splits."""
    args = parse_args()
    solution_directory = Path(__file__).resolve().parent
    data_directory = solution_directory / "data"
    artifacts_directory = solution_directory / "artifacts"
    clean_index_path = artifacts_directory / "task01" / "clean_train_index.parquet"
    output_directory = artifacts_directory / "task02" / "prepared"

    config = PreparationConfig(
        image_size=args.image_size,
        batch_size=args.batch_size,
        workers=args.workers,
        timeout_seconds=args.timeout_seconds,
    )

    print("script=prepare.py")
    print(f"timeout_seconds={args.timeout_seconds}")
    print(f"data_directory={data_directory}")
    print(f"data_exists={data_directory.exists()}")
    print(f"artifacts_directory={artifacts_directory}")
    print(f"output_directory={output_directory}")
    print(f"image_size={args.image_size}")
    print(f"batch_size={args.batch_size}")
    print(f"workers={args.workers}")

    summaries = prepare_all_splits(
        data_directory=data_directory,
        clean_index_path=clean_index_path,
        output_directory=output_directory,
        config=config,
    )

    total_samples = sum(summary.samples for summary in summaries)
    total_cache_bytes = sum(summary.cache_bytes for summary in summaries)
    print(f"prepared_samples={total_samples}")
    print(f"cache_bytes={total_cache_bytes}")


if __name__ == "__main__":
    main()