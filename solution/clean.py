"""Dataset exploration and deterministic cleaning for Task 1."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from src.cleaning import CLEANING_CONFIG, binary_label, is_clean_image_record
from src.data_io import (
    IMAGE_COLUMN_CANDIDATES,
    SOURCE_CLASS_COLUMN_CANDIDATES,
    DatasetSplit,
    discover_labeled_splits,
    relative_to_data,
    resolve_column,
)
from src.image_utils import read_image_info
from src.paths import get_project_paths
from src.reporting import (
    ensure_dir,
    save_class_distribution_plot,
    save_encoded_size_boxplot,
    save_image_size_scatter,
    save_source_class_distribution_plot,
    write_csv,
    write_json,
    write_parquet,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_images_per_split", type=int, default=None)
    return parser.parse_args()


def process_labeled_split(
    split: DatasetSplit,
    data_dir: Path,
    batch_size: int,
    max_images: int | None,
) -> list[dict[str, object]]:
    """Extract metadata records for one labeled split."""

    records: list[dict[str, object]] = []
    split_count = 0

    for parquet_file in tqdm(split.parquet_files, desc=f"scan {split.name}", unit="file"):
        parquet = pq.ParquetFile(parquet_file)
        schema_names = parquet.schema_arrow.names
        image_column = resolve_column(schema_names, IMAGE_COLUMN_CANDIDATES)
        source_class_column = resolve_column(schema_names, SOURCE_CLASS_COLUMN_CANDIDATES)
        row_offset = 0

        for batch in parquet.iter_batches(
            batch_size=batch_size,
            columns=[image_column, source_class_column],
        ):
            columns = batch.to_pydict()
            images = columns[image_column]
            source_classes = columns[source_class_column]

            for batch_index, (image_value, source_class_value) in enumerate(zip(images, source_classes)):
                if max_images is not None and split_count >= max_images:
                    return records

                source_class = int(source_class_value)
                info = read_image_info(image_value).to_dict()
                record: dict[str, object] = {
                    "split": split.name,
                    "parquet_file": relative_to_data(parquet_file, data_dir),
                    "row_index": row_offset + batch_index,
                    "source_class": source_class,
                    "binary_label": binary_label(source_class),
                    **info,
                }
                record["is_clean"] = is_clean_image_record(record)
                records.append(record)
                split_count += 1

            row_offset += batch.num_rows

    return records


def build_summaries(metadata: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build tabular summaries used in the report."""

    split_summary = (
        metadata.groupby("split", dropna=False)
        .agg(
            n_images=("split", "size"),
            n_clean=("is_clean", "sum"),
            n_decode_failed=("decode_ok", lambda values: int((~values).sum())),
            mean_width=("width", "mean"),
            mean_height=("height", "mean"),
            mean_encoded_bytes=("image_bytes", "mean"),
        )
        .reset_index()
    )

    class_distribution = (
        metadata.groupby(["split", "source_class", "binary_label"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["split", "source_class"])
    )

    image_size_summary = (
        metadata[metadata["decode_ok"]]
        .groupby(["split", "binary_label"], dropna=False)
        .agg(
            n_images=("split", "size"),
            min_width=("width", "min"),
            median_width=("width", "median"),
            max_width=("width", "max"),
            min_height=("height", "min"),
            median_height=("height", "median"),
            max_height=("height", "max"),
            median_encoded_bytes=("image_bytes", "median"),
        )
        .reset_index()
    )

    error_summary = (
        metadata[~metadata["decode_ok"]]
        .groupby(["split", "error_type"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["split", "error_type"])
    )

    return {
        "split_summary": split_summary,
        "class_distribution": class_distribution,
        "image_size_summary": image_size_summary,
        "decode_error_summary": error_summary,
    }


def write_outputs(metadata: pd.DataFrame, artifacts_dir: Path) -> None:
    """Write Task 1 tables and figures."""

    task_dir = ensure_dir(artifacts_dir / "task01")
    figures_dir = ensure_dir(task_dir / "figures")
    summaries = build_summaries(metadata)

    write_parquet(metadata, task_dir / "image_metadata.parquet")
    for name, dataframe in summaries.items():
        write_csv(dataframe, task_dir / f"{name}.csv")

    clean_train_index = metadata[(metadata["split"] == "train") & (metadata["is_clean"])][
        [
            "split",
            "parquet_file",
            "row_index",
            "source_class",
            "binary_label",
            "width",
            "height",
            "image_bytes",
        ]
    ].copy()
    write_parquet(clean_train_index, task_dir / "clean_train_index.parquet")

    write_json(CLEANING_CONFIG, task_dir / "cleaning_config.json")
    save_class_distribution_plot(metadata, figures_dir / "class_distribution.png")
    save_source_class_distribution_plot(metadata, figures_dir / "source_class_distribution.png")
    save_image_size_scatter(metadata, figures_dir / "image_size_distribution.png")
    save_encoded_size_boxplot(metadata, figures_dir / "encoded_size_by_label.png")


def main() -> None:
    """Run dataset exploration and deterministic cleaning."""

    args = parse_args()
    paths = get_project_paths(__file__)
    ensure_dir(paths.artifacts_dir)

    if not paths.data_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {paths.data_dir}")

    splits = discover_labeled_splits(paths.data_dir)
    if not splits:
        raise FileNotFoundError(f"No labeled Parquet files found below: {paths.data_dir}")

    all_records: list[dict[str, object]] = []
    for split in splits:
        all_records.extend(
            process_labeled_split(
                split=split,
                data_dir=paths.data_dir,
                batch_size=args.batch_size,
                max_images=args.max_images_per_split,
            )
        )

    metadata = pd.DataFrame.from_records(all_records)
    write_outputs(metadata, paths.artifacts_dir)

    print(f"timeout_seconds={args.timeout_seconds}")
    print(f"processed_images={len(metadata)}")
    print(f"output_directory={paths.artifacts_dir / 'task01'}")


if __name__ == "__main__":
    main()
