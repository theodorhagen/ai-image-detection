"""Dataset exploration and deterministic cleaning for Task 1."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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
    save_aspect_ratio_histogram,
    save_categorical_percentage_plot,
    save_class_distribution_plot,
    save_class_percentage_plot,
    save_encoded_size_boxplot,
    save_encoded_size_boxplot_log,
    save_image_size_scatter,
    save_image_size_scatter_by_label,
    save_megapixels_boxplot,
    save_source_class_distribution_plot,
    save_source_class_percentage_plot,
    write_csv,
    write_json,
    write_parquet,
)


CLEAN_BATCH_SIZE = 512
MAX_IMAGES_PER_SPLIT = None


def parse_args() -> argparse.Namespace:
    """Parse the evaluator-provided runtime limit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=float, required=True)
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


def add_derived_columns(metadata: pd.DataFrame) -> pd.DataFrame:
    """Add derived metadata columns for shortcut and cleaning analysis."""

    metadata = metadata.copy()

    numeric_columns = [
        "source_class",
        "binary_label",
        "image_bytes",
        "width",
        "height",
        "row_index",
    ]
    for column in numeric_columns:
        metadata[column] = pd.to_numeric(metadata[column], errors="coerce")

    metadata["decode_ok"] = metadata["decode_ok"].astype(bool)
    metadata["is_clean"] = metadata["is_clean"].astype(bool)

    metadata["aspect_ratio"] = np.where(
        (metadata["decode_ok"]) & (metadata["height"] > 0),
        metadata["width"] / metadata["height"],
        np.nan,
    )
    metadata["megapixels"] = np.where(
        metadata["decode_ok"],
        (metadata["width"] * metadata["height"]) / 1_000_000.0,
        np.nan,
    )
    metadata["is_square"] = (
        (metadata["decode_ok"])
        & (metadata["width"].notna())
        & (metadata["height"].notna())
        & (metadata["width"] == metadata["height"])
    )
    metadata["label_name"] = metadata["binary_label"].map({0: "real", 1: "ai_generated"})
    metadata["aspect_bucket"] = pd.cut(
        metadata["aspect_ratio"],
        bins=[0.0, 0.8, 0.95, 1.05, 1.25, np.inf],
        labels=[
            "portrait",
            "near_portrait",
            "near_square",
            "near_landscape",
            "landscape",
        ],
        include_lowest=True,
    )

    return metadata


def add_percent_within_group(
    dataframe: pd.DataFrame,
    group_columns: list[str],
    count_column: str = "count",
    percent_column: str = "percentage",
) -> pd.DataFrame:
    """Add a percentage column normalized within selected grouping columns."""

    dataframe = dataframe.copy()
    totals = dataframe.groupby(group_columns)[count_column].transform("sum")
    dataframe[percent_column] = np.where(totals > 0, dataframe[count_column] / totals * 100.0, 0.0)
    return dataframe


def categorical_distribution(
    metadata: pd.DataFrame,
    column: str,
    group_columns: list[str],
) -> pd.DataFrame:
    """Build a categorical count and percentage table."""

    table = (
        metadata[metadata["decode_ok"]]
        .assign(**{column: metadata[column].fillna("unknown").astype(str)})
        .groupby([*group_columns, column], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values([*group_columns, column])
    )
    return add_percent_within_group(table, group_columns)


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
            median_aspect_ratio=("aspect_ratio", "median"),
            square_image_rate=("is_square", "mean"),
        )
        .reset_index()
    )

    class_distribution = (
        metadata.groupby(["split", "source_class", "binary_label"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["split", "source_class"])
    )
    class_distribution = add_percent_within_group(class_distribution, ["split"])

    binary_class_distribution = (
        metadata.groupby(["split", "binary_label"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["split", "binary_label"])
    )
    binary_class_distribution = add_percent_within_group(binary_class_distribution, ["split"])

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
            median_aspect_ratio=("aspect_ratio", "median"),
            median_megapixels=("megapixels", "median"),
            median_encoded_bytes=("image_bytes", "median"),
            square_image_rate=("is_square", "mean"),
        )
        .reset_index()
    )

    dimension_shortcut_summary = (
        metadata[metadata["decode_ok"]]
        .groupby(["split", "source_class", "binary_label"], dropna=False)
        .agg(
            n_images=("split", "size"),
            median_width=("width", "median"),
            median_height=("height", "median"),
            median_aspect_ratio=("aspect_ratio", "median"),
            median_megapixels=("megapixels", "median"),
            median_encoded_bytes=("image_bytes", "median"),
            square_image_rate=("is_square", "mean"),
        )
        .reset_index()
        .sort_values(["split", "source_class"])
    )

    aspect_bucket_distribution = (
        metadata[metadata["decode_ok"]]
        .groupby(["split", "binary_label", "aspect_bucket"], dropna=False, observed=False)
        .size()
        .reset_index(name="count")
        .sort_values(["split", "binary_label", "aspect_bucket"])
    )
    aspect_bucket_distribution = add_percent_within_group(
        aspect_bucket_distribution,
        ["split", "binary_label"],
    )

    decode_error_summary = (
        metadata[~metadata["decode_ok"]]
        .groupby(["split", "error_type"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["split", "error_type"])
    )

    format_distribution = categorical_distribution(
        metadata,
        column="image_format",
        group_columns=["split", "binary_label"],
    )

    mode_distribution = categorical_distribution(
        metadata,
        column="mode",
        group_columns=["split", "binary_label"],
    )

    return {
        "split_summary": split_summary,
        "class_distribution": class_distribution,
        "binary_class_distribution": binary_class_distribution,
        "image_size_summary": image_size_summary,
        "dimension_shortcut_summary": dimension_shortcut_summary,
        "aspect_bucket_distribution": aspect_bucket_distribution,
        "format_distribution": format_distribution,
        "mode_distribution": mode_distribution,
        "decode_error_summary": decode_error_summary,
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
            "aspect_ratio",
            "megapixels",
            "image_bytes",
        ]
    ].copy()
    write_parquet(clean_train_index, task_dir / "clean_train_index.parquet")

    write_json(CLEANING_CONFIG, task_dir / "cleaning_config.json")

    save_class_distribution_plot(metadata, figures_dir / "class_distribution.png")
    save_class_percentage_plot(metadata, figures_dir / "class_distribution_percent.png")
    save_source_class_distribution_plot(metadata, figures_dir / "source_class_distribution.png")
    save_source_class_percentage_plot(metadata, figures_dir / "source_class_distribution_percent.png")
    save_image_size_scatter(metadata, figures_dir / "image_size_distribution.png")
    save_image_size_scatter_by_label(metadata, figures_dir / "image_size_distribution_by_label.png")
    save_encoded_size_boxplot(metadata, figures_dir / "encoded_size_by_label.png")
    save_encoded_size_boxplot_log(metadata, figures_dir / "encoded_size_by_label_log.png")
    save_aspect_ratio_histogram(metadata, figures_dir / "aspect_ratio_by_label.png")
    save_megapixels_boxplot(metadata, figures_dir / "megapixels_by_label.png")
    save_categorical_percentage_plot(
        metadata,
        column="image_format",
        path=figures_dir / "image_format_distribution.png",
        title="Image-format distribution by binary label",
    )
    save_categorical_percentage_plot(
        metadata,
        column="mode",
        path=figures_dir / "image_mode_distribution.png",
        title="Image-mode distribution by binary label",
    )


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
                batch_size=CLEAN_BATCH_SIZE,
                max_images=MAX_IMAGES_PER_SPLIT,
            )
        )

    metadata = add_derived_columns(pd.DataFrame.from_records(all_records))
    write_outputs(metadata, paths.artifacts_dir)

    print(f"timeout_seconds={args.timeout_seconds}")
    print(f"processed_images={len(metadata)}")
    print(f"output_directory={paths.artifacts_dir / 'task01'}")


if __name__ == "__main__":
    main()
