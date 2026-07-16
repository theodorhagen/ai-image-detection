"""Data discovery and Parquet access helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq


LABELED_SPLIT_NAMES = (
    "train",
    "calibration",
    "calibration_augmented",
    "validation",
    "validation_augmented",
)

IMAGE_COLUMN_CANDIDATES = ("image",)
SOURCE_CLASS_COLUMN_CANDIDATES = ("source_class", "source class", "source-class", "sourceClass")
ROW_ID_COLUMN_CANDIDATES = ("row_id", "row id", "row-id", "rowId")


@dataclass(frozen=True)
class DatasetSplit:
    """Description of one dataset split."""

    name: str
    path: Path
    parquet_files: tuple[Path, ...]


def find_parquet_files(directory: Path) -> tuple[Path, ...]:
    """Return all Parquet files below a directory in deterministic order."""

    if not directory.exists():
        return tuple()
    return tuple(sorted(directory.rglob("*.parquet")))


def discover_labeled_splits(data_dir: Path) -> tuple[DatasetSplit, ...]:
    """Discover labeled dataset splits that exist locally."""

    splits: list[DatasetSplit] = []
    for split_name in LABELED_SPLIT_NAMES:
        split_path = data_dir / split_name
        parquet_files = find_parquet_files(split_path)
        if parquet_files:
            splits.append(DatasetSplit(split_name, split_path, parquet_files))
    return tuple(splits)


def resolve_column(schema_names: Iterable[str], candidates: Iterable[str]) -> str:
    """Resolve a column name from candidate names, case-insensitively."""

    names = list(schema_names)
    exact = {name: name for name in names}
    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]

    lowered = {name.lower(): name for name in names}
    for candidate in candidates:
        match = lowered.get(candidate.lower())
        if match is not None:
            return match

    raise KeyError(f"Could not resolve any of {tuple(candidates)} from columns {tuple(names)}")


def parquet_schema_names(parquet_file: Path) -> tuple[str, ...]:
    """Return column names for a Parquet file."""

    return tuple(pq.ParquetFile(parquet_file).schema_arrow.names)


def relative_to_data(path: Path, data_dir: Path) -> str:
    """Return a POSIX-style path relative to the dataset root."""

    return path.relative_to(data_dir).as_posix()
