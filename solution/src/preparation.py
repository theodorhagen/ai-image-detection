"""Preparation pipeline for deterministic NumPy image caches."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.preprocessing import decode_and_resize


LABELED_SPLITS = (
    "train",
    "calibration",
    "calibration_augmented",
    "validation",
    "validation_augmented",
)


@dataclass(frozen=True)
class PreparationConfig:
    """Configuration for deterministic image preparation."""

    image_size: int = 128
    batch_size: int = 256
    workers: int = 8
    timeout_seconds: float = 600.0
    safety_margin_seconds: float = 10.0


@dataclass(frozen=True)
class SplitSummary:
    """Summary of one prepared labeled split."""

    split: str
    samples: int
    real_samples: int
    ai_samples: int
    cache_bytes: int
    elapsed_seconds: float


@dataclass(frozen=True)
class RowReference:
    """Location and labels of one image row in a parquet file."""

    parquet_file: str
    row_index: int
    source_class: int
    label: int


class Deadline:
    """Track the remaining runtime of the preparation script."""

    def __init__(self, timeout_seconds: float, safety_margin_seconds: float) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if safety_margin_seconds < 0:
            raise ValueError("safety_margin_seconds must be non-negative")

        self._deadline = time.monotonic() + timeout_seconds
        self._safety_margin_seconds = safety_margin_seconds

    def check(self, operation: str) -> None:
        """Raise TimeoutError when the configured safety margin is reached."""
        remaining = self._deadline - time.monotonic()
        if remaining <= self._safety_margin_seconds:
            raise TimeoutError(
                f"Insufficient time remaining before {operation}: "
                f"{remaining:.1f} seconds"
            )


class CleanTrainSelection:
    """Represent the train rows retained by the deterministic cleaning step."""

    def __init__(
        self,
        file_rows: dict[str, set[int]] | None = None,
        global_rows: set[int] | None = None,
        select_all: bool = False,
    ) -> None:
        self.file_rows = file_rows
        self.global_rows = global_rows
        self.select_all = select_all

        active_modes = sum(
            (
                file_rows is not None,
                global_rows is not None,
                select_all,
            )
        )
        if active_modes != 1:
            raise ValueError("Exactly one train-selection mode must be active")

    @property
    def expected_count(self) -> int | None:
        """Return the number of selected rows when it is known directly."""
        if self.file_rows is not None:
            return sum(len(rows) for rows in self.file_rows.values())
        if self.global_rows is not None:
            return len(self.global_rows)
        return None

    def allows(
        self,
        relative_file: str,
        row_index: int,
        global_index: int,
    ) -> bool:
        """Return whether a train row belongs to the cleaned dataset."""
        if self.select_all:
            return True
        if self.global_rows is not None:
            return global_index in self.global_rows
        if self.file_rows is None:
            return False

        candidates = (
            _normalize_relative_path(relative_file),
            Path(relative_file).name,
        )
        return any(
            row_index in self.file_rows.get(candidate, set())
            for candidate in candidates
        )


def prepare_all_splits(
    data_directory: Path,
    clean_index_path: Path,
    output_directory: Path,
    config: PreparationConfig,
) -> list[SplitSummary]:
    """Prepare all labeled splits and write memory-mappable NumPy arrays."""
    _validate_configuration(config)
    _validate_input_directories(data_directory, clean_index_path)

    output_directory.mkdir(parents=True, exist_ok=True)
    deadline = Deadline(config.timeout_seconds, config.safety_margin_seconds)
    train_files = discover_parquet_files(data_directory / "train")
    train_selection = load_clean_train_selection(clean_index_path, train_files)

    split_counts = {
        split: count_selected_rows(
            data_directory / split,
            train_selection if split == "train" else None,
        )
        for split in LABELED_SPLITS
    }
    _check_available_space(output_directory, split_counts, config.image_size)

    summaries: list[SplitSummary] = []
    train_statistics: dict[str, list[float]] | None = None

    for split in LABELED_SPLITS:
        deadline.check(f"preparing split {split}")
        summary, statistics = prepare_split(
            split=split,
            split_directory=data_directory / split,
            output_directory=output_directory,
            sample_count=split_counts[split],
            config=config,
            deadline=deadline,
            train_selection=train_selection if split == "train" else None,
        )
        summaries.append(summary)
        if split == "train":
            train_statistics = statistics

    write_preparation_metadata(
        output_directory=output_directory,
        data_directory=data_directory,
        clean_index_path=clean_index_path,
        config=config,
        summaries=summaries,
        train_statistics=train_statistics,
    )
    return summaries


def prepare_split(
    split: str,
    split_directory: Path,
    output_directory: Path,
    sample_count: int,
    config: PreparationConfig,
    deadline: Deadline,
    train_selection: CleanTrainSelection | None,
) -> tuple[SplitSummary, dict[str, list[float]] | None]:
    """Prepare one split into image, label, source-class, and index files."""
    start_time = time.perf_counter()
    image_path = output_directory / f"{split}_images.npy"
    label_path = output_directory / f"{split}_labels.npy"
    source_path = output_directory / f"{split}_source_classes.npy"
    index_path = output_directory / f"{split}_index.parquet"
    temporary_paths = [
        _temporary_path(image_path),
        _temporary_path(label_path),
        _temporary_path(source_path),
        _temporary_path(index_path),
    ]
    _remove_paths(temporary_paths)

    images = np.lib.format.open_memmap(
        temporary_paths[0],
        mode="w+",
        dtype=np.uint8,
        shape=(sample_count, 3, config.image_size, config.image_size),
    )
    labels = np.lib.format.open_memmap(
        temporary_paths[1],
        mode="w+",
        dtype=np.uint8,
        shape=(sample_count,),
    )
    source_classes = np.lib.format.open_memmap(
        temporary_paths[2],
        mode="w+",
        dtype=np.int8,
        shape=(sample_count,),
    )

    index_records: list[dict[str, Any]] = []
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_squared_sum = np.zeros(3, dtype=np.float64)
    channel_pixel_count = 0
    write_position = 0

    try:
        with ThreadPoolExecutor(max_workers=config.workers) as executor:
            for encoded_images, references in iter_selected_batches(
                split_directory=split_directory,
                batch_size=config.batch_size,
                train_selection=train_selection,
            ):
                deadline.check(f"decoding split {split}")
                prepared_images = list(
                    executor.map(
                        _decode_task,
                        encoded_images,
                        [config.image_size] * len(encoded_images),
                    )
                )

                batch_end = write_position + len(prepared_images)
                if batch_end > sample_count:
                    raise RuntimeError(
                        f"Prepared more rows than expected for split {split}"
                    )

                batch_array = np.stack(prepared_images, axis=0)
                batch_labels = np.fromiter(
                    (reference.label for reference in references),
                    dtype=np.uint8,
                    count=len(references),
                )
                batch_sources = np.fromiter(
                    (reference.source_class for reference in references),
                    dtype=np.int8,
                    count=len(references),
                )

                images[write_position:batch_end] = batch_array
                labels[write_position:batch_end] = batch_labels
                source_classes[write_position:batch_end] = batch_sources

                for offset, reference in enumerate(references):
                    index_records.append(
                        {
                            "sample_index": write_position + offset,
                            "parquet_file": reference.parquet_file,
                            "row_index": reference.row_index,
                            "source_class": reference.source_class,
                            "label": reference.label,
                        }
                    )

                if split == "train":
                    batch_float = batch_array.astype(np.float64)
                    channel_sum += batch_float.sum(axis=(0, 2, 3))
                    channel_squared_sum += np.square(batch_float).sum(axis=(0, 2, 3))
                    channel_pixel_count += (
                        batch_array.shape[0]
                        * batch_array.shape[2]
                        * batch_array.shape[3]
                    )

                write_position = batch_end
                _print_progress(split, write_position, sample_count, start_time)

        if write_position != sample_count:
            raise RuntimeError(
                f"Prepared {write_position} rows for split {split}, "
                f"but expected {sample_count}"
            )

        images.flush()
        labels.flush()
        source_classes.flush()
        images = None
        labels = None
        source_classes = None

        index_frame = pd.DataFrame.from_records(index_records)
        index_frame.to_parquet(temporary_paths[3], index=False)

        _replace_file(temporary_paths[0], image_path)
        _replace_file(temporary_paths[1], label_path)
        _replace_file(temporary_paths[2], source_path)
        _replace_file(temporary_paths[3], index_path)
    except Exception:
        images = None
        labels = None
        source_classes = None
        _remove_paths(temporary_paths)
        raise

    elapsed_seconds = time.perf_counter() - start_time
    real_samples = int(sum(record["label"] == 0 for record in index_records))
    ai_samples = sample_count - real_samples
    cache_bytes = sum(
        path.stat().st_size
        for path in (image_path, label_path, source_path, index_path)
    )
    summary = SplitSummary(
        split=split,
        samples=sample_count,
        real_samples=real_samples,
        ai_samples=ai_samples,
        cache_bytes=cache_bytes,
        elapsed_seconds=elapsed_seconds,
    )

    statistics = None
    if split == "train":
        statistics = _finalize_channel_statistics(
            channel_sum=channel_sum,
            channel_squared_sum=channel_squared_sum,
            pixel_count=channel_pixel_count,
        )

    print(
        f"prepared_split={split} samples={sample_count} "
        f"elapsed_seconds={elapsed_seconds:.3f}"
    )
    return summary, statistics


def discover_parquet_files(split_directory: Path) -> list[Path]:
    """Return parquet files in deterministic relative-path order."""
    if not split_directory.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_directory}")

    files = sorted(
        split_directory.rglob("*.parquet"),
        key=lambda path: path.relative_to(split_directory).as_posix(),
    )
    if not files:
        raise FileNotFoundError(f"No parquet files found in {split_directory}")
    return files


def count_selected_rows(
    split_directory: Path,
    train_selection: CleanTrainSelection | None,
) -> int:
    """Count rows that will be written for one split."""
    files = discover_parquet_files(split_directory)
    if train_selection is not None and train_selection.expected_count is not None:
        return train_selection.expected_count
    return sum(pq.ParquetFile(path).metadata.num_rows for path in files)


def load_clean_train_selection(
    clean_index_path: Path,
    train_files: Sequence[Path],
) -> CleanTrainSelection:
    """Load the cleaned train index using common row-reference schemas."""
    frame = pd.read_parquet(clean_index_path)
    if frame.empty:
        raise ValueError(f"Clean train index is empty: {clean_index_path}")

    valid_column = _find_column(
        frame.columns,
        ("is_valid", "valid", "decode_ok", "is_decodable", "keep"),
    )
    if valid_column is not None:
        frame = frame[frame[valid_column].astype(bool)].copy()

    path_column = _find_column(
        frame.columns,
        (
            "parquet_file",
            "parquet_path",
            "file_path",
            "relative_path",
            "source_file",
            "file_name",
            "filename",
            "path",
        ),
    )
    row_column = _find_column(
        frame.columns,
        (
            "row_index",
            "row_idx",
            "row_number",
            "index_in_file",
            "record_index",
            "row_id",
        ),
    )
    global_column = _find_column(
        frame.columns,
        ("global_index", "dataset_index"),
    )

    if path_column is not None and row_column is not None:
        file_rows: dict[str, set[int]] = {}
        for path_value, row_value in zip(
            frame[path_column].tolist(),
            frame[row_column].tolist(),
        ):
            normalized_path = _normalize_relative_path(str(path_value))
            file_rows.setdefault(normalized_path, set()).add(int(row_value))
        return CleanTrainSelection(file_rows=file_rows)

    if global_column is not None:
        global_rows = {int(value) for value in frame[global_column].tolist()}
        return CleanTrainSelection(global_rows=global_rows)

    if row_column is not None and len(train_files) == 1:
        relative_name = train_files[0].name
        rows = {int(value) for value in frame[row_column].tolist()}
        return CleanTrainSelection(file_rows={relative_name: rows})

    total_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in train_files)
    if valid_column is not None and len(frame) == total_rows:
        return CleanTrainSelection(select_all=True)

    available_columns = ", ".join(str(column) for column in frame.columns)
    raise ValueError(
        "Could not infer row references from clean_train_index.parquet. "
        "Expected a parquet-file column together with a row-index column, "
        "or a global-index column. "
        f"Available columns: {available_columns}"
    )


def iter_selected_batches(
    split_directory: Path,
    batch_size: int,
    train_selection: CleanTrainSelection | None,
) -> Iterator[tuple[list[bytes], list[RowReference]]]:
    """Yield encoded image bytes and references in deterministic batches."""
    files = discover_parquet_files(split_directory)
    global_index = 0
    pending_images: list[bytes] = []
    pending_references: list[RowReference] = []

    for parquet_path in files:
        parquet_file = pq.ParquetFile(parquet_path)
        image_column = _resolve_schema_column(
            parquet_file.schema.names,
            ("image", "image_bytes", "encoded_image"),
        )
        source_column = _resolve_schema_column(
            parquet_file.schema.names,
            ("source_class", "source class", "sourceclass", "class"),
        )
        relative_file = parquet_path.relative_to(split_directory).as_posix()
        file_row_index = 0

        for record_batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=[image_column, source_column],
        ):
            encoded_values = record_batch.column(0).to_pylist()
            source_values = record_batch.column(1).to_pylist()

            for encoded_value, source_value in zip(encoded_values, source_values):
                selected = (
                    train_selection is None
                    or train_selection.allows(
                        relative_file=relative_file,
                        row_index=file_row_index,
                        global_index=global_index,
                    )
                )

                if selected:
                    if encoded_value is None:
                        raise ValueError(
                            f"Missing image bytes in {relative_file} row {file_row_index}"
                        )
                    source_class = int(source_value)
                    label = 0 if source_class == 0 else 1
                    pending_images.append(bytes(encoded_value))
                    pending_references.append(
                        RowReference(
                            parquet_file=relative_file,
                            row_index=file_row_index,
                            source_class=source_class,
                            label=label,
                        )
                    )

                    if len(pending_images) >= batch_size:
                        yield pending_images, pending_references
                        pending_images = []
                        pending_references = []

                file_row_index += 1
                global_index += 1

    if pending_images:
        yield pending_images, pending_references


def write_preparation_metadata(
    output_directory: Path,
    data_directory: Path,
    clean_index_path: Path,
    config: PreparationConfig,
    summaries: Sequence[SplitSummary],
    train_statistics: dict[str, list[float]] | None,
) -> None:
    """Write preparation configuration and split summaries atomically."""
    summary_frame = pd.DataFrame([asdict(summary) for summary in summaries])
    summary_path = output_directory / "preparation_summary.csv"
    summary_temporary_path = _temporary_path(summary_path)
    summary_frame.to_csv(summary_temporary_path, index=False)
    _replace_file(summary_temporary_path, summary_path)

    metadata = {
        "image_size": config.image_size,
        "array_layout": "NCHW",
        "image_dtype": "uint8",
        "label_dtype": "uint8",
        "source_class_dtype": "int8",
        "color_mode": "RGB",
        "resampling": "bilinear",
        "exif_transpose": True,
        "alpha_background": [255, 255, 255],
        "stored_normalization": None,
        "model_input_scale": 1.0 / 255.0,
        "binary_label_mapping": {
            "0": 0,
            "1": 1,
            "2": 1,
            "3": 1,
            "4": 1,
            "5": 1,
        },
        "prepared_splits": list(LABELED_SPLITS),
        "predict_prepared": False,
        "data_directory": str(data_directory),
        "clean_train_index": str(clean_index_path),
        "configuration": asdict(config),
        "train_channel_statistics": train_statistics,
    }
    metadata_path = output_directory / "preparation_config.json"
    _write_json_atomic(metadata_path, metadata)


def _decode_task(image_bytes: bytes, image_size: int) -> np.ndarray:
    """Decode one image for use with an ordered executor map."""
    return decode_and_resize(image_bytes, image_size)


def _finalize_channel_statistics(
    channel_sum: np.ndarray,
    channel_squared_sum: np.ndarray,
    pixel_count: int,
) -> dict[str, list[float]]:
    """Convert accumulated uint8 moments into normalized channel statistics."""
    if pixel_count <= 0:
        raise ValueError("Cannot compute channel statistics without pixels")

    mean_uint8 = channel_sum / pixel_count
    variance_uint8 = channel_squared_sum / pixel_count - np.square(mean_uint8)
    variance_uint8 = np.maximum(variance_uint8, 0.0)
    std_uint8 = np.sqrt(variance_uint8)

    return {
        "mean": (mean_uint8 / 255.0).tolist(),
        "std": (std_uint8 / 255.0).tolist(),
    }


def _check_available_space(
    output_directory: Path,
    split_counts: dict[str, int],
    image_size: int,
) -> None:
    """Check that the output volume can hold the uncompressed NumPy caches."""
    image_bytes = sum(split_counts.values()) * 3 * image_size * image_size
    label_bytes = sum(split_counts.values()) * 2
    estimated_bytes = int((image_bytes + label_bytes) * 1.05)
    available_bytes = shutil.disk_usage(output_directory).free

    if estimated_bytes > available_bytes:
        raise OSError(
            "Insufficient free disk space for prepared image caches: "
            f"required approximately {estimated_bytes} bytes, "
            f"available {available_bytes} bytes"
        )


def _validate_configuration(config: PreparationConfig) -> None:
    """Validate preparation parameters before reading data."""
    if config.image_size <= 0:
        raise ValueError("image_size must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.workers <= 0:
        raise ValueError("workers must be positive")
    if config.workers > 8:
        raise ValueError("workers must not exceed the eight allocated CPUs")


def _validate_input_directories(
    data_directory: Path,
    clean_index_path: Path,
) -> None:
    """Validate required runtime inputs."""
    if not data_directory.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_directory}")
    if not clean_index_path.exists():
        raise FileNotFoundError(
            f"Clean train index does not exist: {clean_index_path}. "
            "Run clean.py before prepare.py."
        )


def _find_column(
    columns: Iterable[Any],
    candidates: Sequence[str],
) -> Any | None:
    """Find a dataframe column by normalized candidate name."""
    normalized_columns = {
        _normalize_column_name(str(column)): column
        for column in columns
    }
    for candidate in candidates:
        match = normalized_columns.get(_normalize_column_name(candidate))
        if match is not None:
            return match
    return None


def _resolve_schema_column(
    columns: Sequence[str],
    candidates: Sequence[str],
) -> str:
    """Resolve a required parquet column by normalized name."""
    column = _find_column(columns, candidates)
    if column is None:
        expected = ", ".join(candidates)
        available = ", ".join(columns)
        raise KeyError(
            f"Could not find required parquet column. "
            f"Expected one of [{expected}], available [{available}]"
        )
    return str(column)


def _normalize_column_name(value: str) -> str:
    """Normalize column names for schema matching."""
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _normalize_relative_path(value: str) -> str:
    """Normalize stored parquet paths to a train-relative representation."""
    normalized = value.replace("\\", "/")
    marker = "/train/"
    if marker in normalized:
        normalized = normalized.split(marker, maxsplit=1)[1]
    if normalized.startswith("train/"):
        normalized = normalized[len("train/") :]
    return normalized.lstrip("./")


def _temporary_path(path: Path) -> Path:
    """Return a sibling path used for atomic writes."""
    return path.with_name(f"{path.name}.tmp")


def _replace_file(source: Path, destination: Path) -> None:
    """Atomically replace a completed output file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)


def _remove_paths(paths: Iterable[Path]) -> None:
    """Remove temporary files left by an interrupted run."""
    for path in paths:
        path.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON using a temporary file and atomic replacement."""
    temporary_path = _temporary_path(path)
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    _replace_file(temporary_path, path)


def _print_progress(
    split: str,
    completed: int,
    total: int,
    start_time: float,
) -> None:
    """Print periodic preparation progress without external dependencies."""
    if completed != total and completed % 1024 != 0:
        return

    elapsed = time.perf_counter() - start_time
    rate = completed / elapsed if elapsed > 0 else 0.0
    print(
        f"split={split} prepared={completed}/{total} "
        f"images_per_second={rate:.2f}"
    )
