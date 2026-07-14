"""Run Task 1.2 inference on the runtime predict split."""

from __future__ import annotations

import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from src.data_io import (
    IMAGE_COLUMN_CANDIDATES,
    ROW_ID_COLUMN_CANDIDATES,
    find_parquet_files,
    resolve_column,
)
from src.preprocessing import decode_and_resize
from src.task02 import PlainCNN, set_reproducibility


BATCH_SIZE = 256
WORKERS = 8
THREADS = 8
SEED = 42


def parse_args() -> argparse.Namespace:
    """Parse the evaluator-provided runtime limit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    """Load the trained model and write predictions.csv."""
    args = parse_args()
    if args.timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    started = time.monotonic()
    deadline = started + args.timeout_seconds - min(10.0, args.timeout_seconds * 0.10)
    set_reproducibility(SEED, THREADS)

    solution_dir = Path(__file__).resolve().parent
    model_path = solution_dir / "artifacts" / "task02" / "model" / "model.pt"
    predict_dir = solution_dir / "data" / "predict"
    output_path = solution_dir / "artifacts" / "task02" / "predictions.csv"

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if checkpoint.get("threshold") is None:
        raise ValueError("The model checkpoint has no calibrated threshold")
    model = PlainCNN(**checkpoint["model"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    image_size = int(checkpoint["image_size"])
    mean = np.asarray(checkpoint["normalization"]["mean"], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(checkpoint["normalization"]["std"], dtype=np.float32).reshape(1, 3, 1, 1)
    threshold = float(checkpoint["threshold"])
    predictions: list[tuple[int, int]] = []

    with ThreadPoolExecutor(max_workers=WORKERS) as executor, torch.inference_mode():
        for parquet_path in find_parquet_files(predict_dir):
            parquet = pq.ParquetFile(parquet_path)
            image_column = resolve_column(parquet.schema_arrow.names, IMAGE_COLUMN_CANDIDATES)
            row_column = resolve_column(parquet.schema_arrow.names, ROW_ID_COLUMN_CANDIDATES)

            for batch in parquet.iter_batches(
                batch_size=BATCH_SIZE,
                columns=[row_column, image_column],
            ):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Prediction deadline reached")

                row_ids = [int(value) for value in batch.column(0).to_pylist()]
                encoded = [bytes(value) for value in batch.column(1).to_pylist()]
                images = np.stack(
                    list(executor.map(decode_and_resize, encoded, [image_size] * len(encoded)))
                ).astype(np.float32)
                images = (images / 255.0 - mean) / std
                probabilities = torch.softmax(model(torch.from_numpy(images)), dim=1)[:, 1]
                labels = (probabilities.numpy() >= threshold).astype(np.uint8)
                predictions.extend(zip(row_ids, labels.tolist()))

    if not predictions:
        raise ValueError(f"No prediction rows found in {predict_dir}")
    predictions.sort(key=lambda item: item[0])
    if len({row_id for row_id, _ in predictions}) != len(predictions):
        raise ValueError("Duplicate row_id values found in predict split")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "predicted_label"])
        writer.writerows(predictions)
    temporary.replace(output_path)

    print(
        f"prediction status=complete samples={len(predictions)} "
        f"threshold={threshold:.6f} elapsed_seconds={time.monotonic() - started:.3f} "
        f"output_path={output_path}"
    )


if __name__ == "__main__":
    main()
