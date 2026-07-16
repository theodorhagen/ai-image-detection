"""Explainability and failure-analysis utilities for the final image detector."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


SOURCE_CLASS_NAMES = {
    0: "Real",
    1: "SD 2.1",
    2: "SDXL",
    3: "SD 3",
    4: "DALL-E 3",
    5: "Midjourney",
}

CATEGORY_ORDER = (
    "true_positive",
    "true_negative",
    "false_positive",
    "false_negative",
)

CATEGORY_NAMES = {
    "true_positive": "True positive",
    "true_negative": "True negative",
    "false_positive": "False positive",
    "false_negative": "False negative",
}


def load_source_classes(prepared_dir: Path, split: str) -> np.ndarray:
    """Load the original source classes for one prepared split."""
    path = prepared_dir / f"{split}_source_classes.npy"
    if not path.exists():
        raise FileNotFoundError(f"Prepared source classes are missing: {path}")
    source_classes = np.asarray(np.load(path)).reshape(-1).astype(np.int64)
    if source_classes.size == 0:
        raise ValueError(f"Prepared source-class array is empty: {split}")
    return source_classes


def prediction_categories(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Return TP, TN, FP, or FN for every sample."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    predictions = np.asarray(scores, dtype=np.float64).reshape(-1) >= threshold
    categories = np.empty(labels.size, dtype=object)
    categories[(labels == 1) & predictions] = "true_positive"
    categories[(labels == 0) & ~predictions] = "true_negative"
    categories[(labels == 0) & predictions] = "false_positive"
    categories[(labels == 1) & ~predictions] = "false_negative"
    return categories


def select_examples(
    split: str,
    labels: np.ndarray,
    source_classes: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    examples_per_category: int,
) -> list[dict[str, Any]]:
    """Select confident and boundary examples without manual cherry-picking."""
    if examples_per_category <= 0:
        return []

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    source_classes = np.asarray(source_classes, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not (labels.size == source_classes.size == scores.size):
        raise ValueError("Labels, source classes, and scores must have equal length")

    categories = prediction_categories(labels, scores, threshold)
    margins = scores - threshold
    records: list[dict[str, Any]] = []

    for category in CATEGORY_ORDER:
        category_indices = np.flatnonzero(categories == category)
        if category_indices.size == 0:
            continue

        ranked: list[tuple[str, int]] = [
            (
                "confident",
                int(category_indices[np.argmax(np.abs(margins[category_indices]))]),
            )
        ]
        if examples_per_category >= 2:
            ranked.append(
                (
                    "boundary",
                    int(category_indices[np.argmin(np.abs(margins[category_indices]))]),
                )
            )

        if examples_per_category > 2:
            ordered = category_indices[np.argsort(np.abs(margins[category_indices]))]
            positions = np.linspace(0, ordered.size - 1, examples_per_category, dtype=int)
            ranked.extend((f"rank_{rank + 1}", int(ordered[position])) for rank, position in enumerate(positions))

        used: set[int] = set()
        for selection_type, sample_index in ranked:
            if sample_index in used or len(used) >= examples_per_category:
                continue
            used.add(sample_index)
            predicted_label = int(scores[sample_index] >= threshold)
            source_class = int(source_classes[sample_index])
            records.append(
                {
                    "split": split,
                    "sample_index": sample_index,
                    "category": category,
                    "category_name": CATEGORY_NAMES[category],
                    "selection_type": selection_type,
                    "true_label": int(labels[sample_index]),
                    "predicted_label": predicted_label,
                    "source_class": source_class,
                    "source_name": SOURCE_CLASS_NAMES.get(source_class, str(source_class)),
                    "ai_probability": float(scores[sample_index]),
                    "threshold": float(threshold),
                    "signed_margin": float(margins[sample_index]),
                    "absolute_margin": float(abs(margins[sample_index])),
                }
            )

    return records


def source_class_metrics(
    split: str,
    labels: np.ndarray,
    source_classes: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    """Summarize predictions separately for real images and each generator."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    source_classes = np.asarray(source_classes, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    predictions = scores >= threshold
    rows: list[dict[str, Any]] = []

    for source_class in sorted(np.unique(source_classes).tolist()):
        mask = source_classes == source_class
        group_labels = labels[mask]
        group_predictions = predictions[mask]
        binary_label = int(round(float(np.mean(group_labels))))
        predicted_ai_rate = float(np.mean(group_predictions))
        rows.append(
            {
                "split": split,
                "source_class": int(source_class),
                "source_name": SOURCE_CLASS_NAMES.get(int(source_class), str(source_class)),
                "binary_label": binary_label,
                "n_samples": int(np.sum(mask)),
                "mean_ai_probability": float(np.mean(scores[mask])),
                "predicted_ai_rate": predicted_ai_rate,
                "operating_metric": "fpr_real" if binary_label == 0 else "recall_ai",
                "operating_metric_value": predicted_ai_rate,
                "accuracy": float(np.mean(group_predictions == group_labels)),
            }
        )

    return rows


def _linear_classifier(model: nn.Module) -> nn.Linear:
    """Return the final two-class linear layer used by PlainCNN."""
    classifier = getattr(model, "classifier", None)
    if not isinstance(classifier, nn.Sequential) or not classifier:
        raise TypeError("Expected model.classifier to be a non-empty nn.Sequential")
    linear = classifier[-1]
    if not isinstance(linear, nn.Linear) or linear.out_features != 2:
        raise TypeError("Signed CAM requires a final nn.Linear layer with two outputs")
    return linear


@torch.inference_mode()
def signed_class_activation_map(
    model: nn.Module,
    normalized_image: torch.Tensor,
) -> dict[str, Any]:
    """Compute an exact signed CAM for AI versus real at the final feature map."""
    if normalized_image.ndim != 3:
        raise ValueError("Expected one normalized image with shape C x H x W")

    model.eval()
    features_module = getattr(model, "features", None)
    if not isinstance(features_module, nn.Module):
        raise TypeError("Signed CAM requires model.features")

    feature_maps = features_module(normalized_image.unsqueeze(0))
    logits = model.classifier(feature_maps)
    linear = _linear_classifier(model)
    weight_difference = linear.weight[1] - linear.weight[0]
    bias_difference = linear.bias[1] - linear.bias[0] if linear.bias is not None else 0.0
    cam = torch.einsum("c,chw->hw", weight_difference, feature_maps[0])

    logit_difference = logits[0, 1] - logits[0, 0]
    reconstructed_difference = cam.mean() + bias_difference
    reconstruction_error = torch.abs(logit_difference - reconstructed_difference)
    probabilities = torch.softmax(logits, dim=1)[0]

    return {
        "map": cam.cpu().numpy().astype(np.float32),
        "ai_probability": float(probabilities[1].item()),
        "real_probability": float(probabilities[0].item()),
        "logit_difference": float(logit_difference.item()),
        "reconstructed_logit_difference": float(reconstructed_difference.item()),
        "reconstruction_error": float(reconstruction_error.item()),
    }


def resize_map(values: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a two-dimensional explanation map with bilinear interpolation."""
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32))[None, None]
    resized = F.interpolate(
        tensor,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return resized[0, 0].numpy()


def normalized_signed_map(values: np.ndarray) -> np.ndarray:
    """Scale one signed map to the interval [-1, 1] for visualization."""
    values = np.asarray(values, dtype=np.float32)
    scale = float(np.max(np.abs(values)))
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.zeros_like(values)
    return values / scale


def map_statistics(values: np.ndarray) -> dict[str, float]:
    """Return simple spatial summaries of absolute and signed CAM energy."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Expected a two-dimensional explanation map")

    height, width = values.shape
    absolute = np.abs(values)
    total = float(np.sum(absolute))
    positive = float(np.sum(np.clip(values, 0.0, None)))
    negative = float(np.sum(np.clip(-values, 0.0, None)))

    top = height // 4
    bottom = height - top
    left = width // 4
    right = width - left
    center = float(np.sum(absolute[top:bottom, left:right]))

    border_mask = np.ones((height, width), dtype=bool)
    border_y = max(1, height // 8)
    border_x = max(1, width // 8)
    border_mask[border_y : height - border_y, border_x : width - border_x] = False
    border = float(np.sum(absolute[border_mask]))

    signed_total = positive + negative
    return {
        "cam_absolute_energy": total,
        "cam_positive_fraction": 0.0 if signed_total == 0 else positive / signed_total,
        "cam_negative_fraction": 0.0 if signed_total == 0 else negative / signed_total,
        "cam_center_fraction": 0.0 if total == 0 else center / total,
        "cam_border_fraction": 0.0 if total == 0 else border / total,
    }


def _patch_positions(length: int, patch_size: int, stride: int) -> list[int]:
    """Return deterministic patch starts that also cover the final pixels."""
    patch_size = min(patch_size, length)
    final_start = length - patch_size
    positions = list(range(0, final_start + 1, stride))
    if not positions or positions[-1] != final_start:
        positions.append(final_start)
    return positions


@torch.inference_mode()
def occlusion_sensitivity(
    model: nn.Module,
    normalized_image: torch.Tensor,
    patch_size: int,
    stride: int,
    batch_size: int,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Measure probability changes after replacing patches by the train mean."""
    if normalized_image.ndim != 3:
        raise ValueError("Expected one normalized image with shape C x H x W")
    if patch_size <= 0 or stride <= 0 or batch_size <= 0:
        raise ValueError("Patch size, stride, and batch size must be positive")

    model.eval()
    _, height, width = normalized_image.shape
    patch_height = min(patch_size, height)
    patch_width = min(patch_size, width)
    y_positions = _patch_positions(height, patch_height, stride)
    x_positions = _patch_positions(width, patch_width, stride)
    patches = [(y, x) for y in y_positions for x in x_positions]

    base_probability = float(
        torch.softmax(model(normalized_image.unsqueeze(0)), dim=1)[0, 1].item()
    )
    accumulated = np.zeros((height, width), dtype=np.float64)
    counts = np.zeros((height, width), dtype=np.float64)

    for start in range(0, len(patches), batch_size):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Explainability deadline reached during occlusion analysis")

        current = patches[start : start + batch_size]
        variants = normalized_image.unsqueeze(0).repeat(len(current), 1, 1, 1)
        for row, (y, x) in enumerate(current):
            variants[row, :, y : y + patch_height, x : x + patch_width] = 0.0

        probabilities = torch.softmax(model(variants), dim=1)[:, 1].cpu().numpy()
        changes = base_probability - probabilities
        for change, (y, x) in zip(changes.tolist(), current):
            accumulated[y : y + patch_height, x : x + patch_width] += change
            counts[y : y + patch_height, x : x + patch_width] += 1.0

    sensitivity = np.divide(
        accumulated,
        counts,
        out=np.zeros_like(accumulated),
        where=counts > 0,
    ).astype(np.float32)
    return {
        "map": sensitivity,
        "base_ai_probability": base_probability,
        "patch_count": len(patches),
        "patch_size": patch_size,
        "stride": stride,
        "replacement": "zero in normalized space (training-channel mean)",
    }


def pearson_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    """Return Pearson correlation, or None for constant maps."""
    first_values = np.asarray(first, dtype=np.float64).reshape(-1)
    second_values = np.asarray(second, dtype=np.float64).reshape(-1)
    if first_values.size != second_values.size:
        raise ValueError("Maps must contain the same number of values")
    if np.std(first_values) <= 1e-12 or np.std(second_values) <= 1e-12:
        return None
    return float(np.corrcoef(first_values, second_values)[0, 1])


def choose_occlusion_records_by_category(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select one confident example for TP, TN, FP, and FN.

    The original validation split is preferred so that all four rows belong to
    the same evaluation setting. If a category is unavailable there, the
    augmented validation split is used as a deterministic fallback.
    """
    records = list(records)
    selected: list[dict[str, Any]] = []

    for category in CATEGORY_ORDER:
        candidates = [
            record
            for record in records
            if str(record["category"]) == category
            and str(record["selection_type"]) == "confident"
        ]
        if not candidates:
            candidates = [
                record
                for record in records
                if str(record["category"]) == category
            ]
        if not candidates:
            continue

        candidates.sort(
            key=lambda record: (
                0 if str(record["split"]) == "validation" else 1,
                -float(record["absolute_margin"]),
                int(record["sample_index"]),
            )
        )
        selected.append(candidates[0])

    return selected


def save_cam_figure(
    records: list[dict[str, Any]],
    output_path: Path,
    title: str,
) -> None:
    """Save original images and signed CAM overlays for one split."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        _save_empty_figure(output_path, title, "No examples were available.")
        return

    figure = plt.figure(figsize=(10.5, max(2.7 * len(records) + 1.2, 5.0)))
    grid = figure.add_gridspec(
        nrows=2 * len(records),
        ncols=2,
        height_ratios=[0.22, 1.0] * len(records),
        hspace=0.28,
        wspace=0.16,
    )

    figure.suptitle(title, fontsize=14, y=0.992)
    figure.text(
        0.5,
        0.975,
        "Left: original image. Right: signed CAM overlay (blue supports real, red supports AI).",
        ha="center",
        va="top",
        fontsize=10,
    )

    for row, record in enumerate(records):
        header_ax = figure.add_subplot(grid[2 * row, :])
        header_ax.axis("off")
        header_ax.text(
            0.01,
            0.95,
            _record_title(record),
            ha="left",
            va="top",
            fontsize=9,
            wrap=True,
        )

        image = np.asarray(record["display_image"])
        cam = normalized_signed_map(np.asarray(record["cam_resized"]))

        original_ax = figure.add_subplot(grid[2 * row + 1, 0])
        original_ax.imshow(image)
        original_ax.axis("off")
        original_ax.set_title("Original", fontsize=10, pad=4)

        cam_ax = figure.add_subplot(grid[2 * row + 1, 1])
        cam_ax.imshow(image)
        cam_ax.imshow(cam, cmap="coolwarm", vmin=-1.0, vmax=1.0, alpha=0.50)
        cam_ax.axis("off")
        cam_ax.set_title("Signed CAM", fontsize=10, pad=4)

    figure.subplots_adjust(top=0.955, bottom=0.02, left=0.06, right=0.98)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_individual_cam_figure(record: dict[str, Any], output_path: Path) -> None:
    """Save one report-ready original and signed-CAM comparison."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = np.asarray(record["display_image"])
    cam = normalized_signed_map(np.asarray(record["cam_resized"]))

    figure = plt.figure(figsize=(8.8, 4.8))
    grid = figure.add_gridspec(nrows=2, ncols=2, height_ratios=[0.26, 1.0], hspace=0.12, wspace=0.12)

    header_ax = figure.add_subplot(grid[0, :])
    header_ax.axis("off")
    header_ax.text(0.01, 0.98, _record_title(record), ha="left", va="top", fontsize=11)

    original_ax = figure.add_subplot(grid[1, 0])
    original_ax.imshow(image)
    original_ax.axis("off")
    original_ax.set_title("Original", fontsize=11, pad=4)

    cam_ax = figure.add_subplot(grid[1, 1])
    cam_ax.imshow(image)
    cam_ax.imshow(cam, cmap="coolwarm", vmin=-1.0, vmax=1.0, alpha=0.50)
    cam_ax.axis("off")
    cam_ax.set_title("Signed CAM", fontsize=11, pad=4)

    figure.subplots_adjust(top=0.96, bottom=0.04, left=0.04, right=0.98)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_occlusion_figure(
    records: list[dict[str, Any]],
    output_path: Path,
    title: str,
) -> None:
    """Save original, signed CAM, and occlusion maps side by side."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        _save_empty_figure(output_path, title, "No occlusion examples were selected.")
        return

    figure = plt.figure(figsize=(12.5, max(3.1 * len(records) + 0.8, 5.0)))
    grid = figure.add_gridspec(
        nrows=2 * len(records),
        ncols=3,
        height_ratios=[0.14, 1.0] * len(records),
        hspace=0.18,
        wspace=0.06,
    )

    figure.suptitle(title, fontsize=14, y=0.992)
    figure.text(
        0.5,
        0.975,
        "Columns: original image, signed CAM, occlusion sensitivity. Blue supports real; red supports AI.",
        ha="center",
        va="top",
        fontsize=10,
    )

    for row, record in enumerate(records):
        header_ax = figure.add_subplot(grid[2 * row, :])
        header_ax.axis("off")
        header_ax.text(
            0.01,
            0.95,
            _record_title(record),
            ha="left",
            va="top",
            fontsize=9,
            wrap=True,
        )

        image = np.asarray(record["display_image"])
        cam = normalized_signed_map(np.asarray(record["cam_resized"]))
        occlusion = normalized_signed_map(np.asarray(record["occlusion_map"]))

        original_ax = figure.add_subplot(grid[2 * row + 1, 0])
        original_ax.imshow(image)
        original_ax.axis("off")
        original_ax.set_title("Original", fontsize=10, pad=4)

        cam_ax = figure.add_subplot(grid[2 * row + 1, 1])
        cam_ax.imshow(image)
        cam_ax.imshow(cam, cmap="coolwarm", vmin=-1.0, vmax=1.0, alpha=0.50)
        cam_ax.axis("off")
        cam_ax.set_title("Signed CAM", fontsize=10, pad=4)

        occlusion_ax = figure.add_subplot(grid[2 * row + 1, 2])
        occlusion_ax.imshow(image)
        occlusion_ax.imshow(
            occlusion,
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            alpha=0.50,
        )
        occlusion_ax.axis("off")
        correlation = record.get("cam_occlusion_correlation")
        correlation_text = "undefined" if correlation is None else f"r={correlation:.3f}"
        occlusion_ax.set_title(f"Occlusion sensitivity ({correlation_text})", fontsize=10, pad=4)

    figure.subplots_adjust(
        top=0.94,
        bottom=0.015,
        left=0.025,
        right=0.99,
    )
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert explanation records to a CSV-safe DataFrame."""
    excluded = {"display_image", "cam_map", "cam_resized", "occlusion_map"}
    rows = [
        {key: value for key, value in record.items() if key not in excluded}
        for record in records
    ]
    return pd.DataFrame(rows)


def write_frame_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _record_title(record: dict[str, Any]) -> str:
    true_name = "AI" if int(record["true_label"]) == 1 else "Real"
    predicted_name = "AI" if int(record["predicted_label"]) == 1 else "Real"
    selection_name = {"confident": "conf.", "boundary": "near thr."}.get(
        str(record["selection_type"]),
        str(record["selection_type"]),
    )
    return (
        f"{record['category_name']} ({selection_name}) | src={record['source_name']}\n"
        f"true={true_name}, pred={predicted_name} | "
        f"p(AI)={float(record['ai_probability']):.3f}, "
        f"thr={float(record['threshold']):.3f}"
    )


def _save_empty_figure(path: Path, title: str, message: str) -> None:
    figure = plt.figure(figsize=(8, 3))
    plt.title(title)
    plt.text(0.5, 0.5, message, ha="center", va="center")
    plt.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)