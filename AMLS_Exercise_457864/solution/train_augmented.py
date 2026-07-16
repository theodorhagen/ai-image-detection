"""Fine-tune the Task 1.2 CNN with lightweight robustness augmentations."""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from src.task02 import (
    PlainCNN,
    PreparedDataset,
    atomic_torch_save,
    evaluate_scores,
    load_labels,
    load_preparation_config,
    make_loader,
    score_loader,
    select_threshold,
    set_reproducibility,
    split_train_monitor,
    write_json,
)


EPOCHS = 10
BATCH_SIZE = 256
EVALUATION_BATCH_SIZE = 256
WORKERS = 0
THREADS = 8
MAX_LEARNING_RATE = 8e-4
WEIGHT_DECAY = 1e-4
MONITOR_FRACTION = 0.10
MONITOR_FPR = 0.19
CALIBRATION_FPR = 0.19
VALIDATION_FPR_LIMIT = 0.20
PATIENCE = 3
MIN_EPOCHS = 2
SEED = 42

# Probabilities are independent. Approximately one quarter of samples remain
# unchanged, so clean-image performance is not discarded during fine-tuning.
DOWNSCALE_PROBABILITY = 0.35
BLUR_PROBABILITY = 0.25
QUANTIZATION_PROBABILITY = 0.35
NOISE_PROBABILITY = 0.15


def parse_args() -> argparse.Namespace:
    """Parse the evaluator-provided runtime limit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=float, required=True)
    return parser.parse_args()


def build_dataset(
    prepared_dir: Path,
    split: str,
    mean: list[float],
    std: list[float],
    indices: np.ndarray | None = None,
) -> PreparedDataset:
    """Create one prepared dataset."""
    return PreparedDataset(prepared_dir, split, mean, std, indices)


def augmentation_config() -> dict[str, Any]:
    """Return the augmentation settings stored with the final model."""
    return {
        "downscale_probability": DOWNSCALE_PROBABILITY,
        "downscale_factors": [0.50, 0.65, 0.80],
        "blur_probability": BLUR_PROBABILITY,
        "blur_kernel": 3,
        "quantization_probability": QUANTIZATION_PROBABILITY,
        "quantization_levels": [32, 64, 128],
        "noise_probability": NOISE_PROBABILITY,
        "noise_standard_deviations": [0.005, 0.010, 0.020],
    }


def checkpoint_payload(
    model: nn.Module,
    model_config: dict[str, Any],
    mean: list[float],
    std: list[float],
    image_size: int,
    epoch: int,
    threshold: float | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a portable CPU checkpoint for Task 1.3."""
    return {
        "model": model_config,
        "state_dict": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        "normalization": {"mean": mean, "std": std},
        "image_size": image_size,
        "epoch": epoch,
        "threshold": threshold,
        "metrics": metrics,
        "augmentation": augmentation_config(),
        "continued_from": "artifacts/task02/model/model.pt",
    }


def _sample_mask(
    batch_size: int,
    probability: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample a broadcastable per-image transformation mask."""
    return torch.rand((batch_size, 1, 1, 1), generator=generator) < probability


def augment_batch(
    normalized_images: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Apply cheap scaling, blur, compression-like quantization, and noise."""
    images = (normalized_images * std + mean).clamp(0.0, 1.0)
    batch_size, _, height, width = images.shape

    scale_factors = (0.50, 0.65, 0.80)
    scale_index = int(torch.randint(len(scale_factors), (1,), generator=generator).item())
    scale = scale_factors[scale_index]
    reduced_height = max(8, int(round(height * scale)))
    reduced_width = max(8, int(round(width * scale)))
    scaled = F.interpolate(
        images,
        size=(reduced_height, reduced_width),
        mode="bilinear",
        align_corners=False,
    )
    scaled = F.interpolate(
        scaled,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    scale_mask = _sample_mask(batch_size, DOWNSCALE_PROBABILITY, generator)
    images = torch.where(scale_mask, scaled, images)

    blurred = F.avg_pool2d(
        F.pad(images, (1, 1, 1, 1), mode="reflect"),
        kernel_size=3,
        stride=1,
    )
    blur_mask = _sample_mask(batch_size, BLUR_PROBABILITY, generator)
    images = torch.where(blur_mask, blurred, images)

    quantization_levels = (32, 64, 128)
    level_index = int(torch.randint(len(quantization_levels), (1,), generator=generator).item())
    levels = quantization_levels[level_index]
    quantized = torch.round(images * (levels - 1)) / float(levels - 1)
    quantization_mask = _sample_mask(batch_size, QUANTIZATION_PROBABILITY, generator)
    images = torch.where(quantization_mask, quantized, images)

    noise_levels = (0.005, 0.010, 0.020)
    noise_index = int(torch.randint(len(noise_levels), (1,), generator=generator).item())
    noise = torch.randn(images.shape, generator=generator, dtype=images.dtype)
    noisy = (images + noise * noise_levels[noise_index]).clamp(0.0, 1.0)
    noise_mask = _sample_mask(batch_size, NOISE_PROBABILITY, generator)
    images = torch.where(noise_mask, noisy, images)

    return (images - mean) / std


def train_augmented_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    deadline: float,
    mean: torch.Tensor,
    std: torch.Tensor,
    generator: torch.Generator,
) -> tuple[float, bool]:
    """Train one epoch with online augmentations and deadline handling."""
    model.train()
    loss_sum = 0.0
    samples = 0

    for images, labels in loader:
        if time.monotonic() >= deadline:
            return loss_sum / max(samples, 1), True

        images = augment_batch(images, mean, std, generator)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_size = labels.shape[0]
        loss_sum += float(loss.item()) * batch_size
        samples += batch_size

    return loss_sum / samples, False


@torch.inference_mode()
def score_augmented_loader(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    mean: torch.Tensor,
    std: torch.Tensor,
    seed: int,
    criterion: nn.Module | None = None,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Score one loader using a fixed deterministic augmentation sequence."""
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    scores: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    loss_sum = 0.0
    samples = 0

    for images, labels in loader:
        images = augment_batch(images, mean, std, generator)
        logits = model(images)
        if criterion is not None:
            loss_sum += float(criterion(logits, labels).item()) * labels.shape[0]
            samples += labels.shape[0]
        scores.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        labels_all.append(labels.cpu().numpy())

    mean_loss = None if criterion is None else loss_sum / samples
    return np.concatenate(scores), np.concatenate(labels_all), mean_loss


def robust_threshold(
    clean_labels: np.ndarray,
    clean_scores: np.ndarray,
    augmented_labels: np.ndarray,
    augmented_scores: np.ndarray,
    max_fpr: float,
) -> dict[str, Any]:
    """Choose one conservative threshold that satisfies both calibration splits."""
    clean = select_threshold(clean_labels, clean_scores, max_fpr)
    augmented = select_threshold(augmented_labels, augmented_scores, max_fpr)
    threshold = max(float(clean["threshold"]), float(augmented["threshold"]))
    return {
        "threshold": threshold,
        "target_max_fpr": max_fpr,
        "clean_candidate": clean,
        "augmented_candidate": augmented,
    }


def write_comparison_csv(
    path: Path,
    task2_metrics: dict[str, Any] | None,
    task3_metrics: dict[str, Any],
) -> None:
    """Write the report-ready Task 2 versus Task 3 comparison table."""
    fields = [
        "model",
        "split",
        "fpr_real",
        "recall_ai",
        "precision_ai",
        "f1_ai",
        "roc_auc",
        "pr_auc",
    ]
    rows: list[dict[str, Any]] = []
    for model_name, metrics_by_split in (
        ("task02_cnn", task2_metrics or {}),
        ("task03_augmented_cnn", task3_metrics),
    ):
        for split in ("validation", "validation_augmented"):
            metrics = metrics_by_split.get(split)
            if metrics is None:
                continue
            rows.append(
                {
                    "model": model_name,
                    "split": split,
                    **{field: metrics[field] for field in fields[2:]},
                }
            )

    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    """Continue from Task 1.2 and produce the robust Task 1.3 model."""
    args = parse_args()
    if args.timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    started = time.monotonic()
    reserve = min(120.0, args.timeout_seconds * 0.20)
    training_deadline = started + args.timeout_seconds - reserve
    set_reproducibility(SEED, THREADS)

    solution_dir = Path(__file__).resolve().parent
    prepared_dir = solution_dir / "artifacts" / "task02" / "prepared"
    task2_model_path = solution_dir / "artifacts" / "task02" / "model" / "model.pt"
    output_dir = solution_dir / "artifacts" / "task03" / "model"
    model_path = output_dir / "model.pt"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not task2_model_path.exists():
        raise FileNotFoundError(
            "Task 1.2 checkpoint not found. Run train.py before train_augmented.py."
        )

    mean, std, image_size = load_preparation_config(prepared_dir)
    task2_checkpoint = torch.load(task2_model_path, map_location="cpu", weights_only=False)
    model_config = dict(task2_checkpoint["model"])
    model = PlainCNN(**model_config)
    model.load_state_dict(task2_checkpoint["state_dict"])

    labels = load_labels(prepared_dir, "train")
    train_indices, monitor_indices = split_train_monitor(
        labels,
        MONITOR_FRACTION,
        SEED,
    )
    train_loader = make_loader(
        build_dataset(prepared_dir, "train", mean, std, train_indices),
        BATCH_SIZE,
        True,
        WORKERS,
        SEED,
    )
    monitor_loader = make_loader(
        build_dataset(prepared_dir, "train", mean, std, monitor_indices),
        EVALUATION_BATCH_SIZE,
        False,
        WORKERS,
        SEED,
    )

    counts = np.bincount(labels[train_indices], minlength=2).astype(np.float64)
    class_weights = train_indices.size / (2.0 * counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32)
    )
    optimizer = AdamW(
        model.parameters(),
        lr=MAX_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=MAX_LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.20,
        anneal_strategy="cos",
        div_factor=5.0,
        final_div_factor=50.0,
    )

    mean_tensor = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    training_generator = torch.Generator().manual_seed(SEED + 1)

    print(f"model=augmented_cnn parameters={sum(p.numel() for p in model.parameters())}")
    print(f"continued_from={task2_model_path}")
    print(f"train_samples={train_indices.size} monitor_samples={monitor_indices.size}")
    print(f"class_weights={class_weights.tolist()}")

    history: list[dict[str, Any]] = []
    best_key: tuple[float, float, float] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    training_seconds = 0.0

    if model_path.exists():
        model_path.unlink()

    for epoch in range(1, EPOCHS + 1):
        if time.monotonic() >= training_deadline:
            print(f"epoch={epoch} status=skipped reason=deadline")
            break

        epoch_started = time.monotonic()
        train_loss, interrupted = train_augmented_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            training_deadline,
            mean_tensor,
            std_tensor,
            training_generator,
        )
        epoch_seconds = time.monotonic() - epoch_started
        training_seconds += epoch_seconds

        if interrupted:
            print(f"epoch={epoch} status=interrupted reason=deadline")
            break

        clean_scores, monitor_labels, clean_loss = score_loader(
            model,
            monitor_loader,
            criterion,
        )
        augmented_scores, augmented_labels, augmented_loss = score_augmented_loader(
            model,
            monitor_loader,
            mean_tensor,
            std_tensor,
            SEED + 1000,
            criterion,
        )
        selected = robust_threshold(
            monitor_labels,
            clean_scores,
            augmented_labels,
            augmented_scores,
            MONITOR_FPR,
        )
        clean_metrics = evaluate_scores(
            monitor_labels,
            clean_scores,
            selected["threshold"],
        )
        augmented_metrics = evaluate_scores(
            augmented_labels,
            augmented_scores,
            selected["threshold"],
        )

        key = (
            min(clean_metrics["recall_ai"], augmented_metrics["recall_ai"]),
            0.5 * (clean_metrics["recall_ai"] + augmented_metrics["recall_ai"]),
            0.5 * (clean_metrics["roc_auc"] + augmented_metrics["roc_auc"]),
        )
        improved = best_key is None or key > best_key

        if improved:
            best_key = key
            best_epoch = epoch
            epochs_without_improvement = 0
            atomic_torch_save(
                model_path,
                checkpoint_payload(
                    model,
                    model_config,
                    mean,
                    std,
                    image_size,
                    epoch,
                ),
            )
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "monitor_clean_loss": clean_loss,
                "monitor_augmented_loss": augmented_loss,
                "monitor_threshold": selected["threshold"],
                "monitor_clean": clean_metrics,
                "monitor_augmented": augmented_metrics,
                "learning_rate_end": optimizer.param_groups[0]["lr"],
                "epoch_seconds": epoch_seconds,
                "improved": improved,
            }
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"clean_fpr={clean_metrics['fpr_real']:.4f} "
            f"clean_recall={clean_metrics['recall_ai']:.4f} "
            f"augmented_fpr={augmented_metrics['fpr_real']:.4f} "
            f"augmented_recall={augmented_metrics['recall_ai']:.4f} "
            f"seconds={epoch_seconds:.3f} improved={improved}"
        )

        if epoch >= MIN_EPOCHS and epochs_without_improvement >= PATIENCE:
            print(f"training status=early_stopped epoch={epoch}")
            break

    if not model_path.exists():
        raise TimeoutError("No complete augmented-training epoch finished before the deadline")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = PlainCNN(**checkpoint["model"])
    model.load_state_dict(checkpoint["state_dict"])

    scores: dict[str, np.ndarray] = {}
    labels_by_split: dict[str, np.ndarray] = {}
    for split in (
        "calibration",
        "validation",
        "calibration_augmented",
        "validation_augmented",
    ):
        image_path = prepared_dir / f"{split}_images.npy"
        if not image_path.exists():
            raise FileNotFoundError(f"Prepared split missing: {split}")
        loader = make_loader(
            build_dataset(prepared_dir, split, mean, std),
            EVALUATION_BATCH_SIZE,
            False,
            WORKERS,
            SEED,
        )
        split_scores, split_labels, _ = score_loader(model, loader)
        scores[split] = split_scores
        labels_by_split[split] = split_labels

    calibrated = robust_threshold(
        labels_by_split["calibration"],
        scores["calibration"],
        labels_by_split["calibration_augmented"],
        scores["calibration_augmented"],
        CALIBRATION_FPR,
    )
    metrics = {
        split: evaluate_scores(
            labels_by_split[split],
            split_scores,
            calibrated["threshold"],
        )
        for split, split_scores in scores.items()
    }
    validation_compliant = (
        metrics["validation"]["fpr_real"] <= VALIDATION_FPR_LIMIT
        and metrics["validation_augmented"]["fpr_real"] <= VALIDATION_FPR_LIMIT
    )

    final_checkpoint = checkpoint_payload(
        model,
        model_config,
        mean,
        std,
        image_size,
        best_epoch,
        calibrated["threshold"],
        metrics,
    )
    atomic_torch_save(model_path, final_checkpoint)

    configuration = {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "evaluation_batch_size": EVALUATION_BATCH_SIZE,
        "workers": WORKERS,
        "threads": THREADS,
        "max_learning_rate": MAX_LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "monitor_fraction": MONITOR_FRACTION,
        "monitor_fpr": MONITOR_FPR,
        "calibration_fpr": CALIBRATION_FPR,
        "validation_fpr_limit": VALIDATION_FPR_LIMIT,
        "seed": SEED,
        "augmentation": augmentation_config(),
    }
    result = {
        "model": "augmented_cnn",
        "continued_from": str(task2_model_path),
        "best_epoch": best_epoch,
        "training_seconds": training_seconds,
        "total_seconds": time.monotonic() - started,
        "threshold": calibrated,
        "metrics": metrics,
        "task02_baseline_metrics": task2_checkpoint.get("metrics"),
        "validation_compliant": validation_compliant,
        "configuration": configuration,
        "history": history,
    }
    write_json(output_dir / "metrics.json", result)
    write_comparison_csv(
        output_dir / "comparison.csv",
        task2_checkpoint.get("metrics"),
        metrics,
    )

    clean_validation = metrics["validation"]
    augmented_validation = metrics["validation_augmented"]
    print(
        f"training status=complete best_epoch={best_epoch} "
        f"threshold={calibrated['threshold']:.6f} "
        f"validation_fpr={clean_validation['fpr_real']:.4f} "
        f"validation_recall={clean_validation['recall_ai']:.4f} "
        f"augmented_fpr={augmented_validation['fpr_real']:.4f} "
        f"augmented_recall={augmented_validation['recall_ai']:.4f} "
        f"validation_compliant={validation_compliant} "
        f"model_path={model_path}"
    )


if __name__ == "__main__":
    main()
