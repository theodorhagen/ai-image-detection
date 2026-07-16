"""Train, calibrate, and validate the final Task 1.2 model."""

from __future__ import annotations

import argparse
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
    train_epoch,
    write_json,
)


EPOCHS = 15
BATCH_SIZE = 256
EVALUATION_BATCH_SIZE = 256
WORKERS = 0
THREADS = 8
BASE_CHANNELS = 16
DROPOUT = 0.20
MAX_LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-4
MONITOR_FRACTION = 0.10
MONITOR_FPR = 0.17
CALIBRATION_FPR = 0.17
VALIDATION_FPR_LIMIT = 0.20
PATIENCE = 5
MIN_EPOCHS = 3
SEED = 42


def parse_args() -> argparse.Namespace:
    """Parse the evaluator-provided runtime limit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=float, required=True)
    return parser.parse_args()


def checkpoint_payload(
    model: nn.Module,
    mean: list[float],
    std: list[float],
    image_size: int,
    epoch: int,
    threshold: float | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a portable CPU checkpoint."""
    return {
        "model": {
            "base_channels": BASE_CHANNELS,
            "dropout": DROPOUT,
        },
        "state_dict": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        "normalization": {"mean": mean, "std": std},
        "image_size": image_size,
        "epoch": epoch,
        "threshold": threshold,
        "metrics": metrics,
    }


def build_dataset(
    prepared_dir: Path,
    split: str,
    mean: list[float],
    std: list[float],
    indices: np.ndarray | None = None,
) -> PreparedDataset:
    """Create one prepared dataset."""
    return PreparedDataset(prepared_dir, split, mean, std, indices)


def main() -> None:
    """Train the final plain CNN and calibrate its operating threshold."""
    args = parse_args()
    if args.timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    started = time.monotonic()
    reserve = min(90.0, args.timeout_seconds * 0.20)
    training_deadline = started + args.timeout_seconds - reserve
    set_reproducibility(SEED, THREADS)

    solution_dir = Path(__file__).resolve().parent
    prepared_dir = solution_dir / "artifacts" / "task02" / "prepared"
    output_dir = solution_dir / "artifacts" / "task02" / "model"
    model_path = output_dir / "model.pt"
    output_dir.mkdir(parents=True, exist_ok=True)

    mean, std, image_size = load_preparation_config(prepared_dir)
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
    model = PlainCNN(BASE_CHANNELS, DROPOUT)
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
        pct_start=0.30,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )

    print(f"model=plain_cnn parameters={sum(p.numel() for p in model.parameters())}")
    print(f"train_samples={train_indices.size} monitor_samples={monitor_indices.size}")
    print(f"class_weights={class_weights.tolist()}")

    history: list[dict[str, Any]] = []
    best_key: tuple[float, float, float] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    training_seconds = 0.0

    for epoch in range(1, EPOCHS + 1):
        if time.monotonic() >= training_deadline:
            print(f"epoch={epoch} status=skipped reason=deadline")
            break

        epoch_started = time.monotonic()
        train_loss, interrupted = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            training_deadline,
        )
        epoch_seconds = time.monotonic() - epoch_started
        training_seconds += epoch_seconds

        if interrupted:
            print(f"epoch={epoch} status=interrupted reason=deadline")
            break

        monitor_scores, monitor_labels, monitor_loss = score_loader(
            model,
            monitor_loader,
            criterion,
        )
        threshold = select_threshold(monitor_labels, monitor_scores, MONITOR_FPR)
        monitor_metrics = evaluate_scores(
            monitor_labels,
            monitor_scores,
            threshold["threshold"],
        )
        key = (
            monitor_metrics["recall_ai"],
            monitor_metrics["roc_auc"],
            -float(monitor_loss),
        )
        improved = best_key is None or key > best_key

        if improved:
            best_key = key
            best_epoch = epoch
            epochs_without_improvement = 0
            atomic_torch_save(
                model_path,
                checkpoint_payload(model, mean, std, image_size, epoch),
            )
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "monitor_loss": monitor_loss,
                "monitor_fpr": monitor_metrics["fpr_real"],
                "monitor_recall": monitor_metrics["recall_ai"],
                "monitor_precision": monitor_metrics["precision_ai"],
                "monitor_roc_auc": monitor_metrics["roc_auc"],
                "learning_rate_end": optimizer.param_groups[0]["lr"],
                "epoch_seconds": epoch_seconds,
                "improved": improved,
            }
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"monitor_loss={monitor_loss:.6f} "
            f"monitor_fpr={monitor_metrics['fpr_real']:.4f} "
            f"monitor_recall={monitor_metrics['recall_ai']:.4f} "
            f"seconds={epoch_seconds:.3f} improved={improved}"
        )

        if epoch >= MIN_EPOCHS and epochs_without_improvement >= PATIENCE:
            print(f"training status=early_stopped epoch={epoch}")
            break

    if not model_path.exists():
        raise TimeoutError("No complete epoch finished before the deadline")

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
        if not (prepared_dir / f"{split}_images.npy").exists():
            continue
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

    calibrated = select_threshold(
        labels_by_split["calibration"],
        scores["calibration"],
        CALIBRATION_FPR,
    )
    metrics = {
        split: evaluate_scores(labels_by_split[split], split_scores, calibrated["threshold"])
        for split, split_scores in scores.items()
    }
    validation_compliant = metrics["validation"]["fpr_real"] <= VALIDATION_FPR_LIMIT

    final_checkpoint = checkpoint_payload(
        model,
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
        "base_channels": BASE_CHANNELS,
        "dropout": DROPOUT,
        "max_learning_rate": MAX_LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "monitor_fraction": MONITOR_FRACTION,
        "monitor_fpr": MONITOR_FPR,
        "calibration_fpr": CALIBRATION_FPR,
        "validation_fpr_limit": VALIDATION_FPR_LIMIT,
        "seed": SEED,
    }
    result = {
        "model": "plain_cnn",
        "best_epoch": best_epoch,
        "training_seconds": training_seconds,
        "total_seconds": time.monotonic() - started,
        "threshold": calibrated,
        "metrics": metrics,
        "validation_compliant": validation_compliant,
        "configuration": configuration,
        "history": history,
    }
    write_json(output_dir / "metrics.json", result)

    validation = metrics["validation"]
    print(
        f"training status=complete best_epoch={best_epoch} "
        f"threshold={calibrated['threshold']:.6f} "
        f"validation_fpr={validation['fpr_real']:.4f} "
        f"validation_recall={validation['recall_ai']:.4f} "
        f"validation_precision={validation['precision_ai']:.4f} "
        f"validation_compliant={validation_compliant} "
        f"model_path={model_path}"
    )


if __name__ == "__main__":
    main()
