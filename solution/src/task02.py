"""Shared model, data, training, and evaluation utilities for Task 1.2."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset


class PlainCNN(nn.Module):
    """Sequential CNN with early spatial downsampling."""

    def __init__(self, base_channels: int = 16, dropout: float = 0.20) -> None:
        super().__init__()
        k = base_channels
        self.features = nn.Sequential(
            self._block(3, k, 2),
            self._block(k, k, 1),
            self._block(k, 2 * k, 2),
            self._block(2 * k, 2 * k, 1),
            self._block(2 * k, 4 * k, 2),
            self._block(4 * k, 4 * k, 1),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(4 * k, 2),
        )
        self._initialize()

    @staticmethod
    def _block(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        """Create one convolution, normalization, and activation block."""
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _initialize(self) -> None:
        """Initialize trainable parameters."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return logits for the real and AI-generated classes."""
        return self.classifier(self.features(inputs))


class PreparedDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Read normalized images lazily from prepared NumPy arrays."""

    def __init__(
        self,
        prepared_dir: Path,
        split: str,
        mean: Sequence[float],
        std: Sequence[float],
        indices: np.ndarray | None = None,
    ) -> None:
        self.image_path = prepared_dir / f"{split}_images.npy"
        self.label_path = prepared_dir / f"{split}_labels.npy"
        if not self.image_path.exists() or not self.label_path.exists():
            raise FileNotFoundError(f"Prepared split is incomplete: {split}")

        self.labels = load_labels(prepared_dir, split)
        self.indices = None if indices is None else np.asarray(indices, dtype=np.int64)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.images: np.ndarray | None = None

        image_array = np.load(self.image_path, mmap_mode="r")
        if image_array.shape[0] != self.labels.size or image_array.shape[1] != 3:
            raise ValueError(f"Invalid prepared image shape: {image_array.shape}")
        del image_array

    def __getstate__(self) -> dict[str, Any]:
        """Discard open memory maps before worker processes are spawned."""
        state = self.__dict__.copy()
        state["images"] = None
        return state

    def __len__(self) -> int:
        return int(self.labels.size if self.indices is None else self.indices.size)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(item if self.indices is None else self.indices[item])
        if self.images is None:
            self.images = np.load(self.image_path, mmap_mode="r")
        image = np.array(self.images[index], dtype=np.float32, copy=True) / 255.0
        tensor = (torch.from_numpy(image) - self.mean) / self.std
        return tensor, torch.tensor(int(self.labels[index]), dtype=torch.long)


def set_reproducibility(seed: int, threads: int) -> None:
    """Configure deterministic random seeds and CPU thread limits."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def load_preparation_config(prepared_dir: Path) -> tuple[list[float], list[float], int]:
    """Load channel statistics and image size from preparation metadata."""
    path = prepared_dir / "preparation_config.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    statistics = metadata["train_channel_statistics"]
    mean = np.asarray(statistics["mean"], dtype=np.float32)
    std = np.asarray(statistics["std"], dtype=np.float32)
    if mean.shape != (3,) or std.shape != (3,) or np.any(std <= 0):
        raise ValueError("Invalid train-channel statistics")
    return mean.tolist(), std.tolist(), int(metadata["image_size"])


def load_labels(prepared_dir: Path, split: str) -> np.ndarray:
    """Load binary labels for one prepared split."""
    labels = np.asarray(np.load(prepared_dir / f"{split}_labels.npy")).reshape(-1)
    labels = (labels > 0).astype(np.int64)
    if labels.size == 0:
        raise ValueError(f"Prepared split is empty: {split}")
    return labels


def split_train_monitor(
    labels: np.ndarray,
    monitor_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic stratified training and monitor split."""
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    monitor_parts: list[np.ndarray] = []
    for label in (0, 1):
        indices = rng.permutation(np.flatnonzero(labels == label))
        if indices.size < 2:
            raise ValueError(f"Class {label} needs at least two samples")
        count = min(max(round(indices.size * monitor_fraction), 1), indices.size - 1)
        monitor_parts.append(indices[:count])
        train_parts.append(indices[count:])
    train = rng.permutation(np.concatenate(train_parts)).astype(np.int64)
    monitor = np.sort(np.concatenate(monitor_parts)).astype(np.int64)
    return train, monitor


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int,
) -> DataLoader:
    """Create a reproducible CPU DataLoader."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=False,
        persistent_workers=workers > 0,
        generator=generator,
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    deadline: float,
) -> tuple[float, bool]:
    """Train one epoch and stop safely when the deadline is reached."""
    model.train()
    loss_sum = 0.0
    samples = 0
    for images, labels in loader:
        if time.monotonic() >= deadline:
            return loss_sum / max(samples, 1), True
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
def score_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module | None = None,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Return AI probabilities, labels, and optional mean loss."""
    model.eval()
    scores: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    loss_sum = 0.0
    samples = 0
    for images, labels in loader:
        logits = model(images)
        if criterion is not None:
            loss_sum += float(criterion(logits, labels).item()) * labels.shape[0]
            samples += labels.shape[0]
        scores.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        labels_all.append(labels.cpu().numpy())
    mean_loss = None if criterion is None else loss_sum / samples
    return np.concatenate(scores), np.concatenate(labels_all), mean_loss


def select_threshold(labels: np.ndarray, scores: np.ndarray, max_fpr: float) -> dict[str, Any]:
    """Select the recall-maximizing threshold under the requested FPR."""
    labels, scores = _validate_scores(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    valid = np.flatnonzero(fpr <= max_fpr + 1e-12)
    best_recall = np.max(tpr[valid])
    candidates = valid[np.isclose(tpr[valid], best_recall, atol=1e-12, rtol=0.0)]
    index = int(candidates[np.argmin(thresholds[candidates])])
    threshold = float(thresholds[index])
    if not np.isfinite(threshold):
        threshold = float(np.nextafter(np.max(scores), np.inf))
    counts = _confusion(labels, scores >= threshold)
    real = int(np.sum(labels == 0))
    ai = int(np.sum(labels == 1))
    return {
        "threshold": threshold,
        "target_max_fpr": max_fpr,
        "fpr_real": counts["false_positives"] / real,
        "recall_ai": counts["true_positives"] / ai,
        **counts,
    }


def evaluate_scores(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    """Evaluate probabilities at one fixed operating threshold."""
    labels, scores = _validate_scores(labels, scores)
    counts = _confusion(labels, scores >= threshold)
    real = int(np.sum(labels == 0))
    ai = int(np.sum(labels == 1))
    total = labels.size
    precision = _divide(counts["true_positives"], counts["true_positives"] + counts["false_positives"])
    recall = _divide(counts["true_positives"], ai)
    return {
        "threshold": threshold,
        "n_samples": int(total),
        "n_real": real,
        "n_ai": ai,
        "accuracy": _divide(counts["true_positives"] + counts["true_negatives"], total),
        "precision_ai": precision,
        "recall_ai": recall,
        "f1_ai": _divide(2.0 * precision * recall, precision + recall),
        "fpr_real": _divide(counts["false_positives"], real),
        "specificity_real": _divide(counts["true_negatives"], real),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "confusion_matrix": [
            [counts["true_negatives"], counts["false_positives"]],
            [counts["false_negatives"], counts["true_positives"]],
        ],
        **counts,
    }


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    """Write a PyTorch checkpoint atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically after converting NumPy values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_scores(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size == 0 or labels.size != scores.size:
        raise ValueError("Labels and scores must have equal non-zero length")
    if np.unique(labels).size != 2 or not np.all(np.isfinite(scores)):
        raise ValueError("Both classes and finite scores are required")
    return labels, scores


def _confusion(labels: np.ndarray, predictions: np.ndarray) -> dict[str, int]:
    predictions = np.asarray(predictions, dtype=bool)
    return {
        "true_negatives": int(np.sum((labels == 0) & ~predictions)),
        "false_positives": int(np.sum((labels == 0) & predictions)),
        "false_negatives": int(np.sum((labels == 1) & ~predictions)),
        "true_positives": int(np.sum((labels == 1) & predictions)),
    }


def _divide(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
