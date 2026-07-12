"""Report artifact writers for CSV, JSON, and figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not exist and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(data: dict[str, Any], path: Path) -> None:
    """Write formatted JSON to disk."""

    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(dataframe: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame as CSV."""

    ensure_dir(path.parent)
    dataframe.to_csv(path, index=False)


def write_parquet(dataframe: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame as Parquet."""

    ensure_dir(path.parent)
    dataframe.to_parquet(path, index=False)


def save_class_distribution_plot(dataframe: pd.DataFrame, path: Path) -> None:
    """Save a class-count bar plot grouped by split."""

    ensure_dir(path.parent)
    plt.figure(figsize=(9, 5))
    counts = dataframe.groupby(["split", "binary_label"]).size().unstack(fill_value=0)
    counts = counts.rename(columns={0: "real", 1: "ai_generated"})
    counts.plot(kind="bar", ax=plt.gca())
    plt.xlabel("split")
    plt.ylabel("count")
    plt.title("Binary class distribution by split")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_source_class_distribution_plot(dataframe: pd.DataFrame, path: Path) -> None:
    """Save a source-class-count bar plot grouped by split."""

    ensure_dir(path.parent)
    plt.figure(figsize=(10, 5))
    counts = dataframe.groupby(["split", "source_class"]).size().unstack(fill_value=0)
    counts.plot(kind="bar", ax=plt.gca())
    plt.xlabel("split")
    plt.ylabel("count")
    plt.title("Source-class distribution by split")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_image_size_scatter(dataframe: pd.DataFrame, path: Path, max_points: int = 10000) -> None:
    """Save a deterministic image-size scatter plot."""

    ensure_dir(path.parent)
    plot_df = dataframe[dataframe["decode_ok"]].copy()
    if len(plot_df) > max_points:
        plot_df = plot_df.sample(n=max_points, random_state=0)

    plt.figure(figsize=(7, 6))
    plt.scatter(plot_df["width"], plot_df["height"], s=8, alpha=0.35)
    plt.xlabel("width")
    plt.ylabel("height")
    plt.title("Image size distribution")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_encoded_size_boxplot(dataframe: pd.DataFrame, path: Path) -> None:
    """Save an encoded-byte-size boxplot grouped by binary label."""

    ensure_dir(path.parent)
    plot_df = dataframe[dataframe["decode_ok"]].copy()
    plot_df["label_name"] = plot_df["binary_label"].map({0: "real", 1: "ai_generated"})

    plt.figure(figsize=(7, 5))
    plot_df.boxplot(column="image_bytes", by="label_name", ax=plt.gca())
    plt.suptitle("")
    plt.xlabel("binary label")
    plt.ylabel("encoded bytes")
    plt.title("Encoded image size by label")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
