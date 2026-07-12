"""Report artifact writers for CSV, JSON, and figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


LABEL_NAMES = {0: "real", 1: "ai_generated"}


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


def label_name(label: object) -> str:
    """Return a display name for a binary label."""

    try:
        return LABEL_NAMES[int(label)]
    except (TypeError, ValueError, KeyError):
        return str(label)


def save_class_distribution_plot(dataframe: pd.DataFrame, path: Path) -> None:
    """Save a class-count bar plot grouped by split."""

    ensure_dir(path.parent)
    plt.figure(figsize=(9, 5))
    counts = dataframe.groupby(["split", "binary_label"]).size().unstack(fill_value=0)
    counts = counts.rename(columns=LABEL_NAMES)
    counts.plot(kind="bar", ax=plt.gca())
    plt.xlabel("split")
    plt.ylabel("count")
    plt.title("Binary class distribution by split")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_class_percentage_plot(dataframe: pd.DataFrame, path: Path) -> None:
    """Save a normalized binary-class bar plot grouped by split."""

    ensure_dir(path.parent)
    plt.figure(figsize=(9, 5))
    counts = dataframe.groupby(["split", "binary_label"]).size().unstack(fill_value=0)
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100.0
    percentages = percentages.rename(columns=LABEL_NAMES)
    percentages.plot(kind="bar", ax=plt.gca())
    plt.xlabel("split")
    plt.ylabel("percentage")
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


def save_source_class_percentage_plot(dataframe: pd.DataFrame, path: Path) -> None:
    """Save a normalized source-class bar plot grouped by split."""

    ensure_dir(path.parent)
    plt.figure(figsize=(10, 5))
    counts = dataframe.groupby(["split", "source_class"]).size().unstack(fill_value=0)
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100.0
    percentages.plot(kind="bar", ax=plt.gca())
    plt.xlabel("split")
    plt.ylabel("percentage")
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


def save_image_size_scatter_by_label(
    dataframe: pd.DataFrame,
    path: Path,
    max_points: int = 10000,
) -> None:
    """Save a deterministic image-size scatter plot grouped by binary label."""

    ensure_dir(path.parent)
    plot_df = dataframe[dataframe["decode_ok"]].copy()
    if len(plot_df) > max_points:
        plot_df = plot_df.sample(n=max_points, random_state=0)

    plt.figure(figsize=(7, 6))
    for binary_label, group in plot_df.groupby("binary_label"):
        plt.scatter(
            group["width"],
            group["height"],
            s=8,
            alpha=0.35,
            label=label_name(binary_label),
        )
    plt.xlabel("width")
    plt.ylabel("height")
    plt.title("Image size distribution by binary label")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_encoded_size_boxplot(dataframe: pd.DataFrame, path: Path) -> None:
    """Save an encoded-byte-size boxplot grouped by binary label."""

    ensure_dir(path.parent)
    plot_df = dataframe[dataframe["decode_ok"]].copy()
    plot_df["label_name"] = plot_df["binary_label"].map(LABEL_NAMES)

    plt.figure(figsize=(7, 5))
    plot_df.boxplot(column="image_bytes", by="label_name", ax=plt.gca())
    plt.suptitle("")
    plt.xlabel("binary label")
    plt.ylabel("encoded bytes")
    plt.title("Encoded image size by label")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_encoded_size_boxplot_log(dataframe: pd.DataFrame, path: Path) -> None:
    """Save a log-scaled encoded-byte-size boxplot grouped by binary label."""

    ensure_dir(path.parent)
    plot_df = dataframe[(dataframe["decode_ok"]) & (dataframe["image_bytes"] > 0)].copy()
    plot_df["label_name"] = plot_df["binary_label"].map(LABEL_NAMES)

    plt.figure(figsize=(7, 5))
    plot_df.boxplot(column="image_bytes", by="label_name", ax=plt.gca())
    plt.yscale("log")
    plt.suptitle("")
    plt.xlabel("binary label")
    plt.ylabel("encoded bytes, log scale")
    plt.title("Encoded image size by label")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_aspect_ratio_histogram(dataframe: pd.DataFrame, path: Path) -> None:
    """Save an aspect-ratio histogram grouped by binary label."""

    ensure_dir(path.parent)
    plot_df = dataframe[
        (dataframe["decode_ok"])
        & (dataframe["aspect_ratio"].notna())
        & (dataframe["aspect_ratio"] > 0)
    ].copy()

    plt.figure(figsize=(8, 5))
    for binary_label, group in plot_df.groupby("binary_label"):
        plt.hist(group["aspect_ratio"], bins=40, alpha=0.55, label=label_name(binary_label))
    plt.xlabel("aspect ratio, width / height")
    plt.ylabel("count")
    plt.title("Aspect-ratio distribution by binary label")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_megapixels_boxplot(dataframe: pd.DataFrame, path: Path) -> None:
    """Save a megapixel boxplot grouped by binary label."""

    ensure_dir(path.parent)
    plot_df = dataframe[(dataframe["decode_ok"]) & (dataframe["megapixels"].notna())].copy()
    plot_df["label_name"] = plot_df["binary_label"].map(LABEL_NAMES)

    plt.figure(figsize=(7, 5))
    plot_df.boxplot(column="megapixels", by="label_name", ax=plt.gca())
    plt.suptitle("")
    plt.xlabel("binary label")
    plt.ylabel("megapixels")
    plt.title("Image resolution by label")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_categorical_percentage_plot(
    dataframe: pd.DataFrame,
    column: str,
    path: Path,
    title: str,
) -> None:
    """Save a categorical distribution plot normalized within each binary label."""

    ensure_dir(path.parent)
    plot_df = dataframe[dataframe["decode_ok"]].copy()
    plot_df[column] = plot_df[column].fillna("unknown").astype(str)

    counts = plot_df.groupby(["binary_label", column]).size().unstack(fill_value=0)
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100.0
    percentages.index = [label_name(index) for index in percentages.index]

    plt.figure(figsize=(8, 5))
    percentages.T.plot(kind="bar", ax=plt.gca())
    plt.xlabel(column)
    plt.ylabel("percentage within binary label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
