"""Explain the final Task 1.3 model with signed CAM and occlusion analysis."""

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

from src.explainability import (
    choose_occlusion_records_by_category,
    load_source_classes,
    map_statistics,
    occlusion_sensitivity,
    pearson_correlation,
    records_to_frame,
    resize_map,
    save_cam_figure,
    save_individual_cam_figure,
    save_occlusion_figure,
    select_examples,
    signed_class_activation_map,
    source_class_metrics,
    write_frame_atomic,
)
from src.task02 import (
    PlainCNN,
    PreparedDataset,
    evaluate_scores,
    load_labels,
    make_loader,
    score_loader,
    set_reproducibility,
    write_json,
)


SPLITS = ("validation", "validation_augmented")
BATCH_SIZE = 256
WORKERS = 0
THREADS = 8
SEED = 42


def parse_args() -> argparse.Namespace:
    """Parse explainability settings and the runtime limit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=float, required=True)
    parser.add_argument("--examples-per-category", type=int, default=2)
    parser.add_argument("--occlusion-patch-size", type=int, default=24)
    parser.add_argument("--occlusion-stride", type=int, default=12)
    parser.add_argument("--occlusion-batch-size", type=int, default=128)
    return parser.parse_args()


def normalized_tensor(
    image: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> torch.Tensor:
    """Convert one prepared uint8 image to the model's normalized tensor."""
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy((array - mean) / std)


def main() -> None:
    """Create report-ready explanations and quantitative failure analyses."""
    args = parse_args()
    if args.timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if args.examples_per_category <= 0:
        raise ValueError("examples-per-category must be positive")
    started = time.monotonic()
    deadline = started + args.timeout_seconds - min(10.0, args.timeout_seconds * 0.10)
    set_reproducibility(SEED, THREADS)

    solution_dir = Path(__file__).resolve().parent
    prepared_dir = solution_dir / "artifacts" / "task02" / "prepared"
    model_path = solution_dir / "artifacts" / "task03" / "model" / "model.pt"
    output_dir = solution_dir / "artifacts" / "task04"
    example_dir = output_dir / "examples"
    output_dir.mkdir(parents=True, exist_ok=True)
    example_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in example_dir.glob("*.png"):
        stale_path.unlink()

    if not model_path.exists():
        raise FileNotFoundError(
            "Task 1.3 checkpoint not found. Run train_augmented.py before explain.py."
        )

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if checkpoint.get("threshold") is None:
        raise ValueError("The Task 1.3 checkpoint has no calibrated threshold")

    model = PlainCNN(**checkpoint["model"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    threshold = float(checkpoint["threshold"])
    mean_list = checkpoint["normalization"]["mean"]
    std_list = checkpoint["normalization"]["std"]
    mean = np.asarray(mean_list, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(std_list, dtype=np.float32).reshape(3, 1, 1)

    all_records: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    split_metrics: dict[str, Any] = {}
    split_counts: dict[str, Any] = {}

    for split in SPLITS:
        if time.monotonic() >= deadline:
            raise TimeoutError("Explainability deadline reached before split scoring")

        dataset = PreparedDataset(prepared_dir, split, mean_list, std_list)
        loader = make_loader(
            dataset,
            BATCH_SIZE,
            False,
            WORKERS,
            SEED,
        )
        scores, labels, _ = score_loader(model, loader)
        source_classes = load_source_classes(prepared_dir, split)
        if source_classes.size != labels.size:
            raise ValueError(f"Source-class length mismatch for split {split}")

        split_metrics[split] = evaluate_scores(labels, scores, threshold)
        categories = np.asarray(
            [
                "true_positive" if label == 1 and score >= threshold
                else "true_negative" if label == 0 and score < threshold
                else "false_positive" if label == 0
                else "false_negative"
                for label, score in zip(labels.tolist(), scores.tolist())
            ],
            dtype=object,
        )
        split_counts[split] = {
            category: int(np.sum(categories == category))
            for category in (
                "true_positive",
                "true_negative",
                "false_positive",
                "false_negative",
            )
        }

        records = select_examples(
            split=split,
            labels=labels,
            source_classes=source_classes,
            scores=scores,
            threshold=threshold,
            examples_per_category=args.examples_per_category,
        )
        source_rows.extend(
            source_class_metrics(
                split=split,
                labels=labels,
                source_classes=source_classes,
                scores=scores,
                threshold=threshold,
            )
        )

        image_array = np.load(prepared_dir / f"{split}_images.npy", mmap_mode="r")
        for record in records:
            if time.monotonic() >= deadline:
                raise TimeoutError("Explainability deadline reached during CAM generation")

            sample_index = int(record["sample_index"])
            prepared_image = np.array(image_array[sample_index], copy=True)
            tensor = normalized_tensor(prepared_image, mean, std)
            explanation = signed_class_activation_map(model, tensor)
            height, width = prepared_image.shape[1:]
            cam_resized = resize_map(explanation["map"], height, width)

            record.update(
                {
                    "display_image": np.transpose(prepared_image, (1, 2, 0)),
                    "cam_map": explanation["map"],
                    "cam_resized": cam_resized,
                    "cam_ai_probability": explanation["ai_probability"],
                    "cam_logit_difference": explanation["logit_difference"],
                    "cam_reconstructed_logit_difference": explanation[
                        "reconstructed_logit_difference"
                    ],
                    "cam_reconstruction_error": explanation["reconstruction_error"],
                    **map_statistics(explanation["map"]),
                }
            )
            all_records.append(record)

        del image_array

        split_records = [record for record in all_records if record["split"] == split]
        save_cam_figure(
            split_records,
            output_dir / f"signed_cam_{split}.png",
            title=f"Signed class activation maps on {split}",
        )
        for record in split_records:
            filename = (
                f"{record['split']}_{record['category']}_"
                f"{record['selection_type']}_{int(record['sample_index']):05d}.png"
            )
            save_individual_cam_figure(record, example_dir / filename)

        print(
            f"explainability split={split} samples={labels.size} "
            f"selected_examples={len(split_records)}"
        )

    occlusion_records = choose_occlusion_records_by_category(all_records)
    for record in occlusion_records:
        if time.monotonic() >= deadline:
            raise TimeoutError("Explainability deadline reached before occlusion analysis")

        split = str(record["split"])
        sample_index = int(record["sample_index"])
        image_array = np.load(prepared_dir / f"{split}_images.npy", mmap_mode="r")
        prepared_image = np.array(image_array[sample_index], copy=True)
        del image_array
        tensor = normalized_tensor(prepared_image, mean, std)
        result = occlusion_sensitivity(
            model=model,
            normalized_image=tensor,
            patch_size=args.occlusion_patch_size,
            stride=args.occlusion_stride,
            batch_size=args.occlusion_batch_size,
            deadline=deadline,
        )
        correlation = pearson_correlation(record["cam_resized"], result["map"])
        record.update(
            {
                "occlusion_map": result["map"],
                "occlusion_base_ai_probability": result["base_ai_probability"],
                "occlusion_patch_count": result["patch_count"],
                "occlusion_patch_size": result["patch_size"],
                "occlusion_stride": result["stride"],
                "occlusion_replacement": result["replacement"],
                "cam_occlusion_correlation": correlation,
            }
        )

    save_occlusion_figure(
        occlusion_records,
        output_dir / "occlusion_comparison.png",
        title="Signed CAM compared with occlusion sensitivity",
    )

    selected_frame = records_to_frame(all_records)
    source_frame = records_to_frame(source_rows)
    write_frame_atomic(selected_frame, output_dir / "selected_examples.csv")
    write_frame_atomic(source_frame, output_dir / "source_class_metrics.csv")

    correlations = [
        float(record["cam_occlusion_correlation"])
        for record in occlusion_records
        if record.get("cam_occlusion_correlation") is not None
    ]
    reconstruction_errors = [
        float(record["cam_reconstruction_error"])
        for record in all_records
    ]
    summary = {
        "model_path": str(model_path),
        "model": "CNN",
        "threshold": threshold,
        "splits": list(SPLITS),
        "split_metrics": split_metrics,
        "split_category_counts": split_counts,
        "selected_examples": len(all_records),
        "occlusion_examples": len(occlusion_records),
        "occlusion_selection": (
            "one confident validation example for each prediction category: "
            "true positive, true negative, false positive, and false negative"
        ),
        "occlusion_configuration": {
            "patch_size": args.occlusion_patch_size,
            "stride": args.occlusion_stride,
            "batch_size": args.occlusion_batch_size,
            "replacement": "zero in normalized space (training-channel mean)",
        },
        "cam_definition": (
            "sum_k (classifier_weight_ai[k] - classifier_weight_real[k]) "
            "times final_feature_map[k]"
        ),
        "cam_interpretation": {
            "positive": "supports the AI logit relative to the real logit",
            "negative": "supports the real logit relative to the AI logit",
        },
        "maximum_cam_reconstruction_error": max(reconstruction_errors, default=None),
        "mean_cam_occlusion_correlation": (
            None if not correlations else float(np.mean(correlations))
        ),
        "limitations": [
            "The native CAM resolution is lower than the 128 x 128 model input.",
            "Upsampling makes the visualization appear more spatially precise than it is.",
            "CAM describes the final convolutional representation, not every internal operation.",
            "Occlusion changes the input distribution and therefore only provides a plausibility check.",
            "Selected examples are deterministic but do not replace aggregate evaluation.",
        ],
        "total_seconds": time.monotonic() - started,
    }
    write_json(output_dir / "explanation_summary.json", summary)

    print(
        f"explainability status=complete selected_examples={len(all_records)} "
        f"occlusion_examples={len(occlusion_records)} "
        f"elapsed_seconds={time.monotonic() - started:.3f} "
        f"output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()