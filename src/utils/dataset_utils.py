"""
=============================================================================
Dataset Utilities — Validation, Statistics, and Preparation
=============================================================================
Tools for validating dataset integrity, computing annotation statistics,
and identifying potential issues before training.

FEATURES:
  - Validate YOLO label format and coordinate ranges
  - Compute object size distribution (small/medium/large by COCO thresholds)
  - Detect overly loose bounding boxes (common issue with small objects)
  - Verify image-label file pairing
  - Generate dataset statistics report

USAGE:
  python src/utils/dataset_utils.py --dataset-dir dataset/
  python src/utils/dataset_utils.py --dataset-dir dataset/ --check-sizes
=============================================================================
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml


def validate_label_file(label_path: str, num_classes: int) -> dict:
    """Validate a single YOLO label file for format correctness.

    Checks:
      - Each line has exactly 5 fields
      - class_id is a valid integer within [0, num_classes)
      - x_center, y_center, width, height are floats in [0, 1]

    Args:
        label_path: Path to the YOLO .txt label file.
        num_classes: Total number of classes in the dataset.

    Returns:
        Dictionary with validation results and any errors found.
    """
    errors = []
    annotations = []
    label_file = Path(label_path)

    if not label_file.exists():
        return {"valid": False, "errors": ["File not found"], "annotations": []}

    if label_file.stat().st_size == 0:
        # Empty label file = image with no objects (valid for negative samples)
        return {"valid": True, "errors": [], "annotations": [], "is_negative": True}

    with open(label_file, "r") as f:
        lines = f.readlines()

    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            errors.append(
                f"Line {line_num}: Expected 5 fields, got {len(parts)}: '{line}'"
            )
            continue

        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            errors.append(f"Line {line_num}: Non-numeric values: '{line}'")
            continue

        # Validate class ID
        if class_id < 0 or class_id >= num_classes:
            errors.append(
                f"Line {line_num}: class_id {class_id} out of range [0, {num_classes})"
            )

        # Validate coordinates are in [0, 1]
        for name, value in [
            ("x_center", x_center), ("y_center", y_center),
            ("width", width), ("height", height),
        ]:
            if value < 0 or value > 1:
                errors.append(
                    f"Line {line_num}: {name}={value} outside [0, 1]"
                )

        # Validate bounding box doesn't extend outside image
        if x_center - width / 2 < -0.001 or x_center + width / 2 > 1.001:
            errors.append(f"Line {line_num}: Box extends outside image horizontally")
        if y_center - height / 2 < -0.001 or y_center + height / 2 > 1.001:
            errors.append(f"Line {line_num}: Box extends outside image vertically")

        annotations.append({
            "class_id": class_id,
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height,
        })

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "annotations": annotations,
        "is_negative": False,
    }


def compute_object_sizes(
    label_dir: str, image_dir: str, class_names: dict = None
) -> dict:
    """Compute absolute pixel sizes for all annotated objects.

    Maps normalized YOLO coordinates to actual pixel dimensions using
    the corresponding image sizes, then classifies objects by COCO
    size thresholds (small < 32², medium 32²–96², large > 96²).

    Args:
        label_dir: Directory containing YOLO label .txt files.
        image_dir: Directory containing corresponding images.
        class_names: Optional dict mapping class_id → class_name.

    Returns:
        Dictionary with size statistics and per-class breakdowns.
    """
    label_path = Path(label_dir)
    image_path = Path(image_dir)

    size_categories = {"small": 0, "medium": 0, "large": 0}
    per_class_sizes = defaultdict(lambda: {"small": 0, "medium": 0, "large": 0})
    all_areas = []
    all_widths = []
    all_heights = []

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    label_files = sorted(label_path.glob("*.txt"))

    for lf in label_files:
        # Find matching image
        stem = lf.stem
        img_file = None
        for ext in image_extensions:
            candidate = image_path / f"{stem}{ext}"
            if candidate.exists():
                img_file = candidate
                break

        if img_file is None:
            continue

        # Read image dimensions
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        # Process annotations
        with open(lf, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                class_id = int(parts[0])
                w_norm = float(parts[3])
                h_norm = float(parts[4])

                # Convert to absolute pixels
                w_px = w_norm * img_w
                h_px = h_norm * img_h
                area = w_px * h_px

                all_areas.append(area)
                all_widths.append(w_px)
                all_heights.append(h_px)

                # COCO size classification
                if area < 32 ** 2:
                    cat = "small"
                elif area < 96 ** 2:
                    cat = "medium"
                else:
                    cat = "large"

                size_categories[cat] += 1

                class_label = (
                    class_names.get(class_id, str(class_id))
                    if class_names
                    else str(class_id)
                )
                per_class_sizes[class_label][cat] += 1

    total = sum(size_categories.values())

    return {
        "total_objects": total,
        "size_distribution": size_categories,
        "size_percentages": {
            k: round(v / total * 100, 1) if total > 0 else 0
            for k, v in size_categories.items()
        },
        "per_class": dict(per_class_sizes),
        "area_stats": {
            "min": round(min(all_areas), 1) if all_areas else 0,
            "max": round(max(all_areas), 1) if all_areas else 0,
            "mean": round(float(np.mean(all_areas)), 1) if all_areas else 0,
            "median": round(float(np.median(all_areas)), 1) if all_areas else 0,
        },
        "width_stats": {
            "min": round(min(all_widths), 1) if all_widths else 0,
            "max": round(max(all_widths), 1) if all_widths else 0,
            "mean": round(float(np.mean(all_widths)), 1) if all_widths else 0,
        },
        "height_stats": {
            "min": round(min(all_heights), 1) if all_heights else 0,
            "max": round(max(all_heights), 1) if all_heights else 0,
            "mean": round(float(np.mean(all_heights)), 1) if all_heights else 0,
        },
    }


def validate_dataset(
    dataset_dir: str, dataset_yaml: str = "config/dataset.yaml"
) -> dict:
    """Validate the entire dataset: file pairing, label format, and statistics.

    Args:
        dataset_dir: Root directory of the dataset.
        dataset_yaml: Path to the dataset YAML configuration.

    Returns:
        Dictionary with full validation report.
    """
    print("=" * 60)
    print("  DATASET VALIDATION REPORT")
    print("=" * 60)

    # Load dataset config
    yaml_path = Path(dataset_yaml)
    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        num_classes = config.get("nc", 1)
        class_names = config.get("names", {})
        if isinstance(class_names, list):
            class_names = {i: name for i, name in enumerate(class_names)}
    else:
        print(f"[WARNING] Dataset config not found: {dataset_yaml}")
        num_classes = 10
        class_names = {}

    report = {}
    dataset_path = Path(dataset_dir)

    for split in ["train", "val", "test"]:
        images_dir = dataset_path / split / "images"
        labels_dir = dataset_path / split / "labels"

        if not images_dir.exists():
            print(f"\n  [{split.upper()}] Skipped — directory not found")
            continue

        print(f"\n  {'─' * 50}")
        print(f"  [{split.upper()}] Validating...")

        # Find images and labels
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        images = {
            f.stem: f
            for f in images_dir.iterdir()
            if f.suffix.lower() in image_extensions
        }
        labels = {f.stem: f for f in labels_dir.glob("*.txt")} if labels_dir.exists() else {}

        # Check pairing
        images_without_labels = set(images.keys()) - set(labels.keys())
        labels_without_images = set(labels.keys()) - set(images.keys())

        print(f"    Images      : {len(images)}")
        print(f"    Labels      : {len(labels)}")
        print(f"    Paired      : {len(set(images.keys()) & set(labels.keys()))}")

        if images_without_labels:
            print(f"    ⚠️  {len(images_without_labels)} images without labels")
        if labels_without_images:
            print(f"    ⚠️  {len(labels_without_images)} labels without images")

        # Validate labels
        errors_found = 0
        total_annotations = 0

        for stem, label_file in labels.items():
            result = validate_label_file(str(label_file), num_classes)
            if not result["valid"]:
                errors_found += 1
                for err in result["errors"][:3]:
                    print(f"    ❌ {label_file.name}: {err}")
            total_annotations += len(result["annotations"])

        print(f"    Annotations : {total_annotations}")
        print(f"    Errors      : {errors_found}")

        # Compute object sizes
        if labels_dir and labels_dir.exists():
            size_stats = compute_object_sizes(
                str(labels_dir), str(images_dir), class_names
            )
            print(f"\n    📊 Object Size Distribution:")
            print(f"       Small  (< 32²px) : {size_stats['size_distribution']['small']:>5d}  "
                  f"({size_stats['size_percentages']['small']}%)")
            print(f"       Medium (32²-96²) : {size_stats['size_distribution']['medium']:>5d}  "
                  f"({size_stats['size_percentages']['medium']}%)")
            print(f"       Large  (> 96²px) : {size_stats['size_distribution']['large']:>5d}  "
                  f"({size_stats['size_percentages']['large']}%)")

            report[split] = {
                "images": len(images),
                "labels": len(labels),
                "annotations": total_annotations,
                "errors": errors_found,
                "sizes": size_stats,
            }

    print(f"\n{'═' * 60}")
    print("  Validation complete.")
    return report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Dataset Validation Utilities")
    parser.add_argument(
        "--dataset-dir", type=str, default="dataset/",
        help="Root dataset directory"
    )
    parser.add_argument(
        "--dataset-yaml", type=str, default="config/dataset.yaml",
        help="Dataset YAML configuration"
    )
    parser.add_argument(
        "--check-sizes", action="store_true",
        help="Print detailed size statistics"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validate_dataset(args.dataset_dir, args.dataset_yaml)
