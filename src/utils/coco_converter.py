"""
=============================================================================
YOLO to COCO Format Converter
=============================================================================
Converts YOLO annotation format (.txt per image) to a single COCO JSON file
required for mAP_S evaluation with pycocotools.

YOLO FORMAT (per .txt file):
  [class_id] [x_center] [y_center] [width] [height]
  Coordinates are normalized to [0, 1] relative to image dimensions.

COCO FORMAT (single .json file):
  {
    "images": [...],
    "annotations": [...],
    "categories": [...]
  }
  Coordinates are absolute pixels in [x, y, width, height] format.

USAGE:
  python src/utils/coco_converter.py
  python src/utils/coco_converter.py --images dataset/val/images/ --labels dataset/val/labels/ --output dataset/val/annotations/instances_val.json
=============================================================================
"""

import argparse
import json
from pathlib import Path

import cv2
import yaml


def yolo_to_coco(
    images_dir: str,
    labels_dir: str,
    output_json: str,
    dataset_yaml: str = "config/dataset.yaml",
    class_names: dict = None,
):
    """Convert YOLO annotations to COCO JSON format.

    Reads all YOLO .txt label files, maps normalized coordinates to absolute
    pixel values using corresponding image dimensions, and writes a single
    COCO-format JSON file.

    Args:
        images_dir: Directory containing images.
        labels_dir: Directory containing YOLO .txt labels.
        output_json: Output path for the COCO JSON file.
        dataset_yaml: Dataset config YAML (for class names).
        class_names: Optional dict of {class_id: name}. Overrides YAML.

    Returns:
        Path to the generated COCO JSON file.
    """
    print("=" * 60)
    print("  YOLO → COCO FORMAT CONVERTER")
    print("=" * 60)

    # Load class names from dataset YAML if not provided
    if class_names is None:
        yaml_path = Path(dataset_yaml)
        if yaml_path.exists():
            with open(yaml_path, "r") as f:
                config = yaml.safe_load(f)
            class_names = config.get("names", {})
            if isinstance(class_names, list):
                class_names = {i: name for i, name in enumerate(class_names)}
            print(f"[INFO] Loaded {len(class_names)} classes from {dataset_yaml}")
        else:
            class_names = {}
            print(f"[WARNING] No dataset YAML found. Class names will be numeric.")

    img_dir = Path(images_dir)
    lbl_dir = Path(labels_dir)

    if not img_dir.exists():
        print(f"[ERROR] Images directory not found: {images_dir}")
        return None
    if not lbl_dir.exists():
        print(f"[ERROR] Labels directory not found: {labels_dir}")
        return None

    # Initialize COCO structure
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": [],
    }

    # Build categories list
    for class_id, name in sorted(class_names.items(), key=lambda x: int(x[0])):
        coco_output["categories"].append({
            "id": int(class_id),
            "name": name,
            "supercategory": "object",
        })

    # Process images and labels
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_files = sorted(
        [f for f in img_dir.iterdir() if f.suffix.lower() in image_extensions]
    )

    annotation_id = 1
    images_processed = 0
    annotations_total = 0
    skipped = 0

    print(f"\n[INFO] Processing {len(image_files)} images...")

    for image_id, img_file in enumerate(image_files, 1):
        # Read image dimensions
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"  [SKIP] Cannot read: {img_file.name}")
            skipped += 1
            continue

        img_h, img_w = img.shape[:2]

        # Add image entry
        coco_output["images"].append({
            "id": image_id,
            "file_name": img_file.name,
            "width": img_w,
            "height": img_h,
        })

        # Find matching label file
        label_file = lbl_dir / f"{img_file.stem}.txt"
        if not label_file.exists():
            images_processed += 1
            continue

        # Parse YOLO annotations
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                w_norm = float(parts[3])
                h_norm = float(parts[4])

                # Convert normalized YOLO → absolute COCO [x, y, width, height]
                # YOLO: center-based, normalized
                # COCO: top-left corner, absolute pixels
                abs_w = w_norm * img_w
                abs_h = h_norm * img_h
                abs_x = (x_center * img_w) - (abs_w / 2)
                abs_y = (y_center * img_h) - (abs_h / 2)

                # Clamp to image boundaries
                abs_x = max(0, abs_x)
                abs_y = max(0, abs_y)
                abs_w = min(abs_w, img_w - abs_x)
                abs_h = min(abs_h, img_h - abs_y)

                area = abs_w * abs_h

                coco_output["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [
                        round(abs_x, 2),
                        round(abs_y, 2),
                        round(abs_w, 2),
                        round(abs_h, 2),
                    ],
                    "area": round(area, 2),
                    "iscrowd": 0,
                })

                annotation_id += 1
                annotations_total += 1

        images_processed += 1

    # Ensure output directory exists
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write COCO JSON
    with open(output_path, "w") as f:
        json.dump(coco_output, f, indent=2)

    # Summary
    print(f"\n{'─' * 50}")
    print(f"  ✅ Conversion complete!")
    print(f"     Images processed  : {images_processed}")
    print(f"     Images skipped    : {skipped}")
    print(f"     Total annotations : {annotations_total}")
    print(f"     Categories        : {len(coco_output['categories'])}")
    print(f"     Output saved to   : {output_json}")
    print(f"{'─' * 50}")

    # Print size distribution
    if annotations_total > 0:
        small = sum(
            1 for a in coco_output["annotations"] if a["area"] < 32 ** 2
        )
        medium = sum(
            1 for a in coco_output["annotations"]
            if 32 ** 2 <= a["area"] <= 96 ** 2
        )
        large = sum(
            1 for a in coco_output["annotations"] if a["area"] > 96 ** 2
        )
        print(f"\n  📊 COCO Size Distribution:")
        print(f"     Small  (< 32²px)  : {small} ({small/annotations_total*100:.1f}%)")
        print(f"     Medium (32²-96²)  : {medium} ({medium/annotations_total*100:.1f}%)")
        print(f"     Large  (> 96²px)  : {large} ({large/annotations_total*100:.1f}%)")

    return str(output_path)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert YOLO annotations to COCO JSON format",
    )

    parser.add_argument(
        "--images", type=str, default="dataset/val/images/",
        help="Directory containing images"
    )
    parser.add_argument(
        "--labels", type=str, default="dataset/val/labels/",
        help="Directory containing YOLO .txt labels"
    )
    parser.add_argument(
        "--output", type=str,
        default="dataset/val/annotations/instances_val.json",
        help="Output COCO JSON path"
    )
    parser.add_argument(
        "--dataset-yaml", type=str, default="config/dataset.yaml",
        help="Dataset YAML for class names"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    yolo_to_coco(
        images_dir=args.images,
        labels_dir=args.labels,
        output_json=args.output,
        dataset_yaml=args.dataset_yaml,
    )
