"""
=============================================================================
Standard YOLO Inference — Baseline Detection (Non-SAHI)
=============================================================================
Runs standard YOLO inference on individual images or directories.
Use this for quick testing on standard-resolution images where slicing
is not required. For high-resolution images (4K, 8K), use sahi_inference.py.

USAGE:
  python src/inference.py --image test_images/sample.jpg
  python src/inference.py --source test_images/ --save
  python src/inference.py --image test_images/sample.jpg --conf 0.4 --imgsz 1280
=============================================================================
"""

import argparse
import sys
from pathlib import Path

import torch
from ultralytics import YOLO


def run_inference(
    source: str,
    model_path: str = "detection_models/small_object_p2_1280/weights/best.pt",
    imgsz: int = 1280,
    conf: float = 0.35,
    iou: float = 0.45,
    device: str = None,
    save: bool = True,
    save_txt: bool = False,
    show: bool = False,
    project: str = "output",
    name: str = "inference",
):
    """Run standard YOLO inference on an image or directory.

    Args:
        source: Path to image, directory, or video file.
        model_path: Path to trained YOLO model weights (.pt file).
        imgsz: Inference image size (should match training resolution).
        conf: Confidence threshold for detections.
        iou: IoU threshold for Non-Maximum Suppression.
        device: Compute device. Auto-detected if None.
        save: Save annotated images with bounding boxes.
        save_txt: Save detection results as .txt files.
        show: Display results in a window (requires GUI).
        project: Output directory for results.
        name: Run name for this inference session.
    """
    print("=" * 70)
    print("  STANDARD YOLO INFERENCE — SMALL OBJECT DETECTOR")
    print("=" * 70)

    # Validate model weights
    weights_path = Path(model_path)
    if not weights_path.exists():
        print(f"[ERROR] Model weights not found: {model_path}")
        print("        Train the model first using: python src/train.py")
        sys.exit(1)

    # Validate source
    source_path = Path(source)
    if not source_path.exists():
        print(f"[ERROR] Source not found: {source}")
        sys.exit(1)

    # Detect device
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    print(f"\n  Model    : {model_path}")
    print(f"  Source   : {source}")
    print(f"  Image Sz : {imgsz}px")
    print(f"  Conf     : {conf}")
    print(f"  IoU      : {iou}")
    print(f"  Device   : {device}")
    print()

    # Load model
    print("[STEP 1/3] Loading trained model...")
    model = YOLO(model_path)

    # Run inference
    print("[STEP 2/3] Running inference...")
    results = model.predict(
        source=source,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        save=save,
        save_txt=save_txt,
        show=show,
        project=project,
        name=name,
        exist_ok=True,
    )

    # Process results
    print("\n[STEP 3/3] Processing results...")
    print(f"\n{'─' * 60}")
    print(f"{'─' * 60}")

    total_detections = 0
    for idx, result in enumerate(results):
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        image_name = Path(result.path).name if result.path else f"image_{idx}"
        print(f"\n  📷 {image_name}:")

        for box in boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            width = x2 - x1
            height = y2 - y1
            area = width * height

            # Classify object size using COCO thresholds
            if area < 32 ** 2:
                size_label = "SMALL"
            elif area < 96 ** 2:
                size_label = "MEDIUM"
            else:
                size_label = "LARGE"

            print(
                f"    [{size_label:6s}] {class_name.upper():20s} "
                f"| Conf: {confidence:.3f} "
                f"| Box: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}] "
                f"| Size: {width:.0f}×{height:.0f}px"
            )
            total_detections += 1

    print(f"\n{'─' * 60}")
    print(f"  Total detections: {total_detections}")
    if save:
        print(f"  Results saved to: {project}/{name}/")
    print(f"{'─' * 60}")

    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Standard YOLO Inference for Small Object Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--image", "--source", type=str, required=True, dest="source",
        help="Path to image, directory, or video"
    )
    parser.add_argument(
        "--model", type=str,
        default="detection_models/small_object_p2_1280/weights/best.pt",
        help="Path to trained model weights"
    )
    parser.add_argument("--imgsz", type=int, default=1280, help="Image size")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default=None, help="Device")
    parser.add_argument("--save", action="store_true", default=True, help="Save results")
    parser.add_argument("--save-txt", action="store_true", help="Save labels as .txt")
    parser.add_argument("--show", action="store_true", help="Display results")
    parser.add_argument("--project", type=str, default="output", help="Output dir")
    parser.add_argument("--name", type=str, default="inference", help="Run name")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        source=args.source,
        model_path=args.model,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=args.save,
        save_txt=args.save_txt,
        show=args.show,
        project=args.project,
        name=args.name,
    )
