"""
=============================================================================
SAHI Inference — Slicing Aided Hyper Inference
=============================================================================
Processes high-resolution images (4K, 8K, satellite, drone imagery) by
slicing them into overlapping patches and running YOLO detection on each.

WHY SAHI IS ESSENTIAL:
  Feeding a 4K image directly into YOLO forces aggressive downscaling that
  destroys small objects. SAHI solves this by:
  1. Slicing the image into 512×512 overlapping patches
  2. Running detection on each patch at native resolution (no downscaling)
  3. Mapping detections back to original image coordinates
  4. Applying NMS to merge duplicates from overlapping regions

OVERLAP MECHANISM:
  Without overlap, objects on slice boundaries get cut in half → missed.
  20% overlap ensures every boundary object appears whole in at least
  one patch.

USAGE:
  python src/sahi_inference.py --image test_images/drone_shot_4k.jpg
  python src/sahi_inference.py --image path/to/image.jpg --slice 640 --overlap 0.3
  python src/sahi_inference.py --config config/sahi_config.yaml --image path/to/image.jpg
=============================================================================
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def load_sahi_config(config_path: str) -> dict:
    """Load SAHI configuration from a YAML file.

    Args:
        config_path: Path to the SAHI config YAML.

    Returns:
        Dictionary of SAHI parameters.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"[WARNING] SAHI config not found: {config_path}")
        return {}

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    print(f"[INFO] Loaded SAHI configuration from: {config_path}")
    return config


def run_sahi_inference(
    image_path: str,
    model_weights: str = "detection_models/small_object_p2_1280/weights/best.pt",
    model_type: str = "yolov8",
    confidence_threshold: float = 0.35,
    device: str = "cuda:0",
    slice_height: int = 512,
    slice_width: int = 512,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    perform_standard_prediction: bool = True,
    export_dir: str = "output",
    export_visual: bool = True,
    export_crop: bool = False,
    save_json: bool = True,
    config_file: str = None,
):
    """Run SAHI sliced inference on a high-resolution image.

    The image is divided into overlapping patches, each processed independently
    by the YOLO model. Detections from all patches are merged using NMS.

    Args:
        image_path: Path to the input image.
        model_weights: Path to trained YOLO model weights.
        model_type: Detection model type ("yolov8").
        confidence_threshold: Minimum detection confidence.
        device: Compute device ("cuda:0" or "cpu").
        slice_height: Height of each image patch in pixels.
        slice_width: Width of each image patch in pixels.
        overlap_height_ratio: Vertical overlap ratio between patches (0-1).
        overlap_width_ratio: Horizontal overlap ratio between patches (0-1).
        perform_standard_prediction: Also run full-image inference for large objects.
        export_dir: Output directory for visualizations.
        export_visual: Save annotated image with bounding boxes.
        export_crop: Save cropped detection regions.
        save_json: Save detection results as JSON.
        config_file: Optional SAHI config YAML to override defaults.

    Returns:
        SAHI PredictionResult object containing all detections.
    """
    print("=" * 70)
    print("  SAHI — SLICING AIDED HYPER INFERENCE")
    print("=" * 70)

    # Load config overrides
    if config_file:
        config = load_sahi_config(config_file)
        model_weights = config.get("model_path", model_weights)
        model_type = config.get("model_type", model_type)
        confidence_threshold = config.get("confidence_threshold", confidence_threshold)
        device = config.get("device", device)
        slice_height = config.get("slice_height", slice_height)
        slice_width = config.get("slice_width", slice_width)
        overlap_height_ratio = config.get("overlap_height_ratio", overlap_height_ratio)
        overlap_width_ratio = config.get("overlap_width_ratio", overlap_width_ratio)
        perform_standard_prediction = config.get(
            "perform_standard_prediction", perform_standard_prediction
        )
        export_dir = config.get("export_dir", export_dir)
        export_visual = config.get("export_visual", export_visual)
        export_crop = config.get("export_crop", export_crop)

    # Validate inputs
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    weights_path = Path(model_weights)
    if not weights_path.exists():
        print(f"[ERROR] Model weights not found: {model_weights}")
        print("        Train the model first: python src/train.py")
        sys.exit(1)

    # Print configuration
    print(f"\n  Image              : {image_path}")
    print(f"  Model              : {model_weights}")
    print(f"  Device             : {device}")
    print(f"  Confidence         : {confidence_threshold}")
    print(f"  Slice Size         : {slice_width}×{slice_height}px")
    print(f"  Overlap Ratio      : H={overlap_height_ratio}, W={overlap_width_ratio}")
    print(f"  Full-Image Predict : {perform_standard_prediction}")
    print()

    # ─── Step 1: Load Model into SAHI Wrapper ────────────────────────────
    print("[STEP 1/4] Loading model into SAHI wrapper...")

    detection_model = AutoDetectionModel.from_pretrained(
        model_type=model_type,
        model_path=str(model_weights),
        confidence_threshold=confidence_threshold,
        device=device,
    )

    print(f"[INFO] Model loaded successfully ({model_type})")

    # ─── Step 2: Perform Sliced Inference ────────────────────────────────
    print("\n[STEP 2/4] Performing sliced inference...")

    # Calculate expected number of slices for user feedback
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            img_w, img_h = img.size
        
        effective_slice_h = slice_height * (1 - overlap_height_ratio)
        effective_slice_w = slice_width * (1 - overlap_width_ratio)
        n_slices_h = max(1, int((img_h - slice_height) / effective_slice_h) + 1)
        n_slices_w = max(1, int((img_w - slice_width) / effective_slice_w) + 1)
        total_slices = n_slices_h * n_slices_w
        
        print(f"[INFO] Image size: {img_w}×{img_h}px")
        print(f"[INFO] Generating ~{total_slices} overlapping slices "
              f"({n_slices_w} cols × {n_slices_h} rows)")
    except Exception:
        print("[INFO] Processing slices...")

    result = get_sliced_prediction(
        image=image_path,
        detection_model=detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        perform_standard_pred=perform_standard_prediction,
        verbose=1,
    )

    # ─── Step 3: Export Visual Results ───────────────────────────────────
    print("\n[STEP 3/4] Processing detections...")

    output_path = Path(export_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if export_visual:
        visual_filename = f"sahi_{img_path.stem}"
        result.export_visuals(export_dir=str(output_path), file_name=visual_filename)
        print(f"[INFO] Annotated image saved to: {output_path / visual_filename}.png")

    # ─── Step 4: Extract and Display Results ─────────────────────────────
    print("\n[STEP 4/4] Detection results:")
    predictions = result.object_prediction_list

    if not predictions:
        print("\n  ⚠️  No objects detected.")
        print("  Tips:")
        print("    - Lower the confidence threshold (--conf 0.2)")
        print("    - Increase slice overlap (--overlap 0.3)")
        print("    - Verify the model was trained on similar data")
        return result

    # Collect results for JSON export
    detection_results = []

    print(f"\n{'─' * 70}")
    print(f"  {'#':>3s}  {'CLASS':20s}  {'CONF':>6s}  {'BBOX (xyxy)':30s}  {'SIZE':>10s}")
    print(f"{'─' * 70}")

    for idx, obj in enumerate(predictions):
        class_name = obj.category.name
        confidence = obj.score.value
        bbox = obj.bbox.to_xyxy()  # [xmin, ymin, xmax, ymax]

        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height

        # COCO size classification
        if area < 32 ** 2:
            size_label = "SMALL"
        elif area < 96 ** 2:
            size_label = "MEDIUM"
        else:
            size_label = "LARGE"

        print(
            f"  {idx + 1:>3d}  {class_name.upper():20s}  {confidence:>6.3f}  "
            f"[{bbox[0]:>6.0f}, {bbox[1]:>6.0f}, {bbox[2]:>6.0f}, {bbox[3]:>6.0f}]"
            f"       {size_label:>6s}"
        )

        detection_results.append({
            "id": idx + 1,
            "class": class_name,
            "confidence": round(confidence, 4),
            "bbox_xyxy": [round(b, 1) for b in bbox],
            "width": round(width, 1),
            "height": round(height, 1),
            "area": round(area, 1),
            "size_category": size_label,
        })

    # Summary statistics
    small_count = sum(1 for d in detection_results if d["size_category"] == "SMALL")
    medium_count = sum(1 for d in detection_results if d["size_category"] == "MEDIUM")
    large_count = sum(1 for d in detection_results if d["size_category"] == "LARGE")

    print(f"{'─' * 70}")
    print(f"\n  📊 Detection Summary:")
    print(f"     Total : {len(predictions)}")
    print(f"     Small : {small_count} (< 32²px)")
    print(f"     Medium: {medium_count} (32²–96²px)")
    print(f"     Large : {large_count} (> 96²px)")

    # Save JSON results
    if save_json:
        json_output = {
            "image": str(image_path),
            "model": str(model_weights),
            "timestamp": datetime.now().isoformat(),
            "config": {
                "slice_size": f"{slice_width}x{slice_height}",
                "overlap": f"H={overlap_height_ratio}, W={overlap_width_ratio}",
                "confidence_threshold": confidence_threshold,
            },
            "summary": {
                "total_detections": len(predictions),
                "small": small_count,
                "medium": medium_count,
                "large": large_count,
            },
            "detections": detection_results,
        }

        json_path = output_path / f"sahi_{img_path.stem}_results.json"
        with open(json_path, "w") as f:
            json.dump(json_output, f, indent=2)
        print(f"\n  💾 Results saved to: {json_path}")

    return result


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SAHI Sliced Inference for Small Object Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python src/sahi_inference.py --image test_images/drone_4k.jpg

  # Custom slice size and overlap
  python src/sahi_inference.py --image test_images/sat.jpg --slice 640 --overlap 0.3

  # Use config file
  python src/sahi_inference.py --config config/sahi_config.yaml --image test.jpg

  # Lower confidence for more detections
  python src/sahi_inference.py --image test.jpg --conf 0.2
        """,
    )

    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to the input image"
    )
    parser.add_argument(
        "--model", type=str,
        default="detection_models/small_object_p2_1280/weights/best.pt",
        help="Path to trained model weights"
    )
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--slice", type=int, default=512, help="Slice size (pixels)")
    parser.add_argument("--overlap", type=float, default=0.2, help="Overlap ratio")
    parser.add_argument("--output", type=str, default="output", help="Output dir")
    parser.add_argument("--no-visual", action="store_true", help="Skip visual export")
    parser.add_argument("--no-json", action="store_true", help="Skip JSON export")
    parser.add_argument("--crop", action="store_true", help="Export cropped detections")
    parser.add_argument(
        "--no-full-image", action="store_true",
        help="Disable full-image prediction (only sliced)"
    )
    parser.add_argument("--config", type=str, default=None, help="SAHI config YAML")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_sahi_inference(
        image_path=args.image,
        model_weights=args.model,
        confidence_threshold=args.conf,
        device=args.device,
        slice_height=args.slice,
        slice_width=args.slice,
        overlap_height_ratio=args.overlap,
        overlap_width_ratio=args.overlap,
        perform_standard_prediction=not args.no_full_image,
        export_dir=args.output,
        export_visual=not args.no_visual,
        export_crop=args.crop,
        save_json=not args.no_json,
        config_file=args.config,
    )
