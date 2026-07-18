"""
=============================================================================
COCO mAP_S Evaluation — Small Object Detection Metrics
=============================================================================
Evaluates the full SAHI pipeline using industry-standard COCO metrics,
with focus on mAP_S (Mean Average Precision for Small objects < 32²px).

WHY NOT USE model.val():
  Standard YOLO validation bypasses SAHI slicing logic. It evaluates the
  model on full images at a single resolution, producing artificially low
  scores for small objects. To measure the true end-to-end pipeline
  performance, predictions must go through SAHI slicing.

COCO METRIC BREAKDOWN:
  mAP_S: Area < 32² pixels (1024 px²)  — PRIMARY METRIC
  mAP_M: 32² ≤ Area ≤ 96² pixels
  mAP_L: Area > 96² pixels

REQUIREMENTS:
  1. Validation images in a directory
  2. COCO-format ground truth JSON (instances_val.json)
     Use src/utils/coco_converter.py to convert from YOLO format
  3. Trained model weights

USAGE:
  python src/evaluate.py
  python src/evaluate.py --val-images dataset/val/images/ --gt dataset/val/annotations/instances_val.json
=============================================================================
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def run_sahi_validation(
    val_images_dir: str,
    coco_ground_truth_json: str,
    model_weights: str = "detection_models/small_object_p2_1280/weights/best.pt",
    model_type: str = "yolov8",
    slice_height: int = 512,
    slice_width: int = 512,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    confidence_threshold: float = 0.01,
    device: str = "cuda:0",
    project: str = "metrics",
    name: str = "sahi_evaluation",
):
    """Run SAHI sliced predictions over the entire validation set.

    Processes every image in the validation directory through SAHI slicing
    and exports predictions in COCO-compatible JSON format.

    Args:
        val_images_dir: Directory containing validation images.
        coco_ground_truth_json: Path to COCO ground truth JSON.
        model_weights: Path to trained model weights.
        model_type: Detection model type.
        slice_height: Patch height for slicing.
        slice_width: Patch width for slicing.
        overlap_height_ratio: Vertical overlap ratio.
        overlap_width_ratio: Horizontal overlap ratio.
        confidence_threshold: Low threshold for evaluation (to capture full PR curve).
        device: Compute device.
        project: Output directory.
        name: Run name.

    Returns:
        Path to the COCO-format prediction results JSON.
    """
    print("=" * 70)
    print("  SAHI VALIDATION — GENERATING PREDICTIONS")
    print("=" * 70)

    # Validate inputs
    val_dir = Path(val_images_dir)
    gt_path = Path(coco_ground_truth_json)

    if not val_dir.exists():
        print(f"[ERROR] Validation images directory not found: {val_images_dir}")
        sys.exit(1)

    if not gt_path.exists():
        print(f"[ERROR] Ground truth JSON not found: {coco_ground_truth_json}")
        print("        Convert YOLO labels to COCO format using:")
        print("        python src/utils/coco_converter.py")
        sys.exit(1)

    weights_path = Path(model_weights)
    if not weights_path.exists():
        print(f"[ERROR] Model weights not found: {model_weights}")
        sys.exit(1)

    # Count validation images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    val_images = [f for f in val_dir.iterdir() if f.suffix.lower() in image_extensions]
    print(f"\n  Validation images  : {len(val_images)} found in {val_images_dir}")
    print(f"  Ground truth       : {coco_ground_truth_json}")
    print(f"  Model              : {model_weights}")
    print(f"  Slice size         : {slice_width}×{slice_height}px")
    print(f"  Overlap            : {overlap_height_ratio}")
    print(f"  Confidence         : {confidence_threshold}")
    print()

    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    # Load the model once using the same API as sahi_inference.py
    print("[INFO] Loading model into SAHI wrapper...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type=model_type,
        model_path=str(model_weights),
        confidence_threshold=confidence_threshold,
        device=device,
    )
    print("[INFO] Model loaded successfully.")

    # Build filename → integer image_id map from the ground truth COCO JSON
    # so prediction image_ids match exactly what pycocotools expects
    with open(gt_path, "r") as f:
        gt_data = json.load(f)
    filename_to_id = {img["file_name"]: img["id"] for img in gt_data["images"]}
    stem_to_id = {Path(fn).stem: img_id for fn, img_id in filename_to_id.items()}

    # Run sliced inference over every validation image and collect COCO predictions
    coco_predictions = []
    annotation_id = 1

    print("[INFO] Running SAHI sliced prediction over validation set...")
    for i, img_path in enumerate(sorted(val_images), 1):
        print(f"  [{i}/{len(val_images)}] Processing: {img_path.name}")
        image_id = stem_to_id.get(img_path.stem) or filename_to_id.get(img_path.name)
        if image_id is None:
            print(f"  [WARN] {img_path.name} not found in ground truth JSON — skipping.")
            continue

        result = get_sliced_prediction(
            image=str(img_path),
            detection_model=detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            verbose=0,
        )
        for obj_pred in result.object_prediction_list:
            bbox = obj_pred.bbox
            coco_predictions.append({
                "id": annotation_id,
                "image_id": image_id,              # integer ID from GT JSON
                "category_id": obj_pred.category.id,
                "bbox": [
                    round(float(bbox.minx), 2),
                    round(float(bbox.miny), 2),
                    round(float(bbox.maxx - bbox.minx), 2),
                    round(float(bbox.maxy - bbox.miny), 2),
                ],
                "score": round(float(obj_pred.score.value), 4),
                "area": round(float((bbox.maxx - bbox.minx) * (bbox.maxy - bbox.miny)), 2),
                "iscrowd": 0,
            })
            annotation_id += 1

    # Save COCO-format predictions JSON
    output_path = Path(project) / name
    output_path.mkdir(parents=True, exist_ok=True)
    result_json = output_path / "result.json"
    with open(result_json, "w") as f:
        json.dump(coco_predictions, f, indent=2)

    print(f"\n[INFO] Predictions saved to: {result_json}")
    print(f"[INFO] Total predictions across all images: {len(coco_predictions)}")
    return str(result_json)


def run_coco_evaluation(
    coco_ground_truth_json: str,
    prediction_results_json: str,
    output_dir: str = "metrics",
):
    """Execute COCO evaluation and compute mAP metrics by object size.

    Uses pycocotools COCOeval to compute the full suite of COCO metrics
    including mAP_S (small), mAP_M (medium), and mAP_L (large).

    Args:
        coco_ground_truth_json: Path to COCO ground truth JSON.
        prediction_results_json: Path to COCO-format prediction JSON.
        output_dir: Directory to save evaluation report.

    Returns:
        Dictionary of evaluation metrics.
    """
    print("\n" + "=" * 70)
    print("  COCO EVALUATION ENGINE — mAP BY OBJECT SIZE")
    print("=" * 70)

    # Load ground truth
    print("\n[STEP 1/3] Loading ground truth annotations...")
    coco_gt = COCO(coco_ground_truth_json)

    # Check whether the predictions file is non-empty before passing to pycocotools.
    # pycocotools crashes with IndexError if the predictions list is empty ([]).
    print("[STEP 2/3] Loading SAHI predictions...")
    with open(prediction_results_json, "r") as f:
        predictions_list = json.load(f)

    zero_metrics = {
        "mAP_50:95": 0.0, "mAP_50": 0.0, "mAP_75": 0.0,
        "mAP_S": 0.0, "mAP_M": 0.0, "mAP_L": 0.0,
        "AR_1": 0.0, "AR_10": 0.0, "AR_100": 0.0,
        "AR_S": 0.0, "AR_M": 0.0, "AR_L": 0.0,
    }

    if not predictions_list:
        print("\n  ⚠️  No predictions were made (model confidence too low or model untrained).")
        print("     This is expected for a 1-epoch smoke-test on toy data.")
        print("     Train the real model with: python src/train.py")
        print("\n[STEP 3/3] Reporting zero metrics (no predictions to evaluate).\n")
        metrics = zero_metrics
    else:
        coco_dt = coco_gt.loadRes(prediction_results_json)

        # Run evaluation
        print("[STEP 3/3] Computing COCO metrics...\n")
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract metrics
        stats = coco_eval.stats
        metrics = {
            "mAP_50:95": round(float(stats[0]), 4),
            "mAP_50":    round(float(stats[1]), 4),
            "mAP_75":    round(float(stats[2]), 4),
            "mAP_S":     round(float(stats[3]), 4),
            "mAP_M":     round(float(stats[4]), 4),
            "mAP_L":     round(float(stats[5]), 4),
            "AR_1":      round(float(stats[6]), 4),
            "AR_10":     round(float(stats[7]), 4),
            "AR_100":    round(float(stats[8]), 4),
            "AR_S":      round(float(stats[9]), 4),
            "AR_M":      round(float(stats[10]), 4),
            "AR_L":      round(float(stats[11]), 4),
        }

    # Display formatted results
    print(f"\n{'═' * 60}")
    print(f"  📊 COCO EVALUATION RESULTS")
    print(f"{'═' * 60}")
    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │  MEAN AVERAGE PRECISION (mAP)                   │")
    print(f"  ├─────────────────────────────────────────────────┤")
    print(f"  │  mAP @[IoU=0.50:0.95]     :  {metrics['mAP_50:95']:.4f}            │")
    print(f"  │  mAP @[IoU=0.50]          :  {metrics['mAP_50']:.4f}            │")
    print(f"  │  mAP @[IoU=0.75]          :  {metrics['mAP_75']:.4f}            │")
    print(f"  ├─────────────────────────────────────────────────┤")
    print(f"  │  BY OBJECT SIZE                                 │")
    print(f"  ├─────────────────────────────────────────────────┤")
    print(f"  │  ★ mAP_S (Small < 32²px)  :  {metrics['mAP_S']:.4f}  ← PRIMARY │")
    print(f"  │    mAP_M (Medium)          :  {metrics['mAP_M']:.4f}            │")
    print(f"  │    mAP_L (Large)           :  {metrics['mAP_L']:.4f}            │")
    print(f"  ├─────────────────────────────────────────────────┤")
    print(f"  │  AVERAGE RECALL (AR)                            │")
    print(f"  ├─────────────────────────────────────────────────┤")
    print(f"  │  AR_S  (Small)             :  {metrics['AR_S']:.4f}            │")
    print(f"  │  AR_M  (Medium)            :  {metrics['AR_M']:.4f}            │")
    print(f"  │  AR_L  (Large)             :  {metrics['AR_L']:.4f}            │")
    print(f"  └─────────────────────────────────────────────────┘")

    # Save evaluation report
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "ground_truth": str(coco_ground_truth_json),
        "predictions": str(prediction_results_json),
        "metrics": metrics,
    }

    report_path = output_path / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  💾 Evaluation report saved to: {report_path}")
    return metrics


def calculate_maps(
    val_images_dir: str = "dataset/val/images/",
    coco_ground_truth_json: str = "dataset/val/annotations/instances_val.json",
    model_weights: str = "detection_models/small_object_p2_1280/weights/best.pt",
    model_type: str = "yolov8",
    slice_height: int = 512,
    slice_width: int = 512,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    output_dir: str = "metrics",
):
    """Full evaluation pipeline: SAHI prediction + COCO mAP computation.

    Args:
        val_images_dir: Directory containing validation images.
        coco_ground_truth_json: Path to COCO ground truth JSON.
        model_weights: Path to trained model weights.
        model_type: Detection model type.
        slice_height: Patch height for slicing.
        slice_width: Patch width for slicing.
        overlap_height_ratio: Vertical overlap ratio.
        overlap_width_ratio: Horizontal overlap ratio.
        output_dir: Directory to save evaluation results.

    Returns:
        Dictionary of COCO evaluation metrics.
    """
    # Step 1: Run SAHI predictions over validation set
    prediction_json = run_sahi_validation(
        val_images_dir=val_images_dir,
        coco_ground_truth_json=coco_ground_truth_json,
        model_weights=model_weights,
        model_type=model_type,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        project=output_dir,
        name="sahi_evaluation",
    )

    # Step 2: Run COCO evaluation
    metrics = run_coco_evaluation(
        coco_ground_truth_json=coco_ground_truth_json,
        prediction_results_json=prediction_json,
        output_dir=output_dir,
    )

    return metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="COCO mAP_S Evaluation for Small Object Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full evaluation with defaults
  python src/evaluate.py

  # Custom paths
  python src/evaluate.py --val-images dataset/val/images/ --gt dataset/val/annotations/instances_val.json

  # Custom slicing parameters
  python src/evaluate.py --slice 640 --overlap 0.3
        """,
    )

    parser.add_argument(
        "--val-images", type=str, default="dataset/val/images/",
        help="Path to validation images directory"
    )
    parser.add_argument(
        "--gt", type=str, default="dataset/val/annotations/instances_val.json",
        help="Path to COCO ground truth JSON"
    )
    parser.add_argument(
        "--model", type=str,
        default="detection_models/small_object_p2_1280/weights/best.pt",
        help="Path to trained model weights"
    )
    parser.add_argument("--slice", type=int, default=512, help="Slice size (pixels)")
    parser.add_argument("--overlap", type=float, default=0.2, help="Overlap ratio")
    parser.add_argument("--output", type=str, default="metrics", help="Output dir")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    calculate_maps(
        val_images_dir=args.val_images,
        coco_ground_truth_json=args.gt,
        model_weights=args.model,
        slice_height=args.slice,
        slice_width=args.slice,
        overlap_height_ratio=args.overlap,
        overlap_width_ratio=args.overlap,
        output_dir=args.output,
    )
