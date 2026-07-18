"""
=============================================================================
Visualization Utilities — Detection Result Display and Analysis
=============================================================================
Tools for visualizing object detection results, comparing SAHI vs standard
inference, and generating annotated images with size-coded bounding boxes.

FEATURES:
  - Draw COCO size-coded bounding boxes (green=small, yellow=medium, red=large)
  - Side-by-side comparison of SAHI vs standard inference
  - Slice grid visualization (how SAHI divides the image)
  - Detection confidence heatmaps
  - Annotation statistics plots

USAGE:
  python src/utils/visualization.py --image output/sahi_result.png
=============================================================================
"""

import json
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
# COCO size thresholds (in pixels²)
SMALL_AREA_THRESHOLD = 32 ** 2    # 1024 px²
MEDIUM_AREA_THRESHOLD = 96 ** 2   # 9216 px²

# Color scheme (BGR for OpenCV)
COLORS = {
    "small": (0, 255, 0),         # Green — small objects
    "medium": (0, 200, 255),      # Yellow — medium objects
    "large": (0, 0, 255),         # Red — large objects
    "default": (255, 128, 0),     # Blue — unclassified
    "grid": (200, 200, 200),      # Light gray — slice grid lines
}


def get_size_category(width: float, height: float) -> str:
    """Classify an object by its area using COCO size thresholds.

    Args:
        width: Object width in pixels.
        height: Object height in pixels.

    Returns:
        Size category string: "small", "medium", or "large".
    """
    area = width * height
    if area < SMALL_AREA_THRESHOLD:
        return "small"
    elif area < MEDIUM_AREA_THRESHOLD:
        return "medium"
    else:
        return "large"


def draw_detection(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    class_name: str,
    confidence: float,
    color: Tuple[int, int, int] = None,
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Draw a single detection bounding box with label on an image.

    Args:
        image: Input image (BGR, numpy array).
        bbox: Bounding box as (x1, y1, x2, y2).
        class_name: Class name string.
        confidence: Detection confidence (0-1).
        color: BGR color tuple. Auto-selected by size if None.
        thickness: Box border thickness.
        font_scale: Label font scale.

    Returns:
        Image with drawn bounding box.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    width = x2 - x1
    height = y2 - y1

    # Auto-select color based on object size
    if color is None:
        size_cat = get_size_category(width, height)
        color = COLORS.get(size_cat, COLORS["default"])

    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Draw label background
    label = f"{class_name} {confidence:.2f}"
    (label_w, label_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
    )

    label_y = max(y1, label_h + 5)
    cv2.rectangle(
        image,
        (x1, label_y - label_h - 5),
        (x1 + label_w + 4, label_y + baseline - 2),
        color,
        -1,  # Filled
    )

    # Draw label text (white on colored background)
    cv2.putText(
        image,
        label,
        (x1 + 2, label_y - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return image


def draw_all_detections(
    image_path: str,
    detections: List[dict],
    output_path: str = None,
    show: bool = False,
) -> np.ndarray:
    """Draw all detections on an image with size-coded colors.

    Args:
        image_path: Path to the input image.
        detections: List of detection dicts with keys:
            "bbox_xyxy", "class", "confidence".
        output_path: Optional path to save the annotated image.
        show: Display the result in a window.

    Returns:
        Annotated image as numpy array.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    for det in detections:
        bbox = det["bbox_xyxy"]
        class_name = det.get("class", "object")
        confidence = det.get("confidence", 0.0)

        image = draw_detection(image, bbox, class_name, confidence)

    if output_path:
        out_dir = Path(output_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, image)
        print(f"[INFO] Annotated image saved to: {output_path}")

    if show:
        cv2.imshow("Detections", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image


def draw_sahi_grid(
    image_path: str,
    slice_width: int = 512,
    slice_height: int = 512,
    overlap_w: float = 0.2,
    overlap_h: float = 0.2,
    output_path: str = None,
) -> np.ndarray:
    """Visualize how SAHI slices an image into overlapping patches.

    Draws the slice grid on the image to help understand coverage and
    identify potential boundary issues.

    Args:
        image_path: Path to the input image.
        slice_width: Width of each slice in pixels.
        slice_height: Height of each slice in pixels.
        overlap_w: Horizontal overlap ratio.
        overlap_h: Vertical overlap ratio.
        output_path: Optional path to save the visualization.

    Returns:
        Image with slice grid drawn.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img_h, img_w = image.shape[:2]

    # Calculate slice positions
    step_w = int(slice_width * (1 - overlap_w))
    step_h = int(slice_height * (1 - overlap_h))

    overlay = image.copy()
    color = COLORS["grid"]

    # Draw vertical lines
    x = 0
    while x < img_w:
        cv2.line(overlay, (x, 0), (x, img_h), color, 1)
        x_end = min(x + slice_width, img_w)
        cv2.line(overlay, (x_end, 0), (x_end, img_h), color, 1)
        x += step_w

    # Draw horizontal lines
    y = 0
    while y < img_h:
        cv2.line(overlay, (0, y), (img_w, y), color, 1)
        y_end = min(y + slice_height, img_h)
        cv2.line(overlay, (0, y_end), (img_w, y_end), color, 1)
        y += step_h

    # Blend with original
    result = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)

    # Add info text
    n_cols = max(1, (img_w - slice_width) // step_w + 1)
    n_rows = max(1, (img_h - slice_height) // step_h + 1)
    info = (
        f"SAHI Grid: {n_cols}x{n_rows} = {n_cols * n_rows} slices | "
        f"Slice: {slice_width}x{slice_height} | Overlap: {overlap_w:.0%}"
    )
    cv2.putText(result, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if output_path:
        out_dir = Path(output_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, result)
        print(f"[INFO] SAHI grid visualization saved to: {output_path}")

    return result


def plot_size_distribution(
    detections: List[dict],
    output_path: str = "output/size_distribution.png",
    title: str = "Detection Size Distribution",
):
    """Generate a bar chart of detection sizes by COCO category.

    Args:
        detections: List of detection dicts with "area" or "width"/"height".
        output_path: Path to save the plot.
        title: Plot title.
    """
    areas = []
    for det in detections:
        if "area" in det:
            areas.append(det["area"])
        elif "width" in det and "height" in det:
            areas.append(det["width"] * det["height"])

    if not areas:
        print("[WARNING] No detections to plot.")
        return

    # Classify
    small = sum(1 for a in areas if a < SMALL_AREA_THRESHOLD)
    medium = sum(1 for a in areas if SMALL_AREA_THRESHOLD <= a < MEDIUM_AREA_THRESHOLD)
    large = sum(1 for a in areas if a >= MEDIUM_AREA_THRESHOLD)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    categories = ["Small\n(< 32²px)", "Medium\n(32²–96²px)", "Large\n(> 96²px)"]
    counts = [small, medium, large]
    bar_colors = ["#2ecc71", "#f39c12", "#e74c3c"]

    axes[0].bar(categories, counts, color=bar_colors, edgecolor="white", linewidth=1.5)
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"{title} — By Category")
    axes[0].grid(axis="y", alpha=0.3)

    for i, count in enumerate(counts):
        axes[0].text(i, count + 0.5, str(count), ha="center", fontweight="bold")

    # Histogram of areas
    axes[1].hist(areas, bins=50, color="#3498db", edgecolor="white", alpha=0.8)
    axes[1].axvline(SMALL_AREA_THRESHOLD, color="#2ecc71", linestyle="--",
                    label=f"Small threshold ({SMALL_AREA_THRESHOLD}px²)")
    axes[1].axvline(MEDIUM_AREA_THRESHOLD, color="#f39c12", linestyle="--",
                    label=f"Medium threshold ({MEDIUM_AREA_THRESHOLD}px²)")
    axes[1].set_xlabel("Object Area (px²)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"{title} — Area Histogram")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    # Save
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Size distribution plot saved to: {output_path}")


def load_sahi_results(json_path: str) -> List[dict]:
    """Load detection results from a SAHI JSON output file.

    Args:
        json_path: Path to the SAHI results JSON file.

    Returns:
        List of detection dictionaries.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "detections" in data:
        return data["detections"]
    elif isinstance(data, list):
        return data
    else:
        print(f"[WARNING] Unexpected JSON structure in {json_path}")
        return []


def export_to_coco(detections: List[dict], image_id: int, image_path: str) -> dict:
    """Convert detection results to a COCO‑style dictionary.

    Args:
        detections: List of detection dictionaries with keys ``bbox_xyxy``, ``class`` and ``confidence``.
        image_id: Identifier for the image (should be unique within a dataset).
        image_path: Path to the source image.

    Returns:
        A dictionary compliant with a minimal COCO format containing ``images``, ``annotations`` and ``categories``.
    """

    annotations = []
    for idx, det in enumerate(detections, start=1):
        bbox = det["bbox_xyxy"]
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        # Simple category handling – always id 1; extend as needed.
        annotation = {
            "id": idx,
            "image_id": image_id,
            "category_id": 1,
            "bbox": [x1, y1, width, height],
            "area": area,
            "iscrowd": 0,
            "segmentation": [],
            "score": det.get("confidence", 0.0),
        }
        annotations.append(annotation)

    coco_dict = {
        "images": [{
            "id": image_id,
            "file_name": Path(image_path).name,
            "width": None,
            "height": None,
        }],
        "annotations": annotations,
        "categories": [{"id": 1, "name": "object"}],
    }
    return coco_dict


def generate_html_report(
    detections: List[dict],
    image_path: str,
    output_path: str = "output/report.html",
) -> None:
    """Create an interactive HTML report visualising detections on the image.

    The report uses Plotly to overlay bounding boxes and labels on the original image.
    It is saved as a self‑contained HTML file that can be opened in any browser.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError(
            "plotly is required for HTML reports. "
            "Install it with: pip install plotly"
        )
    import base64

    # Load image and encode for embedding.
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    height, width = img.shape[:2]
    _, buffer = cv2.imencode('.png', img)
    encoded = base64.b64encode(buffer).decode()

    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source="data:image/png;base64," + encoded,
            xref="x",
            yref="y",
            x=0,
            y=height,
            sizex=width,
            sizey=height,
            sizing="stretch",
            layer="below",
        )
    )

    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        class_name = det.get("class", "object")
        confidence = det.get("confidence", 0.0)
        fig.add_shape(
            type="rect",
            x0=x1,
            y0=y1,
            x1=x2,
            y1=y2,
            line=dict(color="red", width=2),
            fillcolor="rgba(255,0,0,0.1)",
        )
        fig.add_annotation(
            x=x1,
            y=y1,
            text=f"{class_name} {confidence:.2f}",
            showarrow=False,
            yshift=10,
            bgcolor="rgba(255,255,255,0.7)",
        )

    fig.update_xaxes(visible=False, range=[0, width])
    fig.update_yaxes(visible=False, autorange="reversed", range=[0, height])
    fig.update_layout(width=width, height=height, margin=dict(l=0, r=0, t=0, b=0))

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    print(f"[INFO] HTML report saved to: {output_path}")
