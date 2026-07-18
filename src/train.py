"""
=============================================================================
Training Pipeline — Advanced Small Object Detection
=============================================================================
Trains a YOLOv8n-P2 model at 1280px resolution for detecting small objects
(targets < 32×32 pixels). The P2 architecture adds a fourth detection head
operating at stride 4 (160×160 grid at 640px, 320×320 grid at 1280px).

KEY DESIGN DECISIONS:
  - 1280px input: Preserves pixel density for small targets that would
    vanish at the standard 640px resolution
  - P2 head: Taps into early backbone features before excessive downsampling
  - AdamW optimizer: Better convergence for the larger P2 architecture
  - Conservative augmentation: Aggressive transforms destroy tiny objects
  - 150 epochs: Small objects need more training iterations to converge

USAGE:
  python src/train.py
  python src/train.py --config config/training_config.yaml
  python src/train.py --imgsz 1280 --epochs 200 --batch 8
=============================================================================
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


def load_config(config_path: str) -> dict:
    """Load training configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary of configuration parameters.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"[WARNING] Config file not found: {config_path}")
        print("          Using default parameters.")
        return {}

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    print(f"[INFO] Loaded configuration from: {config_path}")
    return config


def get_device() -> str:
    """Detect the best available compute device.

    Returns:
        Device string for YOLO training ("0" for first GPU, "cpu" otherwise).
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[INFO] GPU detected: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        return "0"
    else:
        print("[WARNING] No CUDA GPU detected. Training on CPU (will be slow).")
        return "cpu"


def estimate_batch_size(imgsz: int, device: str) -> int:
    """Estimate a safe batch size based on image size and available VRAM.

    The relationship between image size and VRAM usage is quadratic:
    doubling imgsz quadruples memory consumption.

    Args:
        imgsz: Training image size in pixels.
        device: Compute device string.

    Returns:
        Recommended batch size.
    """
    if device == "cpu":
        return 2

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    # Conservative estimates for P2 architecture (4 heads use more memory)
    if imgsz >= 1280:
        if vram_gb >= 48:
            return 16
        elif vram_gb >= 24:
            return 8
        elif vram_gb >= 16:
            return 4
        else:
            return 2
    elif imgsz >= 960:
        if vram_gb >= 24:
            return 16
        elif vram_gb >= 16:
            return 8
        else:
            return 4
    else:  # 640px
        if vram_gb >= 16:
            return 16
        elif vram_gb >= 8:
            return 8
        else:
            return 4


def train_small_target_detector(
    model_config: str = "models/yolov8n-p2.yaml",
    data_config: str = "config/dataset.yaml",
    epochs: int = 150,
    imgsz: int = 1280,
    batch: int = None,
    device: str = None,
    project: str = "detection_models",
    name: str = "small_object_p2_1280",
    optimizer: str = "AdamW",
    lr0: float = 0.001,
    resume: bool = False,
    config_file: str = None,
):
    """Train a P2 small object detection model.

    Args:
        model_config: Path to the YOLO P2 architecture YAML.
        data_config: Path to the dataset YAML.
        epochs: Number of training epochs.
        imgsz: Input image size (pixels). Use 1280+ for small objects.
        batch: Batch size. Auto-estimated if None.
        device: Compute device. Auto-detected if None.
        project: Output project directory.
        name: Run name for this training session.
        optimizer: Optimizer type ("AdamW", "SGD", "Adam").
        lr0: Initial learning rate.
        resume: Resume from last checkpoint.
        config_file: Optional YAML config file to override defaults.
    """
    print("=" * 70)
    print("  ADVANCED SMALL OBJECT DETECTION — TRAINING PIPELINE")
    print("=" * 70)

    # Load config file overrides if provided
    if config_file:
        config = load_config(config_file)
        model_config = config.get("model", model_config)
        data_config = config.get("data", data_config)
        epochs = config.get("epochs", epochs)
        imgsz = config.get("imgsz", imgsz)
        batch = config.get("batch", batch)
        optimizer = config.get("optimizer", optimizer)
        lr0 = config.get("lr0", lr0)
        project = config.get("project", project)
        name = config.get("name", name)

    # Detect compute device
    if device is None:
        device = get_device()

    # Estimate batch size if not specified
    if batch is None:
        batch = estimate_batch_size(imgsz, device)
        print(f"[INFO] Auto-estimated batch size: {batch}")

    # Print training configuration
    print(f"\n{'─' * 50}")
    print(f"  Model Architecture : {model_config}")
    print(f"  Dataset Config     : {data_config}")
    print(f"  Image Size         : {imgsz}px")
    print(f"  Batch Size         : {batch}")
    print(f"  Epochs             : {epochs}")
    print(f"  Optimizer          : {optimizer}")
    print(f"  Learning Rate      : {lr0}")
    print(f"  Device             : {device}")
    print(f"  Output             : {project}/{name}")
    print(f"{'─' * 50}\n")

    # Validate model config exists
    model_path = Path(model_config)
    if not model_path.exists():
        print(f"[ERROR] Model config not found: {model_config}")
        print("        Ensure yolov8n-p2.yaml exists in models/ directory.")
        sys.exit(1)

    # Validate dataset config exists
    data_path = Path(data_config)
    if not data_path.exists():
        print(f"[ERROR] Dataset config not found: {data_config}")
        print("        Ensure dataset.yaml exists in config/ directory.")
        sys.exit(1)

    # ─── Initialize Model ────────────────────────────────────────────────
    print("[STEP 1/3] Initializing P2 architecture...")

    if resume:
        # Resume from last checkpoint
        last_weights = Path(project) / name / "weights" / "last.pt"
        if last_weights.exists():
            model = YOLO(str(last_weights))
            print(f"[INFO] Resuming training from: {last_weights}")
        else:
            print(f"[WARNING] No checkpoint found at {last_weights}. Starting fresh.")
            model = YOLO(model_config)
    else:
        model = YOLO(model_config)

    print(f"[INFO] Model initialized with P2 head (4 detection heads)")

    # ─── Execute Training ────────────────────────────────────────────────
    print("\n[STEP 2/3] Starting high-resolution training...")
    print(f"[INFO] Training at {imgsz}px to preserve small object pixel density")

    results = model.train(
        data=data_config,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        optimizer=optimizer,
        lr0=lr0,
        exist_ok=True,
        save_period=10,         # Save checkpoint every 10 epochs
        plots=True,             # Generate training plots
        patience=30,            # Early stopping patience
        # Conservative augmentation for small objects
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=5.0,            # Minimal rotation
        translate=0.1,
        scale=0.3,
        shear=2.0,
        flipud=0.3,
        fliplr=0.5,
        mosaic=0.8,
        mixup=0.1,
        copy_paste=0.2,
    )

    # ─── Results Summary ─────────────────────────────────────────────────
    print("\n[STEP 3/3] Training complete!")
    print(f"{'─' * 50}")
    print(f"  Best weights saved to: {project}/{name}/weights/best.pt")
    print(f"  Last weights saved to: {project}/{name}/weights/last.pt")
    print(f"  Training plots saved to: {project}/{name}/")
    print(f"{'─' * 50}")

    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv8-P2 for Small Object Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with defaults (1280px, P2 architecture)
  python src/train.py

  # Train with custom config file
  python src/train.py --config config/training_config.yaml

  # Train with custom parameters
  python src/train.py --imgsz 1280 --epochs 200 --batch 8

  # Resume interrupted training
  python src/train.py --resume
        """,
    )

    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to training config YAML file"
    )
    parser.add_argument(
        "--model", type=str, default="models/yolov8n-p2.yaml",
        help="Path to model architecture YAML"
    )
    parser.add_argument(
        "--data", type=str, default="config/dataset.yaml",
        help="Path to dataset YAML"
    )
    parser.add_argument(
        "--epochs", type=int, default=150,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--imgsz", type=int, default=1280,
        help="Training image size in pixels"
    )
    parser.add_argument(
        "--batch", type=int, default=None,
        help="Batch size (auto-estimated if not set)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Compute device ('0' for GPU, 'cpu' for CPU)"
    )
    parser.add_argument(
        "--optimizer", type=str, default="AdamW",
        choices=["AdamW", "SGD", "Adam"],
        help="Optimizer type"
    )
    parser.add_argument(
        "--lr0", type=float, default=0.001,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--project", type=str, default="detection_models",
        help="Output project directory"
    )
    parser.add_argument(
        "--name", type=str, default="small_object_p2_1280",
        help="Run name"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from last checkpoint"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_small_target_detector(
        model_config=args.model,
        data_config=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        optimizer=args.optimizer,
        lr0=args.lr0,
        resume=args.resume,
        config_file=args.config,
    )
