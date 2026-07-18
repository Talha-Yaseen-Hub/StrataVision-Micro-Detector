# 🔬 Advanced Small Object Detection

### Architectural Optimization, Sliced Inference (SAHI), and High-Resolution Analysis

---

## 📖 Overview

This project implements an end-to-end pipeline for detecting **small objects** (targets occupying fewer than 32×32 pixels) in high-resolution imagery. It addresses the fundamental challenge of **Feature Loss** in CNNs — where small objects vanish as spatial resolution is reduced through successive network layers.

### Key Components

| Component | Description |
|---|---|
| **P2 Detection Head** | Custom 4-head YOLO architecture tapping into shallow, high-resolution P2 features (160×160 grid at 640px) |
| **High-Resolution Training** | Training at 1280px to preserve pixel density for small targets |
| **SAHI Inference** | Slicing Aided Hyper Inference for processing 4K/8K images without downscaling |
| **COCO mAP_S Evaluation** | Rigorous evaluation using industry-standard small-object metrics |

---

## 📁 Project Structure

```
Object-Detection/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
├── config/
│   ├── dataset.yaml               # Dataset configuration
│   ├── training_config.yaml       # Training hyperparameters
│   └── sahi_config.yaml           # SAHI inference settings
├── models/
│   └── yolov8n-p2.yaml            # P2 architecture definition
├── src/
│   ├── __init__.py
│   ├── train.py                   # Training pipeline
│   ├── inference.py               # Standard YOLO inference
│   ├── sahi_inference.py          # SAHI sliced inference
│   ├── evaluate.py                # COCO mAP_S evaluation
│   └── utils/
│       ├── __init__.py
│       ├── dataset_utils.py       # Dataset preparation & validation
│       ├── visualization.py       # Detection visualization tools
│       └── coco_converter.py      # YOLO-to-COCO format converter
├── dataset/
│   ├── train/
│   │   ├── images/                # Training images
│   │   └── labels/                # YOLO format labels (.txt)
│   ├── val/
│   │   ├── images/                # Validation images
│   │   ├── labels/                # YOLO format labels (.txt)
│   │   └── annotations/          # COCO JSON annotations
│   └── test/
│       ├── images/                # Test images
│       └── labels/                # Test labels
├── test_images/                   # High-resolution test images for SAHI
├── output/                        # Inference output (visualizations)
├── detection_models/              # Trained model weights
└── metrics/                       # Evaluation results
```

---

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Place your annotated images in the `dataset/` directory following the structure above. Labels must be in YOLO format:

```
[class_id] [x_center] [y_center] [width] [height]
```

> ⚠️ **Critical**: Bounding boxes for small objects must be **extremely tight**. Any background pixels inside the box act as noise.

### 3. Train the Model

```bash
python src/train.py
```

### 4. Run SAHI Inference

```bash
python src/sahi_inference.py --image test_images/your_image.jpg
```

### 5. Evaluate (mAP_S)

```bash
python src/evaluate.py
```

---

## 🏗️ Architecture: P2 Feature Head

Standard YOLO models extract predictions from three layers (P3, P4, P5). This project adds the **P2 layer** for micro-object detection:

| Layer | Stride | Grid (640px) | Grid (1280px) | Target |
|---|---|---|---|---|
| **P2** | 4 | 160×160 | 320×320 | Tiny objects (< 16px) |
| **P3** | 8 | 80×80 | 160×160 | Small objects (16–32px) |
| **P4** | 16 | 40×40 | 80×80 | Medium objects (32–96px) |
| **P5** | 32 | 20×20 | 40×40 | Large objects (> 96px) |

The P2 layer undergoes fewer downsampling operations, retaining fine spatial details critical for tiny target detection.

---

## 📊 Evaluation Metrics (COCO Standard)

| Metric | Area Threshold | Description |
|---|---|---|
| **mAP_S** | < 32² pixels | Small object precision — **primary metric** |
| **mAP_M** | 32²–96² pixels | Medium object precision |
| **mAP_L** | > 96² pixels | Large object precision |

---

## ⚙️ Configuration

All hyperparameters are configurable via YAML files in `config/`:

- **`dataset.yaml`** — Dataset paths and class definitions
- **`training_config.yaml`** — Epochs, image size, batch size, optimizer, learning rate
- **`sahi_config.yaml`** — Slice dimensions, overlap ratios, confidence thresholds

---

## 📝 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
