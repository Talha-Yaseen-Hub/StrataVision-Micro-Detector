"""
=============================================================================
Synthetic Dataset Generator — Advanced Small Object Detection
=============================================================================
Generates a realistic synthetic training dataset covering all 5 classes:
  0: drone, 1: bird, 2: vehicle, 3: person, 4: small_component

Each image is 640×640 with a randomized background (sky, ground, urban)
and multiple small objects rendered as styled shapes with noise/blur.
Generates YOLO-format .txt label files automatically.

USAGE:
  python generate_dataset.py
  python generate_dataset.py --train 500 --val 100 --test 50
=============================================================================
"""

import argparse
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ─── Configuration ────────────────────────────────────────────────────────────
IMG_W, IMG_H = 640, 640
CLASS_NAMES = {0: "drone", 1: "bird", 2: "vehicle", 3: "person", 4: "small_component"}

# Object pixel size ranges (small = COCO small-object category)
SIZE_RANGES = {
    0: (8,  22),   # drone  — tiny, airborne
    1: (6,  16),   # bird   — very tiny
    2: (18, 45),   # vehicle — slightly larger
    3: (12, 28),   # person
    4: (4,  12),   # small_component — micro
}

# Realistic colour palettes per class
COLORS = {
    0: [(220, 30, 30), (200, 200, 50), (30, 30, 200), (150, 150, 150)],  # drone
    1: [(80, 60, 40), (60, 80, 60), (200, 180, 100), (240, 240, 200)],   # bird
    2: [(50, 120, 200), (200, 50, 50), (50, 180, 50), (250, 200, 50)],   # vehicle
    3: [(220, 160, 100), (100, 80, 60), (200, 120, 80), (180, 140, 100)],# person
    4: [(200, 200, 200), (50, 200, 200), (200, 50, 200), (200, 200, 50)],# component
}


# ─── Background Generators ────────────────────────────────────────────────────

def make_sky_background():
    """Blue-gradient sky with random clouds."""
    img = Image.new("RGB", (IMG_W, IMG_H))
    draw = ImageDraw.Draw(img)
    for y in range(IMG_H):
        t = y / IMG_H
        r = int(135 + t * 60 + random.randint(-10, 10))
        g = int(180 + t * 50 + random.randint(-10, 10))
        b = int(220 + t * 30 + random.randint(-10, 10))
        draw.line([(0, y), (IMG_W, y)], fill=(min(r,255), min(g,255), min(b,255)))
    # Add cloud puffs
    for _ in range(random.randint(3, 8)):
        cx, cy = random.randint(0, IMG_W), random.randint(0, IMG_H // 2)
        for _ in range(random.randint(3, 7)):
            r = random.randint(15, 50)
            ox, oy = random.randint(-20, 20), random.randint(-10, 10)
            alpha = random.randint(180, 240)
            draw.ellipse([cx+ox-r, cy+oy-r, cx+ox+r, cy+oy+r],
                         fill=(alpha, alpha, alpha))
    return img.filter(ImageFilter.GaussianBlur(1))


def make_ground_background():
    """Earth/field ground with texture."""
    img = Image.new("RGB", (IMG_W, IMG_H))
    pixels = np.random.randint(0, 50, (IMG_H, IMG_W, 3), dtype=np.uint8)
    # Green/brown base
    base_r = random.randint(60, 120)
    base_g = random.randint(80, 150)
    base_b = random.randint(30, 80)
    pixels[:, :, 0] += base_r
    pixels[:, :, 1] += base_g
    pixels[:, :, 2] += base_b
    pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels).filter(ImageFilter.GaussianBlur(1.5))


def make_urban_background():
    """Overhead urban grid — rooftops, roads."""
    img = Image.new("RGB", (IMG_W, IMG_H), color=(90, 90, 90))
    draw = ImageDraw.Draw(img)
    # Draw road grid
    for x in range(0, IMG_W, random.randint(60, 120)):
        w = random.randint(8, 20)
        draw.rectangle([x, 0, x + w, IMG_H], fill=(60, 60, 60))
    for y in range(0, IMG_H, random.randint(60, 120)):
        h = random.randint(8, 20)
        draw.rectangle([0, y, IMG_W, y + h], fill=(60, 60, 60))
    # Draw rooftop rectangles
    for _ in range(random.randint(8, 20)):
        x1 = random.randint(0, IMG_W - 30)
        y1 = random.randint(0, IMG_H - 30)
        w = random.randint(20, 80)
        h = random.randint(20, 80)
        shade = random.randint(100, 180)
        draw.rectangle([x1, y1, x1 + w, y1 + h],
                       fill=(shade, shade - 10, shade - 20))
    # Add some noise
    noise = np.random.randint(-20, 20, (IMG_H, IMG_W, 3))
    arr = np.array(img).astype(np.int16) + noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def random_background():
    choice = random.random()
    if choice < 0.4:
        return make_sky_background()
    elif choice < 0.7:
        return make_ground_background()
    else:
        return make_urban_background()


# ─── Object Renderers ─────────────────────────────────────────────────────────

def draw_drone(draw, cx, cy, size, color):
    """X-frame drone shape."""
    arm = size // 2
    lw = max(1, size // 6)
    # Arms
    draw.line([cx - arm, cy - arm, cx + arm, cy + arm], fill=color, width=lw)
    draw.line([cx + arm, cy - arm, cx - arm, cy + arm], fill=color, width=lw)
    # Propellers
    for dx, dy in [(-arm, -arm), (arm, -arm), (-arm, arm), (arm, arm)]:
        pr = max(2, size // 5)
        draw.ellipse([cx+dx-pr, cy+dy-pr, cx+dx+pr, cy+dy+pr], fill=color)
    # Central body
    b = max(2, size // 4)
    draw.rectangle([cx - b, cy - b, cx + b, cy + b], fill=color)


def draw_bird(draw, cx, cy, size, color):
    """Simple bird silhouette — two arc wings."""
    w = size
    draw.arc([cx - w, cy - w // 2, cx, cy + w // 2], 180, 0, fill=color,
             width=max(1, w // 4))
    draw.arc([cx, cy - w // 2, cx + w, cy + w // 2], 180, 0, fill=color,
             width=max(1, w // 4))


def draw_vehicle(draw, cx, cy, size, color):
    """Overhead car silhouette."""
    hw = size // 2
    hh = max(3, int(size * 0.7)) // 2
    draw.rectangle([cx - hw, cy - hh, cx + hw, cy + hh], fill=color)
    # Windshields (lighter)
    ws = max(1, hw // 2)
    wh = max(1, hh // 3)
    draw.rectangle([cx - ws, cy - hh, cx + ws, cy - hh + wh],
                   fill=tuple(min(c + 60, 255) for c in color))
    draw.rectangle([cx - ws, cy + hh - wh, cx + ws, cy + hh],
                   fill=tuple(min(c + 60, 255) for c in color))


def draw_person(draw, cx, cy, size, color):
    """Simple overhead person — circle head + oval body."""
    head_r = max(2, size // 5)
    body_h = max(3, int(size * 0.7))
    body_w = max(2, size // 3)
    draw.ellipse([cx - head_r, cy - size // 2, cx + head_r,
                  cy - size // 2 + head_r * 2], fill=color)
    draw.ellipse([cx - body_w, cy - size // 2 + head_r * 2,
                  cx + body_w, cy - size // 2 + head_r * 2 + body_h],
                 fill=color)


def draw_small_component(draw, cx, cy, size, color):
    """Tiny cross / circuit component."""
    half = max(1, size // 2)
    lw = max(1, size // 4)
    draw.line([cx - half, cy, cx + half, cy], fill=color, width=lw)
    draw.line([cx, cy - half, cx, cy + half], fill=color, width=lw)
    draw.rectangle([cx - lw, cy - lw, cx + lw, cy + lw], fill=color)


RENDERERS = {
    0: draw_drone,
    1: draw_bird,
    2: draw_vehicle,
    3: draw_person,
    4: draw_small_component,
}


# ─── Image + Label Generator ─────────────────────────────────────────────────

def generate_sample(num_objects_range=(3, 10)):
    """Generate one (image, labels) pair."""
    img = random_background()
    draw = ImageDraw.Draw(img)

    num_objects = random.randint(*num_objects_range)
    labels = []

    for _ in range(num_objects):
        cls = random.randint(0, 4)
        size_min, size_max = SIZE_RANGES[cls]
        size = random.randint(size_min, size_max)
        margin = size
        cx = random.randint(margin, IMG_W - margin)
        cy = random.randint(margin, IMG_H - margin)
        color = random.choice(COLORS[cls])

        RENDERERS[cls](draw, cx, cy, size, color)

        # Compute YOLO bbox (bounding box tightly around object)
        half = size // 2 + 2
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(IMG_W, cx + half)
        y2 = min(IMG_H, cy + half)

        bw = (x2 - x1) / IMG_W
        bh = (y2 - y1) / IMG_H
        bcx = (x1 + (x2 - x1) / 2) / IMG_W
        bcy = (y1 + (y2 - y1) / 2) / IMG_H

        labels.append(f"{cls} {bcx:.6f} {bcy:.6f} {bw:.6f} {bh:.6f}")

    # Slight blur to simulate motion/depth
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 1.2)))

    return img, labels


def generate_split(split_dir: Path, n: int, start_idx: int = 0):
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        idx = start_idx + i
        img, labels = generate_sample()
        img_path = img_dir / f"img_{idx:05d}.jpg"
        lbl_path = lbl_dir / f"img_{idx:05d}.txt"
        img.save(img_path, quality=92)
        with open(lbl_path, "w") as f:
            f.write("\n".join(labels))

        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  [{i+1}/{n}] saved")

    print(f"  ✓ {n} samples written to {split_dir}")


# ─── Entry Point ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic Dataset Generator")
    parser.add_argument("--train", type=int, default=300,
                        help="Number of training images (default: 300)")
    parser.add_argument("--val", type=int, default=80,
                        help="Number of validation images (default: 80)")
    parser.add_argument("--test", type=int, default=40,
                        help="Number of test images (default: 40)")
    parser.add_argument("--out", type=str, default="dataset",
                        help="Output root directory (default: dataset/)")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.out)

    print("=" * 60)
    print("  SYNTHETIC DATASET GENERATOR")
    print("  Classes: drone, bird, vehicle, person, small_component")
    print("=" * 60)

    print(f"\n[1/3] Generating {args.train} TRAINING images...")
    generate_split(root / "train", args.train, start_idx=0)

    print(f"\n[2/3] Generating {args.val} VALIDATION images...")
    generate_split(root / "val", args.val, start_idx=args.train)

    print(f"\n[3/3] Generating {args.test} TEST images...")
    generate_split(root / "test", args.test, start_idx=args.train + args.val)

    total = args.train + args.val + args.test
    print(f"\n{'=' * 60}")
    print(f"  Dataset complete!  Total images: {total}")
    print(f"  Location: {root.resolve()}")
    print(f"\n  Next steps:")
    print(f"    1. Validate: python src/utils/dataset_utils.py --dataset-dir {root}")
    print(f"    2. Train:    python src/train.py --config config/training_config.yaml")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
