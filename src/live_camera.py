"""
=============================================================================
Live Universal Object Detection — YOLOv8x + Pose + YOLO-World
=============================================================================
THREE detection modes:

  DEFAULT MODE  (yolov8x.pt + yolov8n-pose.pt):
    80 COCO objects + 17 body keypoints. No internet beyond first download.
    Reliable, fast, works offline after download.

  YOLO-WORLD MODE  (--world flag):
    Open-vocabulary: detects 200+ classes described in plain English.
    Requires CLIP (~338MB). Pre-download with: python download_clip.py
    Press C to cycle class sets: All / Body / Clothing / Everyday / Tiny
    Classes: face, hand, arm, leg, shoes, cap, door, fan, plate, can,
             ant, button, coin, biscuit, candy, bag, basket, bulb...

  SAHI mode  (--sahi flag, works in both modes):
    Slices each frame into overlapping patches for tiny object detection.

CONTROLS:
  Q / ESC  → Quit              S  → Save screenshot
  P        → Pause/Resume      M  → Toggle SAHI mode
  Z        → Zoom 1x/2x/4x    +/- → Confidence
  O        → Objects ON/OFF    B  → Body pose ON/OFF
  C        → Cycle class set   (YOLO-World mode only)
  I        → Toggle labels     H  → Help

USAGE:
  python src/live_camera.py                    # default (objects + pose)
  python src/live_camera.py --world            # YOLO-World (200+ classes)
  python src/live_camera.py --world --sahi     # YOLO-World + SAHI
  python src/live_camera.py --sahi             # default + SAHI
  python src/live_camera.py --source 1         # second camera
=============================================================================
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ─── COCO 80 class list ───────────────────────────────────────────────────────
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# ─── YOLO-World class sets (plain-English open-vocabulary) ────────────────────
WORLD_SETS = {
    "body": [
        "face", "head", "eye", "nose", "mouth", "ear", "forehead", "chin",
        "hand", "finger", "thumb", "palm", "wrist",
        "arm", "forearm", "elbow", "shoulder",
        "leg", "foot", "toe", "knee", "ankle", "thigh", "calf",
        "neck", "torso", "chest", "back", "hip",
        "hair", "beard", "person", "baby",
    ],
    "clothing": [
        "shirt", "t-shirt", "blouse", "jacket", "coat", "hoodie", "sweater",
        "dress", "skirt", "trousers", "jeans", "shorts",
        "shoes", "boots", "sneakers", "sandals", "socks",
        "cap", "hat", "helmet", "glasses", "sunglasses",
        "tie", "belt", "scarf", "gloves", "watch", "bracelet",
        "ring", "necklace", "wallet", "handbag", "backpack", "bag",
    ],
    "everyday": [
        "door", "window", "wall", "floor", "stairs", "ceiling",
        "table", "chair", "sofa", "bed", "shelf", "cabinet",
        "fan", "ceiling fan", "bulb", "lamp", "light", "switch",
        "plate", "bowl", "cup", "glass", "mug", "spoon", "fork", "knife",
        "bottle", "water bottle", "water gallon", "can",
        "plastic bag", "basket", "box", "bucket", "tray",
        "biscuit", "cookie", "candy", "bread", "food",
        "mobile phone", "laptop", "keyboard", "mouse", "TV", "monitor",
        "remote control", "book", "pen", "pencil", "notebook",
        "toothbrush", "soap", "towel", "mirror",
        "car", "bicycle", "motorcycle", "bus", "truck",
        "clock", "umbrella", "ball", "toy",
    ],
    "tiny": [
        "ant", "fly", "mosquito", "bee", "spider", "ladybug",
        "button", "coin", "key", "pin", "screw", "nail", "paperclip",
        "ring", "earring", "battery", "seed", "pill", "tablet",
        "bead", "needle", "tick", "flea", "crumb",
    ],
}
# Combined "all" set
_all, _seen = [], set()
for _s in WORLD_SETS.values():
    for _c in _s:
        if _c not in _seen:
            _all.append(_c)
            _seen.add(_c)
WORLD_SETS["all"] = _all
WORLD_SET_ORDER = ["all", "body", "clothing", "everyday", "tiny"]

# 17 body keypoint names (COCO pose standard)
KEYPOINT_NAMES = [
    "nose", "left eye", "right eye", "left ear", "right ear",
    "left shoulder", "right shoulder", "left elbow", "right elbow",
    "left wrist", "right wrist", "left hip", "right hip",
    "left knee", "right knee", "left ankle", "right ankle",
]

# Skeleton connections for drawing limbs
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # face
    (5, 6),                                     # shoulders
    (5, 7), (7, 9),                             # left arm
    (6, 8), (8, 10),                            # right arm
    (5, 11), (6, 12), (11, 12),                 # torso
    (11, 13), (13, 15),                         # left leg
    (12, 14), (14, 16),                         # right leg
]

# ─── Colour palette ───────────────────────────────────────────────────────────
_PAL = [
    (255,  56,  56), (255, 157,  51), (255, 212,  51), ( 51, 255, 255),
    ( 51, 153, 255), (153,  51, 255), (255,  51, 255), ( 51, 255, 153),
    (255, 128,   0), (  0, 255, 128), (128,   0, 255), (  0, 128, 255),
    (255,   0, 128), (128, 255,   0), (  0, 200, 255), (255, 255,   0),
    (200, 100, 255), (100, 255, 200), (255, 100, 100), (100, 100, 255),
]

def cls_color(name: str) -> tuple:
    return _PAL[hash(name) % len(_PAL)]

KPT_COLOR  = (0, 255, 128)       # body keypoint dot colour
LIMB_COLOR = (50, 180, 255)      # skeleton limb colour


# ─── Frame slicer (SAHI) ─────────────────────────────────────────────────────

def slice_frame(frame, sh, sw, oh, ow):
    H, W = frame.shape[:2]
    step_h = max(1, int(sh * (1 - oh)))
    step_w = max(1, int(sw * (1 - ow)))
    y = 0
    while y < H:
        x = 0
        while x < W:
            y2 = min(y + sh, H)
            x2 = min(x + sw, W)
            yield frame[y:y2, x:x2], x, y
            if x2 >= W:
                break
            x = (x + step_w) if (x + step_w + sw <= W) else (W - sw)
        if y2 >= H:
            break
        y = (y + step_h) if (y + step_h + sh <= H) else (H - sh)


def nms_merge(raw, iou_thr=0.45):
    if not raw:
        return []
    boxes  = np.array([[d[0], d[1], d[2] - d[0], d[3] - d[1]] for d in raw], np.float32)
    scores = np.array([d[4] for d in raw], np.float32)
    idxs   = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.0, iou_thr)
    if idxs is None or len(idxs) == 0:
        return []
    kept = idxs.flatten() if hasattr(idxs, "flatten") else [i[0] for i in idxs]
    return [raw[i] for i in kept]


# ─── Object detection inference ──────────────────────────────────────────────

def _extract_boxes(results):
    dets = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            dets.append({
                "bbox": (x1, y1, x2, y2),
                "conf": float(box.conf[0]),
                "cls_name": r.names[cls_id],
            })
    return dets


def infer_obj_standard(model, frame, conf, imgsz, device):
    return _extract_boxes(
        model.predict(source=frame, imgsz=imgsz, conf=conf,
                      device=device, verbose=False)
    )


def infer_obj_sahi(model, frame, conf, imgsz, device, sh, sw, oh, ow):
    raw = []
    for patch, ox, oy in slice_frame(frame, sh, sw, oh, ow):
        if patch.shape[0] < 16 or patch.shape[1] < 16:
            continue
        for r in model.predict(source=patch, imgsz=imgsz, conf=conf,
                               device=device, verbose=False):
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                raw.append([x1+ox, y1+oy, x2+ox, y2+oy,
                             float(box.conf[0]), r.names[int(box.cls[0])]])
    merged = nms_merge(raw)
    return [{"bbox": (d[0], d[1], d[2], d[3]),
             "conf": d[4], "cls_name": d[5]} for d in merged]


# ─── Pose inference ──────────────────────────────────────────────────────────

def infer_pose(pose_model, frame, conf, imgsz, device):
    """
    Run pose estimation. Returns list of persons, each with:
      bbox  : (x1,y1,x2,y2)
      conf  : float
      kpts  : list of (x, y, visibility) for each of 17 keypoints
    """
    results = pose_model.predict(source=frame, imgsz=imgsz, conf=conf,
                                 device=device, verbose=False)
    persons = []
    for r in results:
        if r.keypoints is None:
            continue
        for i, box in enumerate(r.boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            kpts_data = r.keypoints.data[i].cpu().numpy()  # shape (17, 3)
            kpts = [(float(k[0]), float(k[1]), float(k[2])) for k in kpts_data]
            persons.append({
                "bbox": (x1, y1, x2, y2),
                "conf": float(box.conf[0]),
                "kpts": kpts,
            })
    return persons


# ─── Drawing ──────────────────────────────────────────────────────────────────

def draw_obj(frame, det, show_info):
    x1, y1, x2, y2 = map(int, det["bbox"])
    name  = det["cls_name"]
    conf  = det["conf"]
    color = cls_color(name)
    area  = max(0, (x2-x1)*(y2-y1))
    thick = 1 if area < 900 else 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

    if show_info:
        label = f"{name} {conf:.2f}"
        fs    = 0.38 if area < 400 else 0.44
        font  = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, fs, 1)
        pad = 3
        lx = max(0, x1)
        ly = max(th + pad*2, y1)
        cv2.rectangle(frame, (lx, ly - th - pad*2),
                      (lx + tw + pad*2, ly), color, -1)
        cv2.putText(frame, label, (lx + pad, ly - pad),
                    font, fs, (15, 15, 15), 1, cv2.LINE_AA)


def draw_pose(frame, person, show_info, conf_thr=0.3):
    kpts = person["kpts"]
    H, W = frame.shape[:2]

    # Draw skeleton limbs
    for a_idx, b_idx in SKELETON:
        xa, ya, va = kpts[a_idx]
        xb, yb, vb = kpts[b_idx]
        if va > conf_thr and vb > conf_thr:
            cv2.line(frame,
                     (int(xa), int(ya)), (int(xb), int(yb)),
                     LIMB_COLOR, 2, cv2.LINE_AA)

    # Draw keypoints + labels
    for idx, (x, y, vis) in enumerate(kpts):
        if vis < conf_thr:
            continue
        ix, iy = int(x), int(y)
        if not (0 <= ix < W and 0 <= iy < H):
            continue
        cv2.circle(frame, (ix, iy), 4, KPT_COLOR, -1)
        cv2.circle(frame, (ix, iy), 4, (255, 255, 255), 1)

        if show_info:
            label = KEYPOINT_NAMES[idx]
            font  = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, label, (ix + 5, iy - 4),
                        font, 0.30, (220, 255, 220), 1, cv2.LINE_AA)

    # Person bounding box (subtle)
    x1, y1, x2, y2 = map(int, person["bbox"])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 255, 150), 1)


HELP_TEXT = [
    "  Q/ESC  → Quit",
    "  S      → Save screenshot",
    "  P      → Pause / Resume",
    "  M      → Toggle SAHI / Standard",
    "  Z      → Zoom  1x / 2x / 4x",
    "  O      → Toggle object detection",
    "  B      → Toggle body pose",
    "  + / -  → Confidence threshold",
    "  I      → Toggle labels",
    "  H      → This help",
]


def draw_hud(frame, fps, conf, paused, sahi, show_info, show_help,
             n_obj, n_pose, show_obj, show_pose, model_name, zoom, W, H):
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (W, 122), (12, 12, 12), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

    font   = cv2.FONT_HERSHEY_SIMPLEX
    green  = (55, 230, 80)
    white  = (210, 210, 210)
    yellow = (40, 210, 240)
    cyan   = (230, 210, 50)
    red    = (55, 70, 240)
    grey   = (130, 130, 130)
    off    = (80, 80, 80)

    cv2.putText(frame, "UNIVERSAL OBJECT DETECTION", (12, 22),
                font, 0.65, green, 2, cv2.LINE_AA)

    cv2.putText(frame, f"Model: {Path(model_name).name}", (12, 42),
                font, 0.40, white, 1, cv2.LINE_AA)

    fps_c = green if fps >= 10 else yellow if fps >= 5 else red
    cv2.putText(frame, f"FPS: {fps:.1f}", (12, 62), font, 0.46, fps_c, 1, cv2.LINE_AA)

    mode_s = "SAHI-Sliced" if sahi else "Full-Frame"
    mode_c = cyan if sahi else yellow
    cv2.putText(frame, f"Mode: {mode_s}", (120, 62), font, 0.46, mode_c, 1, cv2.LINE_AA)

    obj_c  = white if show_obj  else off
    pose_c = (50, 255, 180) if show_pose else off
    cv2.putText(frame,
                f"Objects:{n_obj}  Pose:{n_pose}  Conf:{conf:.2f}  Zoom:{zoom}x",
                (12, 84), font, 0.42, white, 1, cv2.LINE_AA)

    layers = f"O:Objects[{'ON' if show_obj else 'OFF'}]  B:Pose[{'ON' if show_pose else 'OFF'}]"
    cv2.putText(frame, layers, (12, 104), font, 0.40, white, 1, cv2.LINE_AA)

    cv2.putText(frame, "H:Help  M:SAHI  Z:Zoom  S:Save  +/-:Conf  Q:Quit",
                (12, 122), font, 0.35, grey, 1, cv2.LINE_AA)

    if paused:
        cv2.putText(frame, "[ PAUSED ]", (W//2-90, H//2),
                    font, 1.5, (0, 70, 255), 3, cv2.LINE_AA)

    if show_help:
        hx, hy = 12, 140
        cv2.rectangle(frame, (hx-4, hy-16),
                      (370, hy + len(HELP_TEXT)*18 + 4), (18, 18, 18), -1)
        for i, line in enumerate(HELP_TEXT):
            cv2.putText(frame, line, (hx, hy + i*18),
                        font, 0.38, (170, 220, 170), 1, cv2.LINE_AA)

    res = f"{W}x{H}"
    (tw, _), _ = cv2.getTextSize(res, font, 0.36, 1)
    cv2.putText(frame, res, (W-tw-6, H-6), font, 0.36, (100,100,100), 1, cv2.LINE_AA)


def zoom_frame(frame, lvl):
    if lvl == 1:
        return frame
    H, W = frame.shape[:2]
    hw = W // (2 * lvl)
    hh = H // (2 * lvl)
    cx, cy = W // 2, H // 2
    x1 = max(0, cx - hw);  x2 = min(W, cx + hw)
    y1 = max(0, cy - hh);  y2 = min(H, cy + hh)
    return cv2.resize(frame[y1:y2, x1:x2], (W, H), interpolation=cv2.INTER_LINEAR)


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(source, obj_model_path, pose_model_path, conf, imgsz, device,
        sahi_mode, slice_size, overlap, save_dir,
        enable_obj, enable_pose, use_world, world_model_path):

    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    # ── Mode selection ────────────────────────────────────────────────
    if use_world:
        print("=" * 65)
        print("  UNIVERSAL LIVE DETECTION — YOLO-World Mode")
        print("  200+ open-vocabulary classes (face, hand, door, ant...)")
        print("=" * 65)
        print(f"\n  Loading YOLO-World model: {world_model_path}")
        print("  (First run downloads ~130MB model + ~338MB CLIP...)")
        print("  TIP: Pre-download CLIP reliably with: python download_clip.py\n")
        try:
            world_model = YOLO(world_model_path)
            world_set_idx = 0
            cur_classes = WORLD_SETS[WORLD_SET_ORDER[world_set_idx]]
            world_model.set_classes(cur_classes)
            print(f"  ✓ YOLO-World ready. Class set: '{WORLD_SET_ORDER[world_set_idx]}'"
                  f" ({len(cur_classes)} classes)")
        except Exception as e:
            print(f"\n  [ERROR] YOLO-World failed to load: {e}")
            print("\n  The CLIP model download likely failed/incomplete.")
            print("  Run this to download with auto-retry:")
            print("    python download_clip.py")
            print("\n  Then retry:")
            print("    python src/live_camera.py --world --sahi")
            return
        obj_model = pose_model = None
    else:
        print("=" * 65)
        print("  UNIVERSAL LIVE DETECTION — Objects (80) + Pose (17 keypoints)")
        print("=" * 65)
        world_model = None
        world_set_idx = 0

        obj_model = pose_model = None
        if enable_obj:
            print(f"\n[1/2] Loading object model: {obj_model_path}")
            obj_model = YOLO(obj_model_path)
            print("      ✓ Object model ready.")
        if enable_pose:
            print(f"\n[2/2] Loading pose model: {pose_model_path}")
            pose_model = YOLO(pose_model_path)
            print("      ✓ Pose model ready.")

    # ── Camera ────────────────────────────────────────────────────────
    print(f"\n[INFO] Opening camera {source}...")
    # On Windows, try DirectShow backend first if default backend stalls
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(source)
    if not cap.isOpened():
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera (source={source}). Try --source 1")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    # Test-read first frame to ensure camera feed is actually producing frames
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print("[WARN] DirectShow frame read empty, trying default camera backend...")
        cap.release()
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ret, test_frame = cap.read()

    if not ret or test_frame is None:
        print(f"[ERROR] Camera {source} opened but failed to capture video frames.")
        print("  - Check if another application (Zoom, Teams, Skype, Browser) is using your camera.")
        print("  - Try running with a different camera index: python src/live_camera.py --world --source 1")
        cap.release()
        return

    W = test_frame.shape[1]
    H = test_frame.shape[0]
    print(f"[OK] Camera feed live at {W}x{H}")

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    win = ("YOLO-World Detection  |  C=Classes  M=SAHI  Q=Quit"
           if use_world else
           "Universal Object Detection  |  O=Objects  B=Pose  M=SAHI  Q=Quit")
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    win_w = min(W, 1280)
    win_h = min(H, 720)
    cv2.resizeWindow(win, win_w, win_h)

    # Center window on the screen
    try:
        if sys.platform.startswith("win"):
            import ctypes
            scr_w = ctypes.windll.user32.GetSystemMetrics(0)
            scr_h = ctypes.windll.user32.GetSystemMetrics(1)
            pos_x = max(0, (scr_w - win_w) // 2)
            pos_y = max(0, (scr_h - win_h) // 2)
            cv2.moveWindow(win, pos_x, pos_y)
    except Exception:
        pass

    # Show initial loading frame so window appears immediately on screen
    init_disp = test_frame.copy()
    cv2.putText(init_disp, "INITIALIZING LIVE CAMERA FEED...", (W // 4, H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 128), 2, cv2.LINE_AA)
    cv2.imshow(win, init_disp)
    cv2.waitKey(1)

    # State
    paused     = False
    show_info  = True
    show_help  = False
    use_sahi   = sahi_mode
    show_obj   = enable_obj
    show_pose  = enable_pose
    conf_val   = conf
    zoom_opts  = [1, 2, 4]
    zoom_idx   = 0
    fps        = 0.0
    frame_cnt  = 0
    t_fps      = time.time()
    last_disp  = None

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q'), 27):    break
        elif key in (ord('p'), ord('P')):       paused = not paused
        elif key in (ord('m'), ord('M')):
            use_sahi = not use_sahi
            print(f"  Mode → {'SAHI-Sliced' if use_sahi else 'Full-Frame'}")
        elif key in (ord('z'), ord('Z')):
            zoom_idx = (zoom_idx + 1) % len(zoom_opts)
            print(f"  Zoom → {zoom_opts[zoom_idx]}x")
        elif key in (ord('c'), ord('C')) and use_world:
            world_set_idx = (world_set_idx + 1) % len(WORLD_SET_ORDER)
            set_name = WORLD_SET_ORDER[world_set_idx]
            cur_classes = WORLD_SETS[set_name]
            world_model.set_classes(cur_classes)
            print(f"  Classes → '{set_name}' ({len(cur_classes)} classes)")
        elif key in (ord('o'), ord('O')) and obj_model:
            show_obj = not show_obj
            print(f"  Objects → {'ON' if show_obj else 'OFF'}")
        elif key in (ord('b'), ord('B')) and pose_model:
            show_pose = not show_pose
            print(f"  Body Pose → {'ON' if show_pose else 'OFF'}")
        elif key in (ord('i'), ord('I')):       show_info = not show_info
        elif key in (ord('h'), ord('H')):       show_help = not show_help
        elif key in (ord('s'), ord('S')):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            sp = Path(save_dir) / f"det_{ts}.jpg"
            if last_disp is not None:
                cv2.imwrite(str(sp), last_disp)
                print(f"  📸 Saved: {sp}")
        elif key in (ord('+'), ord('=')):
            conf_val = min(0.95, round(conf_val + 0.05, 2))
            print(f"  Conf → {conf_val:.2f}")
        elif key == ord('-'):
            conf_val = max(0.05, round(conf_val - 0.05, 2))
            print(f"  Conf → {conf_val:.2f}")

        if paused and last_disp is not None:
            cv2.imshow(win, last_disp)
            continue

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        zoom_lvl  = zoom_opts[zoom_idx]
        inf_frame = zoom_frame(frame, zoom_lvl)

        # ── Inference ─────────────────────────────────────────────────
        obj_dets     = []
        pose_persons = []

        if use_world and world_model:
            if use_sahi:
                obj_dets = infer_obj_sahi(world_model, inf_frame, conf_val,
                                          imgsz, device, slice_size, slice_size,
                                          overlap, overlap)
            else:
                obj_dets = infer_obj_standard(world_model, inf_frame,
                                              conf_val, imgsz, device)
        else:
            if show_obj and obj_model:
                if use_sahi:
                    obj_dets = infer_obj_sahi(obj_model, inf_frame, conf_val,
                                             imgsz, device, slice_size, slice_size,
                                             overlap, overlap)
                else:
                    obj_dets = infer_obj_standard(obj_model, inf_frame,
                                                  conf_val, imgsz, device)
            if show_pose and pose_model:
                pose_persons = infer_pose(pose_model, inf_frame,
                                         conf_val, imgsz, device)

        # ── FPS ───────────────────────────────────────────────────────
        frame_cnt += 1
        if frame_cnt % 10 == 0:
            elapsed = time.time() - t_fps
            fps = 10 / elapsed if elapsed > 0 else 0
            t_fps = time.time()

        # ── Draw ──────────────────────────────────────────────────────
        display = inf_frame.copy()
        for person in pose_persons:
            draw_pose(display, person, show_info)
        for det in obj_dets:
            draw_obj(display, det, show_info)

        model_label = (world_model_path if use_world
                       else (obj_model_path if show_obj else pose_model_path))
        set_label   = WORLD_SET_ORDER[world_set_idx] if use_world else ""

        draw_hud(display, fps, conf_val, paused, use_sahi, show_info, show_help,
                 len(obj_dets), len(pose_persons),
                 show_obj or use_world, show_pose and not use_world,
                 model_label, zoom_lvl, W, H)

        # Show active class set in World mode
        if use_world:
            cv2.putText(display,
                        f"ClassSet: {set_label} ({len(cur_classes)} classes)   C=cycle",
                        (12, H - 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.40, (80, 220, 220), 1, cv2.LINE_AA)

        last_disp = display
        cv2.imshow(win, display)

    cap.release()
    cv2.destroyAllWindows()
    print("\n[OK] Session ended.")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Universal Live Object Detection — Objects + Pose + YOLO-World",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast Real-Time YOLO-World (Fluid 30 FPS):
  python src/live_camera.py --world

  # High accuracy SAHI mode (optimized for CPU):
  python src/live_camera.py --world --sahi

  # Standard COCO objects + 17 Body keypoints:
  python src/live_camera.py
        """,
    )
    parser.add_argument("--source",       type=int,   default=0)
    parser.add_argument("--world",        action="store_true",
                        help="Use YOLO-World open-vocabulary model (200+ classes)")
    parser.add_argument("--world-model",  type=str,   default="yolov8s-worldv2.pt",
                        help="YOLO-World model weights (default: yolov8s-worldv2.pt for fast real-time CPU inference)")
    parser.add_argument("--obj-model",    type=str,   default="yolov8s.pt",
                        help="Object model (default: yolov8s.pt for real-time CPU performance)")
    parser.add_argument("--pose-model",   type=str,   default="yolov8n-pose.pt")
    parser.add_argument("--conf",         type=float, default=0.25)
    parser.add_argument("--imgsz",        type=int,   default=480,
                        help="Inference image resolution (default: 480 for fast real-time video)")
    parser.add_argument("--device",       type=str,   default=None)
    parser.add_argument("--sahi",         action="store_true")
    parser.add_argument("--slice",        type=int,   default=480,
                        help="Slice size for SAHI mode (default: 480)")
    parser.add_argument("--overlap",      type=float, default=0.20)
    parser.add_argument("--save-dir",     type=str,   default="output/screenshots")
    parser.add_argument("--no-obj",       action="store_true")
    parser.add_argument("--no-pose",      action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        source=args.source,
        obj_model_path=args.obj_model,
        pose_model_path=args.pose_model,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        sahi_mode=args.sahi,
        slice_size=args.slice,
        overlap=args.overlap,
        save_dir=args.save_dir,
        enable_obj=not args.no_obj,
        enable_pose=not args.no_pose,
        use_world=args.world,
        world_model_path=args.world_model,
    )
