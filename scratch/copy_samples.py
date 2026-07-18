import shutil
import os

src_dir = r"C:\Users\User\.gemini\antigravity-ide\brain\3c134752-e576-41bd-8ba6-0ef79cdad19b"
dst_dir = r"c:\Users\User\Desktop\StrataVision-Micro-Detector\output\samples"

os.makedirs(dst_dir, exist_ok=True)

mapping = {
    "media__1784387696388.jpg": "urban_street_detection.jpg",
    "media__1784387712701.jpg": "indoor_scene_detection.jpg",
    "media__1784387725862.jpg": "traffic_scene_detection.jpg"
}

for src_name, dst_name in mapping.items():
    src_path = os.path.join(src_dir, src_name)
    dst_path = os.path.join(dst_dir, dst_name)
    shutil.copyfile(src_path, dst_path)
    print(f"Copied {src_path} -> {dst_path}")

print("All sample images copied successfully!")
