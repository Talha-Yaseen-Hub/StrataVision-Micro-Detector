import os
import shutil

# Source directory where attached images are stored
src_dir = r"C:\Users\User\.gemini\antigravity-ide\brain\3c134752-e576-41bd-8ba6-0ef79cdad19b"
dst_dir = r"output\samples"

os.makedirs(dst_dir, exist_ok=True)

mapping = {
    "media__1784387696388.jpg": "urban_street_detection.jpg",
    "media__1784387712701.jpg": "indoor_scene_detection.jpg",
    "media__1784387725862.jpg": "traffic_scene_detection.jpg"
}

print("Copying sample detection images to output/samples/...")

for src_name, dst_name in mapping.items():
    src_path = os.path.join(src_dir, src_name)
    dst_path = os.path.join(dst_dir, dst_name)
    if os.path.exists(src_path):
        shutil.copyfile(src_path, dst_path)
        print(f"✓ Copied: {dst_name}")
    else:
        print(f"✗ Source not found: {src_path}")

print("\nDone! Now run:\n  git add output/samples/\n  git add README.md LICENSE\n  git commit -m \"Add sample detection images\"\n  git push")
