"""
Download CLIP ViT-B/32 model with auto-retry and resume support.
Run this ONCE before using YOLO-World to ensure the download completes.

Usage:
  python download_clip.py
"""

import os
import sys
import time
import hashlib
from pathlib import Path

CLIP_URL  = "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba3a6b/ViT-B-32.pt"
CLIP_SHA  = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba3a6b"
CLIP_SIZE = 353_976_320   # bytes (~338 MB)
SAVE_DIR  = Path.home() / ".cache" / "clip"
SAVE_PATH = SAVE_DIR / "ViT-B-32.pt"

MAX_RETRIES   = 20
CHUNK_SIZE    = 1024 * 1024   # 1 MB per chunk
RETRY_WAIT    = 5             # seconds between retries


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_with_resume(url: str, dest: Path, max_retries: int = MAX_RETRIES):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    import urllib.request
    import urllib.error

    for attempt in range(1, max_retries + 1):
        existing = dest.stat().st_size if dest.exists() else 0

        headers = {}
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"

        req = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                total_from_header = resp.headers.get("Content-Range", "")
                if total_from_header:
                    total = int(total_from_header.split("/")[-1])
                else:
                    total = int(resp.headers.get("Content-Length", CLIP_SIZE))

                mode = "ab" if existing > 0 else "wb"
                downloaded = existing

                print(f"\n  Attempt {attempt}/{max_retries}  "
                      f"({'resuming' if existing else 'starting'} "
                      f"from {existing // 1024 // 1024} MB)")

                with open(dest, mode) as f:
                    t0 = time.time()
                    while True:
                        chunk = resp.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        elapsed = max(time.time() - t0, 0.001)
                        speed = (downloaded - existing) / elapsed / 1024 / 1024
                        pct = downloaded / max(total, 1) * 100
                        bar_w = 30
                        filled = int(bar_w * pct / 100)
                        bar = "█" * filled + "░" * (bar_w - filled)
                        print(f"\r  [{bar}] {pct:5.1f}%  "
                              f"{downloaded//1024//1024} / {total//1024//1024} MB  "
                              f"{speed:.1f} MB/s    ",
                              end="", flush=True)

            print()  # newline after progress bar

            # Verify file is complete
            if dest.stat().st_size < CLIP_SIZE * 0.98:
                print(f"  [WARN] File seems incomplete "
                      f"({dest.stat().st_size // 1024 // 1024} MB). Retrying...")
                time.sleep(RETRY_WAIT)
                continue

            print(f"  [OK] Download complete: {dest}")
            return True

        except Exception as e:
            print(f"\n  [RETRY] Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                print(f"  Waiting {RETRY_WAIT}s before retry...")
                time.sleep(RETRY_WAIT)
            else:
                print("  [ERROR] All retries exhausted.")
                return False

    return False


def main():
    print("=" * 60)
    print("  CLIP ViT-B/32 Model Downloader")
    print("=" * 60)
    print(f"  Save location : {SAVE_PATH}")
    print(f"  File size     : ~338 MB")
    print(f"  Max retries   : {MAX_RETRIES}")

    if SAVE_PATH.exists():
        size_mb = SAVE_PATH.stat().st_size / 1024 / 1024
        if SAVE_PATH.stat().st_size >= CLIP_SIZE * 0.98:
            print(f"\n  [OK] CLIP model already downloaded ({size_mb:.1f} MB).")
            print("  You can now run:")
            print("    python src/live_camera.py --world --sahi")
            return
        else:
            print(f"\n  [INFO] Partial download found ({size_mb:.1f} MB). Resuming...")
    else:
        print("\n  [INFO] Starting fresh download...")

    ok = download_with_resume(CLIP_URL, SAVE_PATH)

    if ok:
        print("\n" + "=" * 60)
        print("  ✅  CLIP model ready!")
        print("  Now run YOLO-World:")
        print("    python src/live_camera.py --world --sahi")
        print("=" * 60)
    else:
        print("\n  ❌  Download failed. Check your internet and try again:")
        print("       python download_clip.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
