"""Shrink detector training images for fast Colab upload.

Reads each entry in artifacts/detect/{train,val}.json, downscales the image
so the longer side == MAX_SIDE (default 256), writes a JPEG to
data/detect_small/<rel_path>, and emits new manifests at
artifacts/detect/{train,val}_small.json with the same relative paths and
scaled boxes.

Usage (from model-v2/ directory):
    python scripts/shrink_for_colab.py [--max-side 256] [--quality 82]
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is somehow unavailable
    def tqdm(it, **kwargs):  # type: ignore[misc]
        total = kwargs.get("total", "?")
        desc = kwargs.get("desc", "")
        print(f"{desc}: processing {total} items …")
        return it

from aslv2.detect.shrink import resize_keep_aspect

MAX_SIDE_DEFAULT = 256
QUALITY_DEFAULT = 82

SPLITS = [
    ("artifacts/detect/train.json", "artifacts/detect/train_small.json"),
    ("artifacts/detect/val.json", "artifacts/detect/val_small.json"),
]


def process_split(
    manifest_in: str,
    manifest_out: str,
    data_root: str,
    data_root_small: str,
    max_side: int,
    quality: int,
) -> int:
    """Process one split.  Returns the number of images written."""
    with open(manifest_in) as f:
        entries = json.load(f)

    out_entries: list[dict] = []
    written = 0
    split_name = Path(manifest_in).stem

    for entry in tqdm(entries, desc=split_name, total=len(entries), unit="img"):
        rel = entry["image"]
        src_path = os.path.join(data_root, rel)
        dst_path = os.path.join(data_root_small, rel)

        img = cv2.imread(src_path)
        if img is None:
            warnings.warn(f"[skip] unreadable: {src_path}")
            continue

        boxes = np.array(entry["boxes"], dtype=np.float64)
        img2, boxes2, _scale = resize_keep_aspect(img, boxes, max_side=max_side)

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        ok = cv2.imwrite(dst_path, img2, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            warnings.warn(f"[skip] write failed: {dst_path}")
            continue

        out_entries.append(
            {
                "image": rel,
                "boxes": boxes2.tolist(),
                "labels": entry["labels"],
            }
        )
        written += 1

    os.makedirs(os.path.dirname(manifest_out), exist_ok=True)
    with open(manifest_out, "w") as f:
        json.dump(out_entries, f)

    return written


def dir_size_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for fname in files:
            try:
                total += os.path.getsize(os.path.join(root, fname))
            except OSError:
                pass
    return total


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-side", type=int, default=MAX_SIDE_DEFAULT)
    ap.add_argument("--quality", type=int, default=QUALITY_DEFAULT)
    ap.add_argument("--data-root", default="data/detect")
    ap.add_argument("--data-root-small", default="data/detect_small")
    args = ap.parse_args()

    total_written = 0
    for manifest_in, manifest_out in SPLITS:
        n = process_split(
            manifest_in=manifest_in,
            manifest_out=manifest_out,
            data_root=args.data_root,
            data_root_small=args.data_root_small,
            max_side=args.max_side,
            quality=args.quality,
        )
        print(f"  {Path(manifest_in).stem}: {n} images -> {manifest_out}")
        total_written += n

    total_bytes = dir_size_bytes(args.data_root_small)
    print(f"\nTotal images written : {total_written}")
    print(f"data/detect_small    : {human(total_bytes)}")


if __name__ == "__main__":
    main()
