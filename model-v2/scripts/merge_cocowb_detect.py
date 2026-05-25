"""Shrink COCO-WholeBody hand images (256px) + merge hand boxes into the detector
train set (Plan 6 Phase 3). COCO hand boxes are oversampled to ~15% of train images
so the in-the-wild hands actually influence hand recall. Val stays unchanged.

  cd model-v2 && PYTHONPATH=src .venv/bin/python scripts/merge_cocowb_detect.py

Writes:
  data/detect_small/coco_wholebody/...                shrunk COCO images (max_side 256)
  artifacts/detect/cocowb_hands_small.json            COCO detector records (scaled)
  artifacts/detect/train_small.json                   100DOH+WIDER + COCO×N (combined)
  artifacts/detect/train_small_base.json              backup of the original train set
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

SRC_ROOT = Path("data/landmark")        # full-res COCO images live here
SMALL_ROOT = Path("data/detect_small")
MAX_SIDE = 256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cocowb", default="artifacts/detect/cocowb_hands.json")
    ap.add_argument("--target-frac", type=float, default=0.15)
    args = ap.parse_args()

    recs = json.loads(Path(args.cocowb).read_text())
    small = []
    for r in recs:
        rel = r["image"]
        img = cv2.imread(str(SRC_ROOT / rel))
        if img is None:
            continue
        h, w = img.shape[:2]
        s = MAX_SIDE / max(h, w)
        out = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        dst = SMALL_ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), out, [cv2.IMWRITE_JPEG_QUALITY, 82])
        boxes = (np.asarray(r["boxes"], dtype=float) * s).tolist()
        small.append({"image": rel, "boxes": boxes, "labels": r["labels"]})

    Path("artifacts/detect/cocowb_hands_small.json").write_text(json.dumps(small))

    base = json.loads(Path("artifacts/detect/train_small.json").read_text())
    bak = Path("artifacts/detect/train_small_base.json")
    if not bak.exists():
        bak.write_text(json.dumps(base))
    n = len(base)
    k = max(1, round(args.target_frac * n / (len(small) * (1 - args.target_frac))))
    combined = base + small * k
    Path("artifacts/detect/train_small.json").write_text(json.dumps(combined))

    print(f"COCO detector images: {len(small)} | oversample ×{k} -> {len(small)*k}")
    print(f"combined detect train: {len(combined)} (base {n} + COCO {len(small)*k}, COCO {len(small)*k/len(combined):.0%})")
    print("base backup -> artifacts/detect/train_small_base.json")


if __name__ == "__main__":
    main()
