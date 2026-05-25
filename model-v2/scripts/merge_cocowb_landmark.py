"""Shrink COCO-WholeBody hand images + merge into the FreiHAND landmark train set.

COCO-WholeBody val gives ~1.3k clean hands vs FreiHAND's 117k, so we OVERSAMPLE
COCO so it's a meaningful fraction (~15%) of the combined train set — otherwise it
gets drowned out. Strong framing augmentation (KpDataset) keeps the duplicates from
being memorised. Val stays pure FreiHAND for a comparable PCK.

  cd model-v2 && PYTHONPATH=src .venv/bin/python scripts/merge_cocowb_landmark.py

Writes:
  data/landmark_small/coco_wholebody/...           shrunk COCO images
  artifacts/landmark/cocowb_landmark_small.json     COCO records (scaled kpts)
  artifacts/landmark/train_small.json               FreiHAND + COCO×N  (combined)
  artifacts/landmark/train_small_freihand.json      backup of the FreiHAND-only set
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

DATA_ROOT = Path("data/landmark")
SMALL_ROOT = Path("data/landmark_small")
MAX_SIDE = 128


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cocowb", default="artifacts/landmark/cocowb_landmark.json")
    ap.add_argument("--target-frac", type=float, default=0.15,
                    help="desired COCO fraction of the combined train set")
    args = ap.parse_args()

    recs = json.loads(Path(args.cocowb).read_text())
    small_recs = []
    cache = {}
    for r in recs:
        rel = r["image"]
        src = DATA_ROOT / rel
        dst = SMALL_ROOT / rel
        if rel not in cache:
            img = cv2.imread(str(src))
            if img is None:
                continue
            h, w = img.shape[:2]
            scale = MAX_SIDE / max(h, w)
            out = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), out, [cv2.IMWRITE_JPEG_QUALITY, 85])
            cache[rel] = scale
        s = cache[rel]
        kp = (np.asarray(r["keypoints"], dtype=float) * s).tolist()
        box = (np.asarray(r["box"], dtype=float) * s).tolist()
        small_recs.append({"image": rel, "box": box, "keypoints": kp})

    Path("artifacts/landmark/cocowb_landmark_small.json").write_text(json.dumps(small_recs))

    # merge (oversample COCO to target fraction)
    fh = json.loads(Path("artifacts/landmark/train_small.json").read_text())
    # back up the FreiHAND-only set once
    bak = Path("artifacts/landmark/train_small_freihand.json")
    if not bak.exists():
        bak.write_text(json.dumps(fh))
    n_fh = len(fh)
    # n_coco*k / (n_fh + n_coco*k) = frac  ->  k = frac*n_fh / (n_coco*(1-frac))
    k = max(1, round(args.target_frac * n_fh / (len(small_recs) * (1 - args.target_frac))))
    combined = fh + small_recs * k
    Path("artifacts/landmark/train_small.json").write_text(json.dumps(combined))

    frac = len(small_recs) * k / len(combined)
    print(f"COCO unique hands: {len(small_recs)} | oversample ×{k} -> {len(small_recs)*k}")
    print(f"combined train: {len(combined)} (FreiHAND {n_fh} + COCO {len(small_recs)*k}, COCO {frac:.0%})")
    print("FreiHAND-only backup -> artifacts/landmark/train_small_freihand.json")


if __name__ == "__main__":
    main()
