"""Build unified detector training manifests from 100DOH + WIDER FACE.

Outputs:
  model-v2/artifacts/detect/train.json  (100DOH train + WIDER train)
  model-v2/artifacts/detect/val.json    (100DOH val   + WIDER val)

Prints per-split statistics and a path-resolution sanity check.
"""
from __future__ import annotations

import json
import os
import random
import sys

# Resolve repo root so we can run this script from any cwd
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_V2 = os.path.dirname(_SCRIPT_DIR)  # model-v2/
_REPO_ROOT = os.path.dirname(_MODEL_V2)

# Ensure aslv2 is importable
sys.path.insert(0, os.path.join(_MODEL_V2, "src"))

from aslv2.detect.adapters import adapt_100doh, adapt_widerface, write_manifest


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_100DOH_DIR = os.path.join(_REPO_ROOT, "model-v2", "data", "detect", "100doh")
_100DOH_RAW = os.path.join(_100DOH_DIR, "raw")
_100DOH_FILE = os.path.join(_100DOH_DIR, "file")

_WIDER_DIR = os.path.join(_REPO_ROOT, "model-v2", "data", "detect", "widerface")
_WIDER_SPLIT = os.path.join(_WIDER_DIR, "wider_face_split")

_OUT_DIR = os.path.join(_MODEL_V2, "artifacts", "detect")

SPLITS = {
    "train": {
        "doh_json":    os.path.join(_100DOH_FILE, "train.json"),
        "doh_raw":     _100DOH_RAW,
        "wider_gt":    os.path.join(_WIDER_SPLIT, "wider_face_train_bbx_gt.txt"),
        "wider_imgs":  os.path.join(_WIDER_DIR, "WIDER_train", "images"),
        "out":         os.path.join(_OUT_DIR, "train.json"),
    },
    "val": {
        "doh_json":    os.path.join(_100DOH_FILE, "val.json"),
        "doh_raw":     _100DOH_RAW,
        "wider_gt":    os.path.join(_WIDER_SPLIT, "wider_face_val_bbx_gt.txt"),
        "wider_imgs":  os.path.join(_WIDER_DIR, "WIDER_val", "images"),
        "out":         os.path.join(_OUT_DIR, "val.json"),
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_boxes(records: list[dict], label: int) -> int:
    return sum(lbl == label for rec in records for lbl in rec["labels"])


def sanity_check_paths(records: list[dict], sample_n: int = 200) -> float:
    """Return fraction of sampled image paths that exist on disk."""
    if not records:
        return 0.0
    sample = random.sample(records, min(sample_n, len(records)))
    found = sum(1 for r in sample if os.path.isfile(r["image"]))
    return found / len(sample)


def manifest_size_mb(path: str) -> float:
    return os.path.getsize(path) / 1_000_000


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    random.seed(42)
    stats: dict[str, dict] = {}

    for split, cfg in SPLITS.items():
        print(f"\n{'='*60}")
        print(f"  Building {split} manifest")
        print(f"{'='*60}")

        print(f"  [100DOH]  {cfg['doh_json']}")
        doh_records = adapt_100doh(cfg["doh_json"], cfg["doh_raw"])
        print(f"            → {len(doh_records)} images, "
              f"{count_boxes(doh_records, 0)} hand boxes")

        print(f"  [WIDER]   {cfg['wider_gt']}")
        wider_records = adapt_widerface(cfg["wider_gt"], cfg["wider_imgs"])
        print(f"            → {len(wider_records)} images, "
              f"{count_boxes(wider_records, 1)} head boxes")

        all_records = doh_records + wider_records
        n_images = len(all_records)
        n_hands  = count_boxes(all_records, 0)
        n_heads  = count_boxes(all_records, 1)

        write_manifest(all_records, cfg["out"])
        mb = manifest_size_mb(cfg["out"])
        print(f"\n  Written → {cfg['out']}  ({mb:.1f} MB)")
        print(f"  TOTAL:  {n_images} images | {n_hands} hand boxes | {n_heads} head boxes")

        resolution = sanity_check_paths(all_records, sample_n=300)
        print(f"  Path resolution: {resolution*100:.1f}% of sampled images exist on disk")
        if resolution < 0.99:
            print(f"  WARNING: <99% paths resolved — check raw_dir / images_dir settings")

        stats[split] = {
            "images":     n_images,
            "hand_boxes": n_hands,
            "head_boxes": n_heads,
            "size_mb":    round(mb, 2),
            "path_resolution_pct": round(resolution * 100, 1),
        }

    # Write stats sidecar
    stats_path = os.path.join(_OUT_DIR, "manifest_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Stats written → {stats_path}")

    # Warn if any manifest is >50 MB
    for split, cfg in SPLITS.items():
        mb = manifest_size_mb(cfg["out"])
        if mb > 50:
            print(f"\n  NOTICE: {split}.json is {mb:.0f} MB (>50 MB) "
                  "— consider gitignoring it and committing only manifest_stats.json")


if __name__ == "__main__":
    main()
