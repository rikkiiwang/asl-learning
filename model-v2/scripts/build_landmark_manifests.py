"""Build train/val manifests for FreiHAND landmark training.

Run this AFTER downloading FreiHAND v2 into data/landmark/freihand/.
See DATA.md for download instructions.

FreiHAND v2 layout expected under <data_root>/freihand/:
    training/rgb/%08d.jpg      (130,240 images total)
    training_xyz.json          (32,560 × 21 × 3 camera-frame 3-D joints)
    training_K.json            (32,560 × 3×3 intrinsics)

Split strategy:
  - Base indices 0 .. 29,303  (90%) → train  (all 4 bg variants included)
  - Base indices 29,304 .. 32,559 (10%) → val   (all 4 bg variants included)

Total after split:
  train: 29,304 × 4 = 117,216 images
  val:     3,256 × 4 =  13,024 images

Usage (from model-v2/ directory, after FreiHAND is downloaded):
    .venv/bin/python scripts/build_landmark_manifests.py
    .venv/bin/python scripts/build_landmark_manifests.py \\
        --freihand-root data/landmark/freihand \\
        --data-root data/landmark \\
        --train-out artifacts/landmark/train.json \\
        --val-out   artifacts/landmark/val.json
"""
from __future__ import annotations

# NOTE: Do NOT run this script until FreiHAND is downloaded.
# The script will raise FileNotFoundError if training_xyz.json is missing.

import argparse
import json
import os

from aslv2.landmark.freihand import adapt_freihand, _FREIHAND_N_BASE, _FREIHAND_N_VARIANTS

TRAIN_FRAC = 0.9   # 90% of base indices for train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--freihand-root",
        default="data/landmark/freihand",
        help="Path to FreiHAND root dir (contains training_xyz.json etc.)",
    )
    ap.add_argument(
        "--data-root",
        default="data/landmark",
        help="Root passed to KpDataset data_root; image paths made relative to this.",
    )
    ap.add_argument(
        "--train-out",
        default="artifacts/landmark/train.json",
        help="Output path for training manifest.",
    )
    ap.add_argument(
        "--val-out",
        default="artifacts/landmark/val.json",
        help="Output path for validation manifest.",
    )
    args = ap.parse_args()

    # Verify FreiHAND is present before proceeding
    xyz_path = os.path.join(args.freihand_root, "training_xyz.json")
    if not os.path.exists(xyz_path):
        raise FileNotFoundError(
            f"FreiHAND not found at {args.freihand_root!r}. "
            "Download FreiHAND v2 first — see DATA.md."
        )

    n_base       = _FREIHAND_N_BASE            # 32,560
    n_train_base = int(n_base * TRAIN_FRAC)    # 29,304
    n_val_base   = n_base - n_train_base       # 3,256

    n_train_images = n_train_base * _FREIHAND_N_VARIANTS
    n_val_images   = n_val_base   * _FREIHAND_N_VARIANTS

    print(f"FreiHAND root : {args.freihand_root}")
    print(f"n_base        : {n_base}")
    print(f"train base    : {n_train_base}  ({n_train_images} images with 4 bg variants)")
    print(f"val base      :  {n_val_base}  ({n_val_images} images with 4 bg variants)")

    # --- Build train records: images 0 .. n_train_images-1 ---
    # Image index i uses label i % n_base; the first n_train_base base indices
    # are covered by images 0 .. (n_train_base*4 - 1) because FreiHAND orders
    # images as: base_0_bg0, base_0_bg1, ... wait — actually the ordering is:
    #   images 0..32559 = base index 0..32559 (bg version 0)
    #   images 32560..65119 = base index 0..32559 (bg version 1)
    #   etc.
    # So label_idx = i % 32560 correctly wraps.
    # For a clean base-index split we only include images whose label_idx < n_train_base.
    print("\nBuilding train manifest …")
    all_records = adapt_freihand(args.freihand_root, data_root=args.data_root)
    n_base_actual = len(all_records) // _FREIHAND_N_VARIANTS

    train_records = [r for i, r in enumerate(all_records) if (i % n_base_actual) < n_train_base]
    val_records   = [r for i, r in enumerate(all_records) if (i % n_base_actual) >= n_train_base]

    os.makedirs(os.path.dirname(args.train_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.val_out),   exist_ok=True)

    with open(args.train_out, "w") as f:
        json.dump(train_records, f)
    with open(args.val_out, "w") as f:
        json.dump(val_records, f)

    print(f"train: {len(train_records)} records → {args.train_out}")
    print(f"val  : {len(val_records)} records → {args.val_out}")
    print("\nDone. Next: python scripts/shrink_landmark_for_colab.py")


if __name__ == "__main__":
    main()
