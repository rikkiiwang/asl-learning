"""FreiHAND v2 adapter for the ASL landmark pipeline.

FreiHAND v2 dataset layout
---------------------------
  <root>/
    training/rgb/%08d.jpg       224×224, 130,240 images total
                                 = 32,560 base images × 4 background versions
    training_xyz.json            32,560 entries × 21 × 3  (camera-frame 3-D joints)
    training_K.json              32,560 entries × 3×3 intrinsics

Label index for image i: label_idx = i % 32560 (== i % n_base)
The hand fills the full frame, so box = [0, 0, 224, 224].

Usage
-----
>>> records = adapt_freihand("data/landmark/freihand", data_root="data/landmark")
>>> import json; json.dump(records, open("artifacts/landmark/train.json", "w"))
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

# FreiHAND constants
_FREIHAND_IMAGE_SIZE = 224
_FREIHAND_N_BASE     = 32_560   # base images (without background augmentation)
_FREIHAND_N_VARIANTS = 4        # background versions per base image
_FREIHAND_N_TOTAL    = _FREIHAND_N_BASE * _FREIHAND_N_VARIANTS   # 130,240


# ---------------------------------------------------------------------------
# Pure projection function (TDD target)
# ---------------------------------------------------------------------------

def project_2d(xyz: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project 3-D camera-frame points to 2-D pixel coordinates.

    Args:
        xyz: (21, 3) float array of camera-frame 3-D joints.
        K:   (3, 3) float camera intrinsics matrix.

    Returns:
        uv:  (21, 2) float32 array of pixel coordinates [u, v].

    Projection:
        uv_h = (K @ xyz.T).T           # (21, 3) homogeneous
        uv   = uv_h[:, :2] / uv_h[:, 2:3]   # perspective divide
    """
    xyz  = np.asarray(xyz,  dtype=np.float64)
    K    = np.asarray(K,    dtype=np.float64)

    uv_h = (K @ xyz.T).T          # (21, 3) homogeneous pixel coords
    uv   = uv_h[:, :2] / uv_h[:, 2:3]   # (21, 2) pixel coords

    return uv.astype(np.float32)


# ---------------------------------------------------------------------------
# Adapter: build KpDataset-format records from a FreiHAND directory
# ---------------------------------------------------------------------------

def adapt_freihand(
    root: str,
    max_images: Optional[int] = None,
    data_root: str = "",
) -> list[dict]:
    """Build KpDataset-format records from a FreiHAND v2 directory.

    Args:
        root:       Path to the FreiHAND root directory containing
                    ``training/``, ``training_xyz.json``, ``training_K.json``.
        max_images: If given, cap the total number of records (for smoke runs).
        data_root:  If non-empty, image paths in records are made relative to
                    this directory.  Pass the same value as ``data_root`` in
                    the YAML config so KpDataset can resolve them.

    Returns:
        List of dicts, each with:
          - "image"    : str — absolute or data_root-relative path to the JPEG
          - "box"      : [0, 0, 224, 224]
          - "keypoints": list of 21 [x, y] pixel coordinates

    Note: ``evaluation/`` labels are withheld by FreiHAND; only ``training/``
    is used here.  Split into train/val via ``build_landmark_manifests.py``.
    """
    xyz_path = os.path.join(root, "training_xyz.json")
    K_path   = os.path.join(root, "training_K.json")

    with open(xyz_path) as f:
        xyz_all: list = json.load(f)
    with open(K_path) as f:
        K_all:   list = json.load(f)

    n_base   = len(xyz_all)          # may differ from 32,560 in synthetic tests
    n_images = n_base * _FREIHAND_N_VARIANTS

    if max_images is not None:
        n_images = min(n_images, max_images)

    rgb_dir = os.path.join(root, "training", "rgb")
    box     = [0, 0, _FREIHAND_IMAGE_SIZE, _FREIHAND_IMAGE_SIZE]

    records: list[dict] = []
    for i in range(n_images):
        img_path_abs = os.path.join(rgb_dir, f"{i:08d}.jpg")

        if data_root:
            img_path = os.path.relpath(img_path_abs, data_root)
        else:
            img_path = img_path_abs

        label_idx = i % n_base
        xyz = np.array(xyz_all[label_idx], dtype=np.float32)
        K   = np.array(K_all[label_idx],   dtype=np.float32)

        uv = project_2d(xyz, K)   # (21, 2) float32

        records.append({
            "image":     img_path,
            "box":       box,
            "keypoints": uv.tolist(),
        })

    return records
