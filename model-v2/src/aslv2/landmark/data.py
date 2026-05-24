"""Keypoint dataset loader for hand-landmark training.

Manifest format (JSON list):
  [{"image": <path>, "box": [x1,y1,x2,y2], "keypoints": [[x,y]×21]}]

FRAMING (the important part). The hand crop is built from a square window
*centred on the keypoint bounding box*, sized `scale × hand_extent`:

  - train: scale ~ U(1.2, 2.8), centre jittered, plus rotation ±25° and
    brightness/contrast jitter. This teaches the model to localise a hand that
    occupies a VARIABLE fraction of the crop and isn't perfectly centred — the
    robustness the first model lacked (it assumed FreiHAND's fixed centred
    framing and collapsed to a mean hand on real, tightly-cropped webcam hands).
  - val: scale = 2.0, centred, no augmentation (deterministic).

The window is mapped to 64×64 with a single affine warp (border-reflect padding
handles windows that extend past the image edge). Keypoints are transformed by
the same matrix and normalised to [0,1] of the crop.

At inference the same square-window framing is applied around the *detector's*
hand box (see scripts/detect_demo.py), so train and inference framing match.
The manifest "box" field is no longer used for the crop (kept for compatibility).
"""
import json
import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

_CROP_SIZE = 64

# Framing / augmentation knobs
_TRAIN_SCALE = (1.2, 2.8)   # window = scale × hand extent
_VAL_SCALE   = 2.0
_JITTER      = 0.25         # centre jitter as a fraction of hand extent
_ROT_DEG     = 25.0


class KpDataset(Dataset):
    def __init__(self, manifest_path: str, norm: dict, train: bool = True,
                 data_root: str = ""):
        """
        Args:
            manifest_path: path to JSON manifest list.
            norm: dict with "mean" and "std", each a list of 3 floats (RGB).
            train: apply framing + photometric augmentation when True.
            data_root: optional root prepended to relative image paths. Absolute
                paths pass through unchanged (os.path.join), so manifests with
                absolute entries stay backward-compatible.
        """
        with open(manifest_path) as f:
            self._entries = json.load(f)
        self._mean = np.array(norm["mean"], dtype=np.float32)
        self._std  = np.array(norm["std"],  dtype=np.float32)
        self._train = train
        self._data_root = data_root

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx):
        entry = self._entries[idx]
        img_path = os.path.join(self._data_root, entry["image"]) if self._data_root else entry["image"]
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        kps = np.array(entry["keypoints"], dtype=np.float32)   # (21,2) pixels

        # --- hand extent from keypoints ---
        kx1, ky1 = kps.min(0)
        kx2, ky2 = kps.max(0)
        cx, cy = (kx1 + kx2) / 2.0, (ky1 + ky2) / 2.0
        extent = max(float(kx2 - kx1), float(ky2 - ky1), 1.0)

        # --- choose framing window ---
        if self._train:
            scale = random.uniform(*_TRAIN_SCALE)
            cx += random.uniform(-_JITTER, _JITTER) * extent
            cy += random.uniform(-_JITTER, _JITTER) * extent
            angle = random.uniform(-_ROT_DEG, _ROT_DEG)
        else:
            scale, angle = _VAL_SCALE, 0.0

        win = extent * scale
        x1, y1 = cx - win / 2.0, cy - win / 2.0

        # affine: map the window → 64×64 (+ optional rotation about the crop centre)
        s = _CROP_SIZE / win
        M = np.array([[s, 0.0, -x1 * s],
                      [0.0, s, -y1 * s]], dtype=np.float32)
        if angle != 0.0:
            R = cv2.getRotationMatrix2D((_CROP_SIZE / 2.0, _CROP_SIZE / 2.0), angle, 1.0)
            M = (np.vstack([R, [0, 0, 1]]) @ np.vstack([M, [0, 0, 1]]))[:2].astype(np.float32)

        crop_bgr = cv2.warpAffine(img_bgr, M, (_CROP_SIZE, _CROP_SIZE),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101)

        # transform keypoints by the same matrix → 64-px space
        kps_h = np.hstack([kps, np.ones((len(kps), 1), dtype=np.float32)])
        kps_c = (M @ kps_h.T).T                       # (21,2)

        if self._train:
            crop_bgr = _photometric(crop_bgr)

        kps_norm = (kps_c / _CROP_SIZE).clip(0.0, 1.0)

        img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1))
        mean_t = torch.tensor(self._mean).view(3, 1, 1)
        std_t  = torch.tensor(self._std).view(3, 1, 1)
        tensor = (tensor - mean_t) / std_t

        return tensor, torch.from_numpy(kps_norm.astype(np.float32))


# ---------------------------------------------------------------------------
# Photometric augmentation (brightness / contrast jitter)
# ---------------------------------------------------------------------------

def _photometric(img_bgr: np.ndarray) -> np.ndarray:
    alpha = random.uniform(0.8, 1.2)    # contrast
    beta  = random.uniform(-20.0, 20.0)  # brightness
    return np.clip(img_bgr.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
