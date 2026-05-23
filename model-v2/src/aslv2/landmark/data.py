"""Keypoint dataset loader for hand-landmark training.

Manifest format (JSON list):
  [{"image": <path>, "box": [x1,y1,x2,y2], "keypoints": [[x,y]×21]}]

Each entry is a single hand crop: the image is read, the hand region is cropped
(with a small margin), resized to 64×64, normalised per-channel.  Keypoints are
given in original-image pixel coords and are normalised to [0,1] in crop space.

Augmentations (train=True): brightness/contrast jitter, small rotation+scale
that also transforms keypoints.  No horizontal flip — left/right is semantically
meaningful for ASL.
"""
import json
import math
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

_CROP_SIZE = 64
_MARGIN = 0.1   # fractional margin added around the hand box


class KpDataset(Dataset):
    def __init__(self, manifest_path: str, norm: dict, train: bool = True):
        """
        Args:
            manifest_path: path to JSON manifest list.
            norm: dict with "mean" and "std", each a list of 3 floats (RGB).
            train: apply augmentations when True.
        """
        with open(manifest_path) as f:
            self._entries = json.load(f)
        self._mean = np.array(norm["mean"], dtype=np.float32)
        self._std  = np.array(norm["std"],  dtype=np.float32)
        self._train = train

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx):
        entry = self._entries[idx]
        img_bgr = cv2.imread(entry["image"])
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {entry['image']}")

        h_orig, w_orig = img_bgr.shape[:2]
        box = np.array(entry["box"], dtype=np.float32)   # [x1,y1,x2,y2]
        kps = np.array(entry["keypoints"], dtype=np.float32)  # (21,2) pixels

        # --- add margin around box ---
        bw = box[2] - box[0]
        bh = box[3] - box[1]
        mx = bw * _MARGIN
        my = bh * _MARGIN
        x1 = max(0.0, box[0] - mx)
        y1 = max(0.0, box[1] - my)
        x2 = min(float(w_orig), box[2] + mx)
        y2 = min(float(h_orig), box[3] + my)

        # --- crop ---
        ix1, iy1, ix2, iy2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        ix2 = max(ix2, ix1 + 1)
        iy2 = max(iy2, iy1 + 1)
        crop_bgr = img_bgr[iy1:iy2, ix1:ix2]
        crop_h, crop_w = crop_bgr.shape[:2]

        # shift keypoints to crop-local pixel coords
        kps_local = kps.copy()
        kps_local[:, 0] -= ix1
        kps_local[:, 1] -= iy1

        # --- optional augmentation (in crop space) ---
        if self._train:
            crop_bgr, kps_local = _augment(crop_bgr, kps_local)
            crop_h, crop_w = crop_bgr.shape[:2]

        # clip keypoints to crop bounds
        kps_local[:, 0] = kps_local[:, 0].clip(0, crop_w)
        kps_local[:, 1] = kps_local[:, 1].clip(0, crop_h)

        # --- resize to 64×64 ---
        crop_bgr = cv2.resize(crop_bgr, (_CROP_SIZE, _CROP_SIZE),
                              interpolation=cv2.INTER_LINEAR)
        # scale keypoints to 64×64 space and then normalise to [0,1]
        kps_norm = kps_local.copy()
        kps_norm[:, 0] = kps_norm[:, 0] / max(crop_w, 1)
        kps_norm[:, 1] = kps_norm[:, 1] / max(crop_h, 1)
        kps_norm = kps_norm.clip(0.0, 1.0)

        # --- convert image to normalised tensor ---
        img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # (H,W,C) -> (C,H,W)
        tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1))
        mean_t = torch.tensor(self._mean).view(3, 1, 1)
        std_t  = torch.tensor(self._std).view(3, 1, 1)
        tensor = (tensor - mean_t) / std_t

        kp_tensor = torch.from_numpy(kps_norm)  # (21,2) float32

        return tensor, kp_tensor


# ---------------------------------------------------------------------------
# Augmentation helpers (all applied in pre-resize crop space)
# ---------------------------------------------------------------------------

def _augment(img_bgr: np.ndarray, kps: np.ndarray):
    """Brightness/contrast jitter + small rotation+scale (keypoints follow)."""
    # Brightness / contrast jitter
    alpha = random.uniform(0.8, 1.2)
    beta  = random.uniform(-20.0, 20.0)
    img_bgr = np.clip(img_bgr.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # Small rotation (±10°) + scale (0.9–1.1)
    h, w = img_bgr.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    angle = random.uniform(-10.0, 10.0)
    scale = random.uniform(0.9, 1.1)
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)

    img_bgr = cv2.warpAffine(img_bgr, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)

    # Transform keypoints with same matrix
    ones = np.ones((len(kps), 1), dtype=np.float32)
    kps_h = np.hstack([kps, ones])          # (21,3)
    kps_transformed = (M @ kps_h.T).T       # (21,2)
    kps_transformed = kps_transformed.astype(np.float32)

    return img_bgr, kps_transformed
