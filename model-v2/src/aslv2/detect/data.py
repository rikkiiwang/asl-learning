"""Detection dataset loader.

Reads a unified JSON manifest:
  [{"image": <path>, "boxes": [[x1,y1,x2,y2], ...], "labels": [0|1, ...]}]

Resizes images to 128×128, scales boxes accordingly, normalises per-channel,
and applies simple augmentations when train=True (brightness/contrast jitter +
small scale/translate that also updates boxes).  No horizontal flip — the
hand/head spatial arrangement has left/right meaning for ASL.
"""
import json
import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

_IMG_SIZE = 128


class DetDataset(Dataset):
    def __init__(self, manifest_path: str, norm: dict, train: bool = True,
                 data_root: str = "", img_size: int = _IMG_SIZE):
        """
        Args:
            manifest_path: path to unified JSON manifest.
            norm: dict with keys "mean" and "std", each a list of 3 floats.
            train: if True, apply augmentations.
            data_root: optional root directory prepended to relative image paths.
                If an image path is absolute, os.path.join returns it unchanged,
                so absolute entries remain backward-compatible.
            img_size: square side images are resized to (boxes scaled to match).
                Defaults to 128; raise it (with matching cfg['img']) to train at
                higher resolution. Anchors auto-track via anchors.scales_for.
        """
        with open(manifest_path) as f:
            self._entries = json.load(f)
        self._mean = np.array(norm["mean"], dtype=np.float32).reshape(3, 1, 1)
        self._std  = np.array(norm["std"],  dtype=np.float32).reshape(3, 1, 1)
        self._train = train
        self._data_root = data_root
        self._img_size = img_size

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, idx):
        entry = self._entries[idx]
        img_path = os.path.join(self._data_root, entry["image"]) if self._data_root else entry["image"]
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        h_orig, w_orig = img_bgr.shape[:2]
        boxes  = np.array(entry["boxes"],  dtype=np.float32).reshape(-1, 4)
        labels = np.array(entry["labels"], dtype=np.int64)

        # ----- resize to img_size×img_size -----
        S = self._img_size
        img_bgr = cv2.resize(img_bgr, (S, S),
                             interpolation=cv2.INTER_LINEAR)
        sx = S / w_orig
        sy = S / h_orig
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy

        # ----- optional augmentation (train only) -----
        if self._train and len(boxes) > 0:
            img_bgr, boxes = _augment(img_bgr, boxes)

        # Clip boxes to image boundary
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, S)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, S)

        # Drop degenerate boxes (width < 1 or height < 1 after clipping)
        keep = ((boxes[:, 2] - boxes[:, 0]) >= 1) & ((boxes[:, 3] - boxes[:, 1]) >= 1)
        boxes  = boxes[keep]
        labels = labels[keep]

        # ----- convert to tensor (C, H, W) in [0,1], then normalise -----
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1))  # (3, H, W)
        mean = torch.from_numpy(self._mean)
        std  = torch.from_numpy(self._std)
        tensor = (tensor - mean) / std

        return tensor, boxes, labels


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def _augment(img_bgr: np.ndarray, boxes: np.ndarray):
    """Brightness/contrast jitter + small scale/translate (boxes follow)."""
    # Brightness / contrast jitter
    alpha = random.uniform(0.8, 1.2)   # contrast
    beta  = random.uniform(-20, 20)    # brightness (in [0..255] space)
    img_bgr = np.clip(img_bgr.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # Small random scale + translate via affine
    h, w = img_bgr.shape[:2]
    scale = random.uniform(0.9, 1.1)
    tx    = random.uniform(-0.05 * w, 0.05 * w)
    ty    = random.uniform(-0.05 * h, 0.05 * h)

    cx, cy = w / 2.0, h / 2.0
    M = np.array([
        [scale, 0,     tx + cx * (1 - scale)],
        [0,     scale, ty + cy * (1 - scale)],
    ], dtype=np.float32)

    img_bgr = cv2.warpAffine(img_bgr, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)

    # Transform boxes: apply M to each corner, then take tight bbox
    new_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        corners = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]], dtype=np.float32)
        ones = np.ones((4, 1), dtype=np.float32)
        corners_h = np.hstack([corners, ones])       # (4, 3)
        transformed = (M @ corners_h.T).T             # (4, 2)
        new_boxes.append([
            transformed[:, 0].min(),
            transformed[:, 1].min(),
            transformed[:, 0].max(),
            transformed[:, 1].max(),
        ])
    boxes = np.array(new_boxes, dtype=np.float32)
    return img_bgr, boxes
