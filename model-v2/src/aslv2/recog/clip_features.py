"""Pure per-clip feature builder (Plan 4 Task 1).

Turns a clip's per-frame detections + per-hand keypoints into the inputs the
recognizer consumes:

    geom  : (F, GEOM_DIM)            appearance-invariant pose geometry per frame
    crops : (F, 2, 3, crop, crop)    per-slot hand crops in [0,1] (zeros if absent)

It is deliberately model-free (no torch, no detector/landmark) so it can be
unit-tested with synthetic detections, and it reuses `aslv2.geometry` exactly so
the cached features match what the geometry layer was designed for.

Inputs
------
frames          : list length F of HxWx3 RGB uint8 arrays.
dets_per_frame  : list length F of {"hands": (k,4) xyxy, "head": (4,) or None}.
kps_per_frame   : list length F of (k,21,2) keypoints in **frame-pixel coords**,
                  index-aligned with that frame's "hands". (The caller maps the
                  landmark model's crop-space output to frame coords first.)
"""
from __future__ import annotations

import cv2
import numpy as np

from aslv2.geometry import GEOM_DIM, normalize_geometry, resolve_head_anchor, slot_hands


def _crop(frame_rgb: np.ndarray, box: np.ndarray, size: int) -> np.ndarray:
    """Crop an xyxy box from a frame and resize to (3,size,size) float in [0,1]."""
    h, w = frame_rgb.shape[:2]
    x1 = int(max(0, min(w - 1, box[0])))
    y1 = int(max(0, min(h - 1, box[1])))
    x2 = int(max(x1 + 1, min(w, box[2])))
    y2 = int(max(y1 + 1, min(h, box[3])))
    patch = frame_rgb[y1:y2, x1:x2]
    if patch.size == 0:
        return np.zeros((3, size, size), dtype=np.float32)
    patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_LINEAR)
    return (patch.astype(np.float32) / 255.0).transpose(2, 0, 1)


def build_clip_features(frames, dets_per_frame, kps_per_frame, crop_size: int = 64):
    """See module docstring. Returns (geom (F,GEOM_DIM) float32, crops float32)."""
    F = len(frames)
    h, w = frames[0].shape[:2]

    # --- per-clip head anchor (smoothed; fallback when no head ever seen) ---
    heads = np.full((F, 4), np.nan)
    for f, d in enumerate(dets_per_frame):
        if d.get("head") is not None:
            heads[f] = np.asarray(d["head"], dtype=float)
    head_anchor, head_present = resolve_head_anchor(heads, (w, h))
    head_cx = (head_anchor[0] + head_anchor[2]) / 2.0

    # --- slot hands into 2 stable tracks (0=left, 1=right) across the clip ---
    hand_lists = [np.asarray(d["hands"], dtype=float).reshape(-1, 4) for d in dets_per_frame]
    slots, present = slot_hands(hand_lists, head_cx=head_cx)   # (F,2,4), (F,2)

    geom = np.zeros((F, GEOM_DIM), dtype=np.float32)
    crops = np.zeros((F, 2, 3, crop_size, crop_size), dtype=np.float32)

    for f in range(F):
        kps_slot = np.full((2, 21, 2), np.nan)
        hands_f = hand_lists[f]
        kps_f = np.asarray(kps_per_frame[f], dtype=float).reshape(-1, 21, 2)
        for s in range(2):
            if present[f, s] <= 0:
                continue
            box = slots[f, s]
            # the slotted box is one of this frame's hand boxes — match it back to
            # its keypoints by nearest box (exact match in practice).
            if len(hands_f) > 0:
                i = int(np.argmin(np.abs(hands_f - box).sum(axis=1)))
                if i < len(kps_f):
                    kps_slot[s] = kps_f[i]
            crops[f, s] = _crop(frames[f], box, crop_size)
        geom[f] = normalize_geometry(kps_slot, present[f], head_anchor, head_present)

    return geom, crops
