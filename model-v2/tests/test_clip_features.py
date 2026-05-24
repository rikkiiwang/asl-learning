"""Tests for the pure per-clip feature builder (Plan 4 Task 1)."""
import numpy as np

from aslv2.geometry import GEOM_DIM
from aslv2.recog.clip_features import build_clip_features


def _frame():
    return (np.random.rand(120, 160, 3) * 255).astype(np.uint8)


def _kp_in(box):
    """21 fake keypoints (frame-pixel coords) scattered inside the box."""
    x1, y1, x2, y2 = box
    p = np.random.rand(21, 2).astype(np.float32)
    p[:, 0] = x1 + p[:, 0] * (x2 - x1)
    p[:, 1] = y1 + p[:, 1] * (y2 - y1)
    return p


def _clip(n=16):
    """16 frames; frame 0 has no head (fallback), frame 1 has one hand."""
    left = np.array([20.0, 40.0, 60.0, 80.0])    # center (40,60)
    right = np.array([110.0, 40.0, 150.0, 80.0])  # center (130,60)
    head = np.array([60.0, 5.0, 100.0, 45.0])
    frames, dets, kps = [], [], []
    for f in range(n):
        hands = np.stack([left, right])
        if f == 1:
            hands = left[None, :]                # one-hand frame
        h = None if f == 0 else head             # no-head frame
        frames.append(_frame())
        dets.append({"hands": hands, "head": h})
        kps.append(np.stack([_kp_in(b) for b in hands]))
    return frames, dets, kps


def test_build_clip_features_shapes_and_finite():
    frames, dets, kps = _clip(16)
    geom, crops = build_clip_features(frames, dets, kps, crop_size=64)

    assert geom.shape == (16, GEOM_DIM)
    assert np.isfinite(geom).all()          # incl. the no-head fallback frame
    assert crops.shape == (16, 2, 3, 64, 64)
    assert crops.min() >= 0.0 and crops.max() <= 1.0


def test_one_hand_frame_zero_pads_empty_slot():
    frames, dets, kps = _clip(16)
    geom, crops = build_clip_features(frames, dets, kps, crop_size=64)
    # frame 1 has a single hand → exactly one slot crop is all-zero
    zero_slots = (crops[1] == 0).all(axis=(1, 2, 3))
    assert zero_slots.sum() == 1
