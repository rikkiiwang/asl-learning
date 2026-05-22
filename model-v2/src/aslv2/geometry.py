"""Geometry layer: the appearance-invariant primary signal (spec §3.3, §3.6).

All inputs are pixel-space xyxy boxes / xy keypoints. The head anchor is treated
as load-bearing: it is smoothed per clip and never trusted blind.
"""
import numpy as np

HEAD_FALLBACK_SCALE = 0.5   # fraction of frame height used as body-scale when no head


def resolve_head_anchor(heads: np.ndarray, frame_wh: tuple[int, int]):
    """heads: (F,4) xyxy with np.nan rows for frames where no head was detected.
    Returns (anchor_xyxy (4,), head_present_flag {0.0,1.0}).
    Head barely moves over ~3s, so a per-clip median is both stable and a valid
    no-jitter anchor."""
    valid = heads[~np.isnan(heads).any(axis=1)]
    if len(valid) > 0:
        return np.median(valid, axis=0), 1.0
    w, h = frame_wh
    half = HEAD_FALLBACK_SCALE * h / 2.0
    cx, cy = w / 2.0, h / 2.0
    return np.array([cx - half, cy - half, cx + half, cy + half]), 0.0
