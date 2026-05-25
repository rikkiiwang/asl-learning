"""Recognizer dataset over the per-clip cache, with signer-held-out splits.

Mirrors v1's policy (`model/artifacts/manifest/signer_splits.json`):
`{participant: "train"|"val"|"test"}`, so no signer ever appears in two splits.
Train-time augmentation is small Gaussian jitter on the *geometric* part of the
feature (keypoints + hand-relative offsets), leaving the presence flags intact.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

# geometry layout (GEOM_DIM=93): 84 kp + 4 hand2head + 2 hand2hand = 90 geometric,
# then 2 slot-presence + 1 head-present flags. Jitter only the geometric dims.
_GEOM_AUG_DIMS = 90


class RecogDataset(Dataset):
    def __init__(self, cache, split: str, split_map, train: bool = False,
                 kp_jitter: float = 0.01, load_crops: bool = True,
                 strong_aug: bool = False):
        """
        Args:
            cache:     path to the .npz cache (or an in-memory mapping) with arrays
                       geom (N,16,93), crops (N,16,2,3,H,W), y (N,), participant (N,).
            split:     "train" | "val" | "test".
            split_map: {participant: split} dict, or a path to such a JSON file.
            train:     apply geometry jitter when True.
            kp_jitter: std of the Gaussian jitter on geometric dims.
        """
        data = (np.load(cache, allow_pickle=True)
                if isinstance(cache, (str, os.PathLike)) else cache)
        if isinstance(split_map, (str, os.PathLike)):
            split_map = json.load(open(split_map))

        part = np.asarray(data["participant"]).astype(str)
        idx = np.array([i for i, p in enumerate(part) if split_map.get(p) == split],
                       dtype=int)
        self._geom = np.asarray(data["geom"])[idx]
        self._crops = np.asarray(data["crops"])[idx] if load_crops else None
        self._y = np.asarray(data["y"])[idx]
        self._part = part[idx]
        self.train = train
        self.kp_jitter = kp_jitter
        self.load_crops = load_crops
        self.strong_aug = strong_aug

    def _augment_geom(self, g: np.ndarray) -> np.ndarray:
        """Aggressive geometry augmentation for the small-data regime (train only).

        The 90 geometric dims are all (x,y) pairs in the head-centred frame, so a
        global rotation/scale applies uniformly. Adds: temporal warp (sign-speed
        invariance), global pose rotation/scale, per-coord jitter, frame dropout
        (simulate missed detections). The 3 presence/head flags are not rotated.
        """
        F = g.shape[0]
        # temporal warp: random speed + offset, nearest-frame resample
        s = np.random.uniform(0.8, 1.25)
        idx = np.clip(np.round(np.arange(F) * s + np.random.uniform(-1, 1)), 0, F - 1).astype(int)
        g = g[idx].copy()
        geo = g[:, :_GEOM_AUG_DIMS].reshape(F, _GEOM_AUG_DIMS // 2, 2)   # (F,45,2) xy pairs
        flags = g[:, _GEOM_AUG_DIMS:]
        # global rotation about the head centre
        a = np.deg2rad(np.random.uniform(-15, 15))
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], dtype=np.float32)
        geo = geo @ R.T
        geo *= np.random.uniform(0.9, 1.1)                              # global scale
        geo += np.random.randn(*geo.shape).astype(np.float32) * self.kp_jitter
        g = np.concatenate([geo.reshape(F, _GEOM_AUG_DIMS), flags], axis=1).astype(np.float32)
        # frame dropout: blank a few frames (hand "not detected")
        if np.random.rand() < 0.5:
            drop = np.random.choice(F, np.random.randint(1, 4), replace=False)
            g[drop, :_GEOM_AUG_DIMS] = 0.0
            g[drop, _GEOM_AUG_DIMS:_GEOM_AUG_DIMS + 2] = 0.0            # mark hands absent
        return g

    def __len__(self) -> int:
        return len(self._y)

    def participants(self) -> set:
        return set(self._part.tolist())

    def __getitem__(self, i):
        g = self._geom[i].astype(np.float32).copy()
        if self.train and self.strong_aug:
            g = self._augment_geom(g)
        elif self.train and self.kp_jitter > 0:
            noise = np.random.randn(g.shape[0], _GEOM_AUG_DIMS).astype(np.float32)
            g[:, :_GEOM_AUG_DIMS] += noise * self.kp_jitter
        if self._crops is None:
            c = torch.empty(0)
        else:
            c = torch.from_numpy(self._crops[i].astype(np.float32))
        return torch.from_numpy(g), c, int(self._y[i])
