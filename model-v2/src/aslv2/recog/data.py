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
                 kp_jitter: float = 0.01, load_crops: bool = True):
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

    def __len__(self) -> int:
        return len(self._y)

    def participants(self) -> set:
        return set(self._part.tolist())

    def __getitem__(self, i):
        g = self._geom[i].astype(np.float32).copy()
        if self.train and self.kp_jitter > 0:
            noise = np.random.randn(g.shape[0], _GEOM_AUG_DIMS).astype(np.float32)
            g[:, :_GEOM_AUG_DIMS] += noise * self.kp_jitter
        if self._crops is None:
            c = torch.empty(0)
        else:
            c = torch.from_numpy(self._crops[i].astype(np.float32))
        return torch.from_numpy(g), c, int(self._y[i])
