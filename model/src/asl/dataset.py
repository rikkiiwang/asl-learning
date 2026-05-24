"""Manifest-cache-backed torch Dataset with train-time augmentation.

Reads the npz cache from preprocess.py. Split assignment comes straight from the
cache (`split` array), so changing the split strategy is just a re-cache/re-split,
not a code change. Augmentation is applied ONLY to the train split.

Augmentation (no horizontal flip — flipping can change a sign's meaning):
  - photometric: brightness / contrast jitter
  - geometric: small random crop + shift (via resized crop)
  - temporal: small frame-index jitter is handled at sampling time upstream;
    here we optionally drop/repeat is avoided to keep 16 frames fixed.
"""
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def augment_clip(clip, size, rng=None):
    """clip: (F,H,W,3) float32 in 0-1. ONE shared geometric+photometric transform
    across all frames. No horizontal flip (can change a sign's meaning)."""
    rng = rng if rng is not None else np.random.default_rng()
    # photometric: brightness + contrast
    b = rng.uniform(-0.12, 0.12)
    c = rng.uniform(0.85, 1.15)
    clip = np.clip((clip - 0.5) * c + 0.5 + b, 0, 1).astype(np.float32)
    F, H, W, _ = clip.shape
    # geometric: small rotation, same angle for every frame
    ang = float(rng.uniform(-10.0, 10.0))
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), ang, 1.0)
    clip = np.stack([cv2.warpAffine(clip[i], M, (W, H), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT)
                     for i in range(F)], 0)
    # geometric: random resized crop (zoom + translate)
    scale = float(rng.uniform(0.85, 1.0))
    ch, cw = int(H * scale), int(W * scale)
    y0 = int(rng.integers(0, H - ch + 1))
    x0 = int(rng.integers(0, W - cw + 1))
    clip = clip[:, y0:y0 + ch, x0:x0 + cw, :]
    t = torch.from_numpy(np.ascontiguousarray(clip)).permute(0, 3, 1, 2)
    t = torch.nn.functional.interpolate(t, size=(size, size),
                                        mode="bilinear", align_corners=False)
    return t.permute(0, 2, 3, 1).numpy().astype(np.float32)


def _load_norm(norm_path):
    n = json.load(open(norm_path))
    return (np.array(n["mean"], dtype=np.float32),
            np.array(n["std"], dtype=np.float32))


class ClipDataset(Dataset):
    def __init__(self, npz_path, split, norm_path="artifacts/manifest/norm.json",
                 train=False, size=112,
                 signer_splits="artifacts/manifest/signer_splits.json",
                 two_stream=False, flow_norm_path=None):
        data = np.load(npz_path, allow_pickle=True)
        if signer_splits and os.path.exists(signer_splits):
            policy = json.load(open(signer_splits))
            part = data["participant"].astype(str)
            mask = np.array([policy.get(p) == split for p in part])
        else:
            mask = data["split"].astype(str) == split
        self.X = data["X"][mask]
        self.y = data["y"][mask].astype(np.int64)
        self.two_stream = two_stream
        if two_stream:
            self.Xflow = data["Xflow"][mask]
            fm = json.load(open(flow_norm_path))
            self.fmean = np.array(fm["mean"], dtype=np.float32)
            self.fstd = np.array(fm["std"], dtype=np.float32)
        self.train = train
        self.size = size
        self.mean, self.std = _load_norm(norm_path)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        clip = self.X[i].astype(np.float32) / 255.0
        if self.train:
            clip = augment_clip(clip, self.size)
        clip = (clip - self.mean) / self.std
        rgb = torch.from_numpy(clip).permute(0, 3, 1, 2).float()
        if not self.two_stream:
            return rgb, int(self.y[i])
        flow = self.Xflow[i].astype(np.float32)
        flow = (flow - self.fmean) / self.fstd
        flow = torch.from_numpy(flow).permute(0, 3, 1, 2).float()
        return (rgb, flow), int(self.y[i])


class MemmapClipDataset(Dataset):
    """Lazy, disk-backed dataset for the large pretraining cache (frames.dat)."""
    def __init__(self, cache_dir, indices, norm_path="artifacts/manifest/norm.json",
                 train=False, size=112):
        meta = np.load(os.path.join(cache_dir, "meta.npz"), allow_pickle=True)
        shape = tuple(int(s) for s in meta["shape"])
        self.X = np.memmap(os.path.join(cache_dir, "frames.dat"),
                           dtype=np.uint8, mode="r", shape=shape)
        self.y = meta["y"].astype(np.int64)
        self.idx = np.asarray(indices)
        self.train = train
        self.size = size
        self.mean, self.std = _load_norm(norm_path)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        clip = np.asarray(self.X[j], dtype=np.float32) / 255.0
        if self.train:
            clip = augment_clip(clip, self.size)
        clip = (clip - self.mean) / self.std
        return torch.from_numpy(clip).permute(0, 3, 1, 2).float(), int(self.y[j])
