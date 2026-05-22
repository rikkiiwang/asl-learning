"""Phase 1 — decode manifest clips into a cached frame tensor.

Each clip -> 16 frames uniformly sampled across the whole (sign-trimmed) video,
center-cropped to square, resized to 112x112 RGB, stored uint8. Augmentation and
normalization are applied later at train time (NOT baked into the cache).

Output: artifacts/cache/clips.npz with
    X     uint8 (N, 16, 112, 112, 3)
    y     int   (N,)            label index
    split  str  (N,)            train/val/test
    participant str (N,)
    files  str  (N,)
and artifacts/manifest/norm.json with per-channel train mean/std (0-1 scale).

Usage:
    python -m asl.preprocess --manifest artifacts/manifest/manifest.json \
        --videos data/ASL_Citizen/videos --out artifacts/cache/clips.npz
"""
import argparse
import json
import os

import cv2
import numpy as np


def sample_indices(n_frames, k):
    if n_frames <= 0:
        return [0] * k
    if n_frames >= k:
        return np.linspace(0, n_frames - 1, k).round().astype(int).tolist()
    # fewer frames than needed: take all, then repeat the last to pad
    return list(range(n_frames)) + [n_frames - 1] * (k - n_frames)


def load_clip(path, k, size):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        return None
    idx = sample_indices(len(frames), k)
    out = np.empty((k, size, size, 3), dtype=np.uint8)
    for i, fi in enumerate(idx):
        f = frames[fi]
        h, w = f.shape[:2]
        s = min(h, w)                       # center square crop
        y0, x0 = (h - s) // 2, (w - s) // 2
        f = f[y0:y0 + s, x0:x0 + s]
        f = cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA)
        out[i] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="artifacts/manifest/manifest.json")
    ap.add_argument("--videos", default="data/ASL_Citizen/videos")
    ap.add_argument("--out", default="artifacts/cache/clips.npz")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=112)
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    clips = [(c["file"], s["label_idx"], c["split"], c["participant"])
             for s in manifest["signs"] for c in s["clips"]]

    X = np.empty((len(clips), args.frames, args.size, args.size, 3), dtype=np.uint8)
    y = np.empty(len(clips), dtype=np.int64)
    split = np.empty(len(clips), dtype=object)
    participant = np.empty(len(clips), dtype=object)
    files = np.empty(len(clips), dtype=object)

    ok = 0
    for i, (fname, lab, sp, pid) in enumerate(clips):
        clip = load_clip(os.path.join(args.videos, fname), args.frames, args.size)
        if clip is None:
            print(f"[SKIP] could not decode {fname}")
            continue
        X[ok], y[ok], split[ok], participant[ok], files[ok] = clip, lab, sp, pid, fname
        ok += 1
        if ok % 250 == 0:
            print(f"  processed {ok}/{len(clips)} ...")
    X, y, split, participant, files = X[:ok], y[:ok], split[:ok], participant[:ok], files[:ok]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y, split=split.astype(str),
                        participant=participant.astype(str), files=files.astype(str))

    # float64 accumulation — float32 mean over ~2e8 elements loses precision badly
    tr = X[split == "train"].reshape(-1, 3).astype(np.float64) / 255.0
    mean = tr.mean(0).tolist()
    std = tr.std(0).tolist()
    norm = {"mean": mean, "std": std}
    json.dump(norm, open("artifacts/manifest/norm.json", "w"), indent=2)

    sz = os.path.getsize(args.out) / 1e9
    print(f"\nCached {ok} clips -> {args.out} ({sz:.2f} GB)")
    print(f"shape {X.shape}, dtype {X.dtype}")
    print(f"train norm mean={[round(m,3) for m in mean]} std={[round(s,3) for s in std]}")


if __name__ == "__main__":
    main()
