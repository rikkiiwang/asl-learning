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


def motion_roi_box(frames, margin=0.20):
    """Classical (no learned model) motion-based ROI.

    Sign motion lives in the hands/arms; the background is static. We accumulate
    per-pixel temporal variation, threshold it, take a robust bounding box of the
    moving region, pad it, and square it. This zooms the signing space in (more
    pixels on the hands) and strips per-signer background — a likely overfit cue.

    Returns (y0, x0, side) square box, or None to fall back to center crop.
    """
    if len(frames) < 3:
        return None
    h, w = frames[0].shape[:2]
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    g = np.stack(grays, 0)                          # (T,h,w)
    motion = np.abs(np.diff(g, axis=0)).mean(0)     # mean inter-frame change
    motion = cv2.GaussianBlur(motion, (0, 0), sigmaX=max(h, w) / 80.0)
    mx = float(motion.max())
    if mx < 2.0:                                    # almost no motion -> bail
        return None
    mask = motion > 0.25 * mx
    ys, xs = np.where(mask)
    if len(xs) < 25:
        return None
    # robust extent (trim outliers), then pad. Tight percentiles + a safety
    # floor avoid over-cropping low-motion signs (e.g. PURPLE/YELLOW).
    x1, x2 = np.percentile(xs, [5, 95])
    y1, y2 = np.percentile(ys, [5, 95])
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(bw, bh) * (1.0 + margin)
    side = float(np.clip(side, 0.55 * min(h, w), min(h, w)))
    x0 = int(round(np.clip(cx - side / 2, 0, w - side)))
    y0 = int(round(np.clip(cy - side / 2, 0, h - side)))
    return y0, x0, int(round(side))


def compute_flow(frames_rgb):
    """frames_rgb: (k,H,W,3) uint8 RGB. Returns (k,H,W,2) float32 Farneback flow
    (dx,dy). flow[0] is zeros; flow[t] is motion from frame t-1 -> t.

    Classical algorithm, no learned weights (from-scratch-compliant)."""
    k, H, W, _ = frames_rgb.shape
    flow = np.zeros((k, H, W, 2), dtype=np.float32)
    prev = cv2.cvtColor(frames_rgb[0], cv2.COLOR_RGB2GRAY)
    for t in range(1, k):
        cur = cv2.cvtColor(frames_rgb[t], cv2.COLOR_RGB2GRAY)
        flow[t] = cv2.calcOpticalFlowFarneback(
            prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prev = cur
    return flow


def load_clip(path, k, size, roi=False, with_flow=False):
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
    box = motion_roi_box(frames) if roi else None
    idx = sample_indices(len(frames), k)
    out = np.empty((k, size, size, 3), dtype=np.uint8)
    for i, fi in enumerate(idx):
        f = frames[fi]
        h, w = f.shape[:2]
        if box is not None:
            y0, x0, s = box
        else:
            s = min(h, w)                   # center square crop
            y0, x0 = (h - s) // 2, (w - s) // 2
        f = f[y0:y0 + s, x0:x0 + s]
        f = cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA)
        out[i] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    if with_flow:
        flow = compute_flow(out)                       # (k,size,size,2) float32
        return out, flow.astype(np.float16)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="artifacts/manifest/manifest.json")
    ap.add_argument("--videos", default="data/ASL_Citizen/videos")
    ap.add_argument("--out", default="artifacts/cache/clips.npz")
    ap.add_argument("--norm-out", default="artifacts/manifest/norm.json")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--roi", action="store_true",
                    help="motion-based ROI crop instead of center crop")
    ap.add_argument("--with-flow", action="store_true",
                    help="also compute+store Farneback flow (2ch float16)")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    clips = [(c["file"], s["label_idx"], c["split"], c["participant"])
             for s in manifest["signs"] for c in s["clips"]]

    X = np.empty((len(clips), args.frames, args.size, args.size, 3), dtype=np.uint8)
    Xflow = (np.empty((len(clips), args.frames, args.size, args.size, 2),
                      dtype=np.float16) if args.with_flow else None)
    y = np.empty(len(clips), dtype=np.int64)
    split = np.empty(len(clips), dtype=object)
    participant = np.empty(len(clips), dtype=object)
    files = np.empty(len(clips), dtype=object)

    ok = 0
    for i, (fname, lab, sp, pid) in enumerate(clips):
        res = load_clip(os.path.join(args.videos, fname), args.frames, args.size,
                        roi=args.roi, with_flow=args.with_flow)
        if res is None:
            print(f"[SKIP] could not decode {fname}")
            continue
        if args.with_flow:
            clip, fl = res
        else:
            clip = res
        X[ok], y[ok], split[ok], participant[ok], files[ok] = clip, lab, sp, pid, fname
        if args.with_flow:
            Xflow[ok] = fl
        ok += 1
        if ok % 250 == 0:
            print(f"  processed {ok}/{len(clips)} ...")
    X, y, split, participant, files = X[:ok], y[:ok], split[:ok], participant[:ok], files[:ok]
    if args.with_flow:
        Xflow = Xflow[:ok]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_kw = dict(X=X, y=y, split=split.astype(str),
                   participant=participant.astype(str), files=files.astype(str))
    if args.with_flow:
        save_kw["Xflow"] = Xflow
    np.savez_compressed(args.out, **save_kw)

    # float64 accumulation — float32 mean over ~2e8 elements loses precision badly
    tr = X[split == "train"].reshape(-1, 3).astype(np.float64) / 255.0
    mean = tr.mean(0).tolist()
    std = tr.std(0).tolist()
    norm = {"mean": mean, "std": std}
    os.makedirs(os.path.dirname(args.norm_out), exist_ok=True)
    json.dump(norm, open(args.norm_out, "w"), indent=2)

    if args.with_flow:
        flow_tr = Xflow[split == "train"].reshape(-1, 2).astype(np.float64)
        flow_norm = {"mean": flow_tr.mean(0).tolist(),
                     "std": (flow_tr.std(0) + 1e-6).tolist()}
        flow_norm_out = args.norm_out.replace(".json", "_flow.json")
        json.dump(flow_norm, open(flow_norm_out, "w"), indent=2)
        print(f"flow norm -> {flow_norm_out}: {flow_norm}")

    sz = os.path.getsize(args.out) / 1e9
    print(f"\nCached {ok} clips -> {args.out} ({sz:.2f} GB)")
    print(f"shape {X.shape}, dtype {X.dtype}")
    print(f"train norm mean={[round(m,3) for m in mean]} std={[round(s,3) for s in std]}")


if __name__ == "__main__":
    main()
