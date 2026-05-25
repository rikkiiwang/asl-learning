"""Build the per-clip geometry+crops cache (Plan 4 Task 2).

Runs the trained detector + landmark over the ASL Citizen 75-sign subset and
stores, per clip: geom (16,93), crops (16,2,3,64,64) uint8, label y, participant.
The recognizer then trains on this cache (no re-running the heavy models per epoch).

Run (videos are local, so this runs on the M4, not Colab):
    cd model-v2 && PYTHONPATH=src .venv/bin/python -m aslv2.recog.cache \
        --out artifacts/cache/constellation_clips.npz 2>&1 | tee artifacts/cache/cache.log
Use --max-clips N for a quick sanity run first.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # model-v2/src on path

from aslv2.detect.infer import detect_frame
from aslv2.detect.model import Detector
from aslv2.landmark.infer import landmark_hand, map_to_frame
from aslv2.landmark.model import Landmark
from aslv2.recog.clip_features import build_clip_features

REPO = Path(__file__).resolve().parents[4]            # ASL Learning/
VIDEO_DIR = REPO / "model" / "data" / "ASL_Citizen" / "videos"
MANIFEST = REPO / "model" / "artifacts" / "manifest" / "manifest.json"
LM_FRAME_SCALE = 2.0                                  # matches KpDataset val framing
K = 16                                                # frozen clip length


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sample_indices(n: int, k: int = K):
    """v1's uniform temporal resample (model/src/asl/preprocess.py)."""
    if n <= 0:
        return [0] * k
    if n >= k:
        return np.linspace(0, n - 1, k).round().astype(int).tolist()
    return list(range(n)) + [n - 1] * (k - n)


def active_window(frames_bgr, frac: float = 0.3):
    """Motion-based signing window: ASL clips have hands-at-rest lead-in/out with
    little motion; the sign is the high-motion segment. Returns (lo, hi) indices.
    Cheap frame-differencing on 64² grayscale (no model)."""
    n = len(frames_bgr)
    if n < 6:
        return 0, n - 1
    gs = [cv2.cvtColor(cv2.resize(f, (64, 64)), cv2.COLOR_BGR2GRAY).astype(np.float32)
          for f in frames_bgr]
    motion = np.array([0.0] + [np.abs(gs[i] - gs[i - 1]).mean() for i in range(1, n)])
    motion = np.convolve(motion, np.ones(5) / 5, mode="same")     # smooth
    active = np.where(motion > frac * motion.max())[0]
    if len(active) < 2:
        return 0, n - 1
    return int(active[0]), int(active[-1])


def decode_frames(path: Path, k: int = K):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        return []
    lo, hi = active_window(frames)                                 # trim to the sign
    span = frames[lo:hi + 1] if hi > lo else frames
    idx = sample_indices(len(span), k)
    return [cv2.cvtColor(span[i], cv2.COLOR_BGR2RGB) for i in idx]


def landmark_kps(lm, lm_norm, frame_rgb, hands):
    """(k,21,2) frame-coord keypoints aligned with `hands` (square 2× framing)."""
    out = []
    for box in hands:
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        win = max(x2 - x1, y2 - y1) * LM_FRAME_SCALE
        s = 128.0 / win
        M = np.array([[s, 0, -(cx - win / 2) * s], [0, s, -(cy - win / 2) * s]], dtype=np.float32)
        crop = cv2.warpAffine(frame_rgb, M, (128, 128), borderMode=cv2.BORDER_REFLECT_101)
        kp01 = landmark_hand(lm, crop, lm_norm)
        out.append(map_to_frame(kp01, [cx - win / 2, cy - win / 2, cx + win / 2, cy + win / 2]))
    return np.stack(out) if out else np.zeros((0, 21, 2), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detector", default="artifacts/checkpoints/detector/best.pt")
    ap.add_argument("--landmark", default="artifacts/checkpoints/landmark/best.pt")
    ap.add_argument("--out", default="artifacts/cache/constellation_clips.npz")
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--score-thr", type=float, default=0.3)
    # process a flat clip list from another corpus (e.g. WLASL) instead of the
    # ASL Citizen sign-structured manifest:
    ap.add_argument("--clips-json", default=None,
                    help="flat [{file, participant, label_idx}] list")
    ap.add_argument("--video-dir", default=None, help="dir holding those videos")
    args = ap.parse_args()

    dev = _device()
    print(f"device={dev}")
    dck = torch.load(args.detector, map_location=dev, weights_only=False)
    dcfg = dck["cfg"]
    det = Detector(n_classes=2, n_anchors=dcfg["n_anchors"], width=dcfg["width"]).to(dev)
    det.load_state_dict(dck["state_dict"]); det.eval()
    lck = torch.load(args.landmark, map_location=dev, weights_only=False)
    lm = Landmark(width=lck["cfg"]["width"]).to(dev)
    lm.load_state_dict(lck["state_dict"]); lm.eval()
    lnorm = lck["cfg"]["norm"]

    if args.clips_json:
        flat = json.loads(Path(args.clips_json).read_text())
        clips = [(c["file"], c["participant"], c["label_idx"]) for c in flat]
        video_dir = Path(args.video_dir)
    else:
        manifest = json.loads(MANIFEST.read_text())
        clips = [(c["file"], c["participant"], sign["label_idx"])
                 for sign in manifest["signs"] for c in sign["clips"]]
        video_dir = VIDEO_DIR
    if args.max_clips:
        clips = clips[:args.max_clips]
    print(f"{len(clips)} clips to process (detector w={dcfg['width']} img={dcfg['img']}, landmark w={lck['cfg']['width']})")

    geoms, cropss, ys, parts = [], [], [], []
    head_rate, hand_rate, done, missing = 0.0, 0.0, 0, 0
    for n, (file, part, y) in enumerate(clips):
        vp = video_dir / file
        if not vp.exists():
            missing += 1; continue
        frames = decode_frames(vp, K)
        if len(frames) < K:
            missing += 1; continue
        dets, kps = [], []
        for fr in frames:
            d = detect_frame(det, fr, dcfg["norm"], score_thr=args.score_thr, img_size=dcfg["img"])
            dets.append(d)
            kps.append(landmark_kps(lm, lnorm, fr, d["hands"]))
        geom, crops = build_clip_features(frames, dets, kps, crop_size=64)
        geoms.append(geom.astype(np.float32))
        cropss.append((crops * 255).astype(np.uint8))
        ys.append(int(y)); parts.append(part)
        head_rate += float(geom[:, -1].mean())                       # per-clip head-present
        hand_rate += float((geom[:, 90:92].max(axis=1) > 0).mean())  # ≥1 hand per frame
        done += 1
        if done % 100 == 0:
            print(f"  {done}/{len(clips)} cached ...")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out,
             geom=np.stack(geoms), crops=np.stack(cropss),
             y=np.array(ys, dtype=np.int64), participant=np.array(parts))
    print(f"\ncached {done} clips ({missing} missing/short) -> {args.out}")
    if done:
        print(f"SANITY: head-present {head_rate/done:.3f} (gate ≥0.90) | "
              f"≥1-hand-frame {hand_rate/done:.3f} (gate ≥0.95)")


if __name__ == "__main__":
    main()
