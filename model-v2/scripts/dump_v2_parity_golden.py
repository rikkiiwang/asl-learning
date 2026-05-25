"""Dump golden vectors so the app's TypeScript port of the v2 glue
(detector decode + geometry) can be parity-tested against this Python source.

Run: cd model-v2 && PYTHONPATH=src .venv/bin/python scripts/dump_v2_parity_golden.py
Writes JSON fixtures into ../app/src/inference/v2/__fixtures__/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aslv2.detect.anchors import anchors_for
from aslv2.detect.encode import decode
from aslv2.recog.clip_features import build_clip_features
from aslv2.geometry import GEOM_DIM

OUT = Path(__file__).resolve().parents[2] / "app/src/inference/v2/__fixtures__"
OUT.mkdir(parents=True, exist_ok=True)

IMG = 192
SCORE_THR = 0.3
IOU_THR = 0.45
ORIG = 256  # work-resolution square the app crops/detects in


def detect_post(cls_np, box_np, orig_w, orig_h):
    """Mirror aslv2.detect.infer.detect_frame's POST-processing (no model)."""
    anchors = anchors_for(IMG)
    box_np = np.clip(box_np, -10.0, 10.0)
    boxes, scores, labels = decode(anchors, box_np, cls_np,
                                   score_thr=SCORE_THR, iou_thr=IOU_THR)
    if len(boxes) > 0:
        valid = (np.isfinite(boxes).all(axis=1) &
                 (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))
        boxes, scores, labels = boxes[valid], scores[valid], labels[valid]
    sx, sy = orig_w / IMG, orig_h / IMG
    if len(boxes) > 0:
        boxes = boxes * np.array([sx, sy, sx, sy], dtype=np.float32)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, orig_h)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, scores, labels = boxes[valid], scores[valid], labels[valid]
    hand_b, hand_s = boxes[labels == 0], scores[labels == 0]
    if len(hand_b) > 0:
        order = np.argsort(hand_s)[::-1][:2]
        hand_b = hand_b[order]
    else:
        hand_b = np.empty((0, 4), np.float32)
    head = None
    head_b, head_s = boxes[labels == 1], scores[labels == 1]
    if len(head_b) > 0:
        head = head_b[int(np.argmax(head_s))]
    return hand_b, head


def make_decode_golden():
    rng = np.random.default_rng(7)
    n = len(anchors_for(IMG))   # 432
    box_np = (rng.standard_normal((n, 4)) * 0.3).astype(np.float32)
    cls_np = (rng.standard_normal((n, 2)) * 1.5 - 2.0).astype(np.float32)
    # plant confident hands (class 0) and heads (class 1) with DISTINCT scores
    # so top-k / NMS selection is unambiguous (no tie-break dependence).
    for i, v in ((10, 5.0), (120, 4.5), (250, 4.0), (300, 3.5)):
        cls_np[i, 0] = v
    for i, v in ((60, 5.0), (61, 4.0)):
        cls_np[i, 1] = v
    hand_b, head = detect_post(cls_np, box_np, ORIG, ORIG)
    return {
        "img": IMG, "origW": ORIG, "origH": ORIG,
        "scoreThr": SCORE_THR, "iouThr": IOU_THR,
        "cls": cls_np.reshape(-1).tolist(),
        "box": box_np.reshape(-1).tolist(),
        "expect": {
            "hands": hand_b.astype(float).tolist(),
            "head": (None if head is None else head.astype(float).tolist()),
        },
    }


def make_geometry_golden():
    """Synthetic 16-frame clip of detections + keypoints -> geom (16,93)."""
    rng = np.random.default_rng(11)
    F = 16
    W = H = float(ORIG)
    frames = [np.zeros((int(H), int(W), 3), np.uint8) for _ in range(F)]
    dets, kps = [], []
    # two hands oscillating around fixed x-positions, a near-static head
    for f in range(F):
        t = f / (F - 1)
        lh_cx = 80 + 10 * np.sin(t * 6)
        rh_cx = 170 + 10 * np.cos(t * 6)
        cy = 140 + 8 * np.sin(t * 3)
        lh = [lh_cx - 18, cy - 18, lh_cx + 18, cy + 18]
        rh = [rh_cx - 18, cy - 18, rh_cx + 18, cy + 18]
        # drop the left hand on a couple of frames to exercise presence handling
        hands = ([rh] if f in (3, 4) else [lh, rh])
        head = None if f == 7 else [110, 40, 150, 80]   # one missing-head frame
        dets.append({"hands": np.array(hands, float),
                     "head": (None if head is None else np.array(head, float))})
        # keypoints per hand (index-aligned with "hands"): cluster in each box
        kf = []
        for b in hands:
            cx, cyy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            pts = np.stack([cx + rng.uniform(-12, 12, 21),
                            cyy + rng.uniform(-12, 12, 21)], axis=1)
            kf.append(pts)
        kps.append(np.array(kf, float))
    geom, _crops = build_clip_features(frames, dets, kps, crop_size=64)
    return {
        "frameW": W, "frameH": H, "geomDim": GEOM_DIM,
        "dets": [{"hands": d["hands"].tolist(),
                  "head": (None if d["head"] is None else d["head"].tolist())}
                 for d in dets],
        "kps": [k.tolist() for k in kps],
        "expectGeom": geom.astype(float).reshape(-1).tolist(),
    }


def main():
    (OUT / "decode_golden.json").write_text(json.dumps(make_decode_golden()))
    (OUT / "geometry_golden.json").write_text(json.dumps(make_geometry_golden()))
    print("wrote fixtures to", OUT)
    for p in OUT.glob("*.json"):
        print(" ", p.name, p.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
