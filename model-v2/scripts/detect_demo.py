"""Visualize detector output — on still images or a live webcam.

Two modes:
  # annotate still images (e.g. the audit frames) and save them
  .venv/bin/python scripts/detect_demo.py --images artifacts/audit/frames --limit 6 \
      --out artifacts/audit/viz

  # live webcam (run locally in your own terminal — needs camera permission)
  .venv/bin/python scripts/detect_demo.py            # opens a window; press q to quit
  .venv/bin/python scripts/detect_demo.py --mirror   # self-view (display mirrored)

Drawing:
  blue  = predicted HEAD box (+ center dot)   green = predicted HAND box(es)
  yellow line = head-center → hand-center (the relationship Stage-2 normalizes)
  red (still mode only) = your ground-truth labels from asl_audit_slice.json,
       so you can see the model's tight face box sitting inside your head label.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Make the script runnable without an editable install (the `pip install -e`
# .pth occasionally drops off sys.path on this space-containing path).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aslv2.detect.infer import detect_frame
from aslv2.detect.model import Detector
from aslv2.landmark.infer import landmark_hand, map_to_frame
from aslv2.landmark.model import Landmark

BLUE, GREEN, YELLOW, RED = (255, 160, 0), (0, 220, 0), (0, 255, 255), (60, 60, 255)
KP_PT, KP_LINE = (0, 165, 255), (220, 220, 220)   # orange points, light edges

# Inference framing — MUST match KpDataset's val framing (_VAL_SCALE): a square
# window = _LM_FRAME_SCALE × the hand box, centred on it, so the hand occupies
# the same fraction of the crop the landmark model trained on.
_LM_FRAME_SCALE = 2.0

# MediaPipe/FreiHAND 21-keypoint skeleton
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (9, 10), (10, 11), (11, 12),            # middle
    (13, 14), (14, 15), (15, 16),           # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (5, 9), (9, 13), (13, 17),              # palm
]


def load_model(ckpt_path: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    m = Detector(n_classes=2, n_anchors=cfg["n_anchors"], width=cfg["width"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, cfg


def load_landmark(ckpt_path: str):
    """Load the Stage-1.5 landmark model; returns (model, norm) or (None, None)."""
    if not ckpt_path:
        return None, None
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    m = Landmark(width=cfg["width"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    print(f"landmark model loaded (width={cfg['width']}) — overlaying 21 keypoints")
    return m, cfg["norm"]


def _draw_landmarks(img_draw, src_bgr, box, lm_model, lm_norm):
    """Frame the hand in a square window (= _LM_FRAME_SCALE × box, centred), run
    the landmark model, and draw the 21 keypoints + skeleton in frame coords.

    Uses an affine warp so windows extending past the image edge are reflect-
    padded — matching how KpDataset builds val crops.
    """
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    win = max(x2 - x1, y2 - y1) * _LM_FRAME_SCALE
    if win < 2:
        return
    wx1, wy1, wx2, wy2 = cx - win / 2, cy - win / 2, cx + win / 2, cy + win / 2

    s = 128.0 / win
    M = np.array([[s, 0, -wx1 * s], [0, s, -wy1 * s]], dtype=np.float32)
    crop_bgr = cv2.warpAffine(src_bgr, M, (128, 128),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    kp01 = landmark_hand(lm_model, crop_rgb, lm_norm)          # (21,2) in [0,1]
    kp = map_to_frame(kp01, [wx1, wy1, wx2, wy2]).astype(int)  # frame pixels
    for a, b in HAND_EDGES:
        cv2.line(img_draw, tuple(kp[a]), tuple(kp[b]), KP_LINE, 1, cv2.LINE_AA)
    for x, y in kp:
        cv2.circle(img_draw, (x, y), 2, KP_PT, -1)


def _box(img, b, color, label=None, thick=2):
    x1, y1, x2, y2 = [int(v) for v in b]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    if label:
        cv2.putText(img, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def _center(b):
    return (int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2))


def annotate(frame_bgr, det, gt=None, lm_model=None, lm_norm=None):
    """Draw predictions (and optional GT) onto a BGR frame; returns it.

    If lm_model is given, each hand box is filled with 21 landmark keypoints.
    """
    img = frame_bgr.copy()
    # ground truth (still mode): thin red
    if gt is not None:
        if gt.get("head") is not None:
            _box(img, gt["head"], RED, "head (label)", 1)
        for h in gt.get("hands", []):
            _box(img, h, RED, None, 1)

    def _hand(hb):
        _box(img, hb, GREEN, "hand")
        if lm_model is not None:
            _draw_landmarks(img, frame_bgr, hb, lm_model, lm_norm)

    head = det["head"]
    if head is not None:
        _box(img, head, BLUE, "head (pred)")
        hc = _center(head)
        cv2.circle(img, hc, 4, BLUE, -1)
        hw = max(head[2] - head[0], head[3] - head[1])
        for hb in det["hands"]:
            _hand(hb)
            kc = _center(hb)
            cv2.line(img, hc, kc, YELLOW, 1)
            # normalized offset in head-size units (what Stage-2 sees)
            dx = (kc[0] - hc[0]) / hw
            dy = (kc[1] - hc[1]) / hw
            cv2.putText(img, f"({dx:+.1f},{dy:+.1f})", (kc[0] + 4, kc[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1, cv2.LINE_AA)
    else:
        for hb in det["hands"]:
            _hand(hb)
    return img


def run_images(model, cfg, images_dir, out_dir, limit, score_thr, slice_path,
               lm_model=None, lm_norm=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # GT lookup by basename
    gt_by_name = {}
    if slice_path and Path(slice_path).exists():
        for e in json.loads(Path(slice_path).read_text()):
            gt_by_name[Path(e["image"]).name] = e

    paths = sorted(Path(images_dir).glob("*.png"))[:limit]
    saved = []
    for p in paths:
        bgr = cv2.imread(str(p))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        det = detect_frame(model, rgb, cfg["norm"], score_thr=score_thr, img_size=cfg["img"])
        img = annotate(bgr, det, gt=gt_by_name.get(p.name), lm_model=lm_model, lm_norm=lm_norm)
        dst = out / f"viz_{p.name}"
        cv2.imwrite(str(dst), img)
        saved.append(str(dst))
        nh = 0 if det["head"] is None else 1
        print(f"{p.name}: {len(det['hands'])} hand(s), {nh} head -> {dst}")
    print(f"\nwrote {len(saved)} annotated frames to {out}/")


def run_webcam(model, cfg, cam, score_thr, mirror, lm_model=None, lm_norm=None):
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        raise SystemExit(f"could not open webcam {cam} (grant camera permission to your terminal)")
    print("webcam running — press 'q' in the window to quit")
    t0, n = time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        det = detect_frame(model, rgb, cfg["norm"], score_thr=score_thr, img_size=cfg["img"])
        img = annotate(frame, det, lm_model=lm_model, lm_norm=lm_norm)
        n += 1
        fps = n / (time.time() - t0)
        cv2.putText(img, f"{fps:4.1f} FPS  img={cfg['img']} w={cfg['width']}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        if mirror:
            img = cv2.flip(img, 1)
        cv2.imshow("Constellation detector (q=quit)", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Visualize detector boxes on stills or webcam.")
    ap.add_argument("--ckpt", default="artifacts/checkpoints/detector/best.pt")
    ap.add_argument("--landmark-ckpt", default=None,
                    help="Stage-1.5 landmark best.pt — if given, overlay 21 keypoints per hand")
    ap.add_argument("--images", default=None, help="dir of PNGs to annotate (still mode)")
    ap.add_argument("--out", default="artifacts/audit/viz")
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--slice", default="artifacts/audit/asl_audit_slice.json",
                    help="GT slice to overlay in still mode")
    ap.add_argument("--cam", type=int, default=0, help="webcam index (webcam mode)")
    ap.add_argument("--mirror", action="store_true", help="mirror the display (self-view)")
    ap.add_argument("--score-thr", type=float, default=0.3)
    args = ap.parse_args()

    model, cfg = load_model(args.ckpt)
    lm_model, lm_norm = load_landmark(args.landmark_ckpt)
    if args.images:
        run_images(model, cfg, args.images, args.out, args.limit, args.score_thr,
                   args.slice, lm_model=lm_model, lm_norm=lm_norm)
    else:
        run_webcam(model, cfg, args.cam, args.score_thr, args.mirror,
                   lm_model=lm_model, lm_norm=lm_norm)


if __name__ == "__main__":
    main()
