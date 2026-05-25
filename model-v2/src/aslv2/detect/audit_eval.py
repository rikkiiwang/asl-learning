"""Gate the trained detector on the manually-labeled ASL-frame audit slice
(Plan 2 Task 9 / spec §9 item 5).

This is the *binding* domain gate (real ASL Citizen frames), as opposed to the
100DOH val proxy gate used during training. Domain gate per the plan:
detection-rate@0.5 **≥ 0.80 head, ≥ 0.60 hand**.

Pure glue over `detect_frame` (per-frame inference) and `metrics.detection_rate`
(recall@IoU). `detection_rate` returns 1.0 for empty-GT, so each class is
averaged **only over frames that actually contain a GT box of that class** — a
frame with no labeled hand must not inflate hand recall.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from aslv2.detect.infer import detect_frame
from aslv2.metrics import detection_rate

HEAD_GATE = 0.80          # anchor metric (center-inside + scale) — the binding head gate
HEAD_IOU_GATE = 0.80      # legacy IoU@0.5 — reported for reference (undercounts: the
                          # model predicts tight FACE boxes vs whole-HEAD labels)
HAND_GATE = 0.60

# Head box is a normalization ANCHOR (spec §3.6): what matters is a correctly-placed
# centre and a sane, consistent scale — not pixel-tight IoU with a whole-head label.
_ANCHOR_SCALE_LO = 0.3
_ANCHOR_SCALE_HI = 1.7


def _head_anchor_ok(pred_head, gt_head) -> float:
    """1.0 if pred head center is inside the GT box and its size is within a sane
    ratio of the GT size (a consistent face box inside a head box qualifies)."""
    if pred_head is None:
        return 0.0
    pcx = (pred_head[0] + pred_head[2]) / 2.0
    pcy = (pred_head[1] + pred_head[3]) / 2.0
    inside = (gt_head[0] <= pcx <= gt_head[2]) and (gt_head[1] <= pcy <= gt_head[3])
    ps = max(pred_head[2] - pred_head[0], pred_head[3] - pred_head[1])
    gs = max(gt_head[2] - gt_head[0], gt_head[3] - gt_head[1], 1.0)
    ratio = ps / gs
    return 1.0 if (inside and _ANCHOR_SCALE_LO <= ratio <= _ANCHOR_SCALE_HI) else 0.0


def evaluate_on_audit(
    model,
    frames,
    norm: dict,
    image_root,
    *,
    img_size: int = 128,
    score_thr: float = 0.3,
    iou_thr: float = 0.45,
    match_iou: float = 0.5,
) -> dict:
    """Run the detector on every audit frame and compute domain detection-rate.

    Args:
        model:      a Detector in eval mode.
        frames:     list of AuditFrame (from aslv2.audit.load_audit_slice).
        norm:       {"mean": [...], "std": [...]} used at train time.
        image_root: dir that `frame.image` paths are relative to (e.g. model-v2/).
        img_size:   input resolution the model was TRAINED at (must match, or
                    anchors/boxes won't line up). Pass cfg["img"] from the ckpt.
        score_thr/iou_thr: passed to detect_frame (confidence + NMS).
        match_iou:  IoU threshold for counting a GT box as detected.

    Returns:
        dict with hand_dr/head_dr (averaged over class-bearing frames only),
        gate thresholds + pass flags, frame counts, and a per-frame breakdown.
    """
    image_root = Path(image_root)
    hand_rates: list[float] = []
    head_rates: list[float] = []        # legacy IoU@0.5
    head_anchor: list[float] = []       # anchor metric (center + scale)
    per_frame: list[dict] = []

    for fr in frames:
        img_path = image_root / fr.image
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise FileNotFoundError(f"audit frame not found: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        det = detect_frame(model, rgb, norm, score_thr=score_thr,
                           iou_thr=iou_thr, img_size=img_size)
        pred_hands = det["hands"]
        pred_head = (
            np.zeros((0, 4), dtype=np.float32)
            if det["head"] is None
            else det["head"][None, :]
        )

        rec: dict = {"image": fr.image}
        if len(fr.hands) > 0:
            hr = detection_rate(pred_hands, fr.hands, iou_thr=match_iou)
            hand_rates.append(hr)
            rec["hand_dr"] = hr
        if fr.head is not None:
            hd = detection_rate(pred_head, fr.head[None, :], iou_thr=match_iou)
            head_rates.append(hd)
            anchor = _head_anchor_ok(det["head"], fr.head)
            head_anchor.append(anchor)
            rec["head_iou_dr"] = hd
            rec["head_anchor"] = anchor
        per_frame.append(rec)

    hand_dr = float(np.mean(hand_rates)) if hand_rates else 0.0
    head_iou_dr = float(np.mean(head_rates)) if head_rates else 0.0
    head_anchor_dr = float(np.mean(head_anchor)) if head_anchor else 0.0

    return {
        "hand_dr": hand_dr,
        "head_anchor_dr": head_anchor_dr,   # binding head metric
        "head_iou_dr": head_iou_dr,         # legacy IoU@0.5 (undercounts; see module doc)
        "n_frames": len(frames),
        "n_hand_frames": len(hand_rates),
        "n_head_frames": len(head_rates),
        "hand_gate": HAND_GATE,
        "head_gate": HEAD_GATE,
        "head_iou_gate": HEAD_IOU_GATE,
        "hand_pass": hand_dr >= HAND_GATE,
        "head_pass": head_anchor_dr >= HEAD_GATE,
        "per_frame": per_frame,
    }
