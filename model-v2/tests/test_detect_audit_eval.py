"""Tests for the ASL-audit domain gate aggregation (Plan 2 Task 9).

Accuracy of a randomly-initialised detector is irrelevant here — these lock the
*aggregation contract*: which frames count toward each class, the output keys,
and that a missing image fails loudly.
"""
import cv2
import numpy as np
import pytest

from aslv2.audit import AuditFrame
from aslv2.detect.audit_eval import (
    HAND_GATE, HEAD_GATE, _head_anchor_ok, evaluate_on_audit,
)
from aslv2.detect.model import Detector


def test_head_anchor_accepts_face_box_inside_head_label():
    gt_head = np.array([100, 50, 200, 250], float)        # whole head, 100×200
    # a tight FACE box centered inside, ~0.6× the head size → valid anchor
    face = np.array([130, 110, 180, 210], float)
    assert _head_anchor_ok(face, gt_head) == 1.0
    # a box whose center is outside the head → rejected
    off = np.array([10, 10, 60, 60], float)
    assert _head_anchor_ok(off, gt_head) == 0.0
    # a wildly oversized box (center inside but 3× too big) → rejected
    huge = np.array([0, 0, 600, 600], float)
    assert _head_anchor_ok(huge, gt_head) == 0.0
    # no prediction → 0
    assert _head_anchor_ok(None, gt_head) == 0.0

NORM = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def _model():
    m = Detector(n_classes=2, n_anchors=3, width=32)
    m.eval()
    return m


def test_class_rates_average_only_over_class_bearing_frames(tmp_path):
    for name in ["a.png", "b.png"]:
        cv2.imwrite(str(tmp_path / name), (np.random.rand(120, 160, 3) * 255).astype(np.uint8))

    frames = [
        AuditFrame(
            "a.png",
            np.array([[10, 10, 50, 50], [60, 60, 100, 100]], float),
            np.array([20, 5, 60, 45], float),
            None,
        ),
        AuditFrame("b.png", np.zeros((0, 4)), None, None),  # unlabeled — must not count
    ]
    res = evaluate_on_audit(_model(), frames, NORM, tmp_path)

    assert res["n_frames"] == 2
    assert res["n_hand_frames"] == 1   # only frame a contributes to hand_dr
    assert res["n_head_frames"] == 1
    assert 0.0 <= res["hand_dr"] <= 1.0
    assert 0.0 <= res["head_anchor_dr"] <= 1.0
    assert 0.0 <= res["head_iou_dr"] <= 1.0
    assert res["hand_gate"] == HAND_GATE and res["head_gate"] == HEAD_GATE
    assert isinstance(res["hand_pass"], bool) and isinstance(res["head_pass"], bool)
    assert len(res["per_frame"]) == 2


def test_missing_image_raises(tmp_path):
    frames = [AuditFrame("nope.png", np.zeros((0, 4)), None, None)]
    with pytest.raises(FileNotFoundError):
        evaluate_on_audit(_model(), frames, NORM, tmp_path)
