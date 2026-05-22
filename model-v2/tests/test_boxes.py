import numpy as np
from aslv2.boxes import iou, nms


def test_iou_half_overlap():
    a = np.array([0.0, 0.0, 2.0, 2.0])
    b = np.array([1.0, 0.0, 3.0, 2.0])   # overlap area 2, union 6
    assert abs(iou(a, b) - (2.0 / 6.0)) < 1e-6


def test_iou_disjoint_is_zero():
    a = np.array([0.0, 0.0, 1.0, 1.0])
    b = np.array([5.0, 5.0, 6.0, 6.0])
    assert iou(a, b) == 0.0


def test_nms_suppresses_overlapping_keeps_best():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]], float)
    scores = np.array([0.9, 0.8, 0.7])
    keep = nms(boxes, scores, iou_thr=0.5)
    assert keep == [0, 2]            # box 1 suppressed by box 0; far box kept
