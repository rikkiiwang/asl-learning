import numpy as np
from aslv2.detect.anchors import make_anchors
from aslv2.detect.encode import encode_targets, decode

def test_encode_assigns_best_anchor_positive():
    anchors = make_anchors()
    gt = np.array([[0, 0, 64, 64]], float); labels = np.array([0])   # one hand
    cls_t, box_t, pos = encode_targets(anchors, gt, labels, iou_pos=0.5)
    assert pos.sum() >= 1                          # at least one positive anchor
    assert set(np.unique(cls_t[pos])) <= {0}       # positives labelled hand(0)
    assert (cls_t[~pos] == -1).all()               # background = -1 (ignored idx)

def test_decode_roundtrip_recovers_box():
    anchors = make_anchors()
    gt = np.array([[10, 12, 70, 78]], float); labels = np.array([1])
    cls_t, box_t, pos = encode_targets(anchors, gt, labels)
    # build perfect predictions: deltas = box_t, score=+inf on positives
    scores = np.full((len(anchors), 2), -9.0); scores[pos, 1] = 9.0
    boxes, scs, lbl = decode(anchors, box_t, scores, score_thr=0.5, iou_thr=0.5)
    assert len(boxes) == 1 and lbl[0] == 1
    np.testing.assert_allclose(boxes[0], gt[0], atol=1.0)
