import numpy as np
from aslv2.detect.anchors import make_anchors
from aslv2.detect.encode import encode_targets, decode

def test_encode_assigns_best_anchor_positive():
    anchors = make_anchors()
    gt = np.array([[0, 0, 64, 64]], float); labels = np.array([0])   # one hand
    cls_tgt, box_t, pos, valid = encode_targets(anchors, gt, labels, iou_pos=0.5)
    # positives
    assert pos.sum() >= 1                                  # at least one positive anchor
    assert (cls_tgt[pos].argmax(1) == 0).all()             # positives are one-hot at class 0
    assert (cls_tgt[pos].sum(1) == 1).all()                # each positive row sums to 1
    assert valid[pos].all()                                 # positives are always valid
    # clear backgrounds: valid & ~pos => all-zero cls_tgt
    assert (cls_tgt[valid & ~pos] == 0).all()

def test_decode_roundtrip_recovers_box():
    anchors = make_anchors()
    gt = np.array([[10, 12, 70, 78]], float); labels = np.array([1])
    cls_tgt, box_t, pos, valid = encode_targets(anchors, gt, labels)
    # build perfect predictions: deltas = box_t, score=+inf on positives for GT class
    scores = np.full((len(anchors), 2), -9.0); scores[pos, 1] = 9.0
    boxes, scs, lbl = decode(anchors, box_t, scores, score_thr=0.5, iou_thr=0.5)
    assert len(boxes) == 1 and lbl[0] == 1
    np.testing.assert_allclose(boxes[0], gt[0], atol=1.0)


def test_encode_targets_degenerate_gt_box_produces_finite_deltas():
    """A GT box with zero width must not produce -inf/-nan box deltas (log(0) guard)."""
    import warnings
    anchors = make_anchors()
    # GT box [10, 10, 10, 30] — zero width (x1==x2), non-zero height
    gt = np.array([[10.0, 10.0, 10.0, 40.0]], float)
    labels = np.array([0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")   # any RuntimeWarning (divide-by-zero) → error
        cls_tgt, box_t, pos, valid = encode_targets(anchors, gt, labels)
    assert np.isfinite(box_t).all(), (
        f"box_t contains non-finite values: {box_t[~np.isfinite(box_t)]}"
    )
