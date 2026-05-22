import numpy as np
from aslv2.metrics import detection_rate, head_stability, pck


def test_detection_rate_matches_by_iou():
    gt = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], float)
    pred = np.array([[1, 1, 11, 11]], float)            # matches box 0 only
    assert detection_rate(pred, gt, iou_thr=0.5) == 0.5


def test_head_stability_zero_for_static_head():
    heads = np.tile([10, 10, 30, 30], (16, 1)).astype(float)
    assert head_stability(heads) == 0.0                 # no jitter


def test_head_stability_positive_for_jitter():
    heads = np.tile([10, 10, 30, 30], (16, 1)).astype(float)
    heads[::2, 0] += 5                                   # wobble x
    assert head_stability(heads) > 0.0


def test_pck_counts_keypoints_within_threshold():
    gt = np.zeros((21, 2))
    pred = gt.copy()
    pred[0] = [100, 0]                                   # one bad keypoint
    # ref_size 50, thr 0.2 -> tolerance 10px; 20/21 within
    assert abs(pck(pred, gt, ref_size=50.0, thr_frac=0.2) - 20 / 21) < 1e-6
