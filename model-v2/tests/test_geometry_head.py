import numpy as np
from aslv2.geometry import resolve_head_anchor


def test_median_anchor_ignores_jitter_and_reports_present():
    heads = np.array([[10, 10, 30, 40],
                      [11, 9, 31, 41],
                      [9, 11, 29, 39]], float)            # 3 valid frames
    anchor, present = resolve_head_anchor(heads, frame_wh=(640, 480))
    assert present == 1.0
    np.testing.assert_allclose(anchor, [10, 10, 30, 40])  # per-coord median


def test_missing_frames_fall_back_to_median_of_valid():
    heads = np.array([[10, 10, 30, 40],
                      [np.nan] * 4,
                      [12, 12, 32, 42]], float)
    anchor, present = resolve_head_anchor(heads, frame_wh=(640, 480))
    assert present == 1.0
    np.testing.assert_allclose(anchor, [11, 11, 31, 41])  # median of the 2 valid


def test_all_missing_falls_back_to_centered_body_scale_anchor():
    heads = np.full((16, 4), np.nan)
    anchor, present = resolve_head_anchor(heads, frame_wh=(640, 480))
    assert present == 0.0
    cx, cy = (anchor[0] + anchor[2]) / 2, (anchor[1] + anchor[3]) / 2
    assert abs(cx - 320) < 1e-6 and abs(cy - 240) < 1e-6   # frame center
    assert (anchor[3] - anchor[1]) > 0                     # has a body-scale size
