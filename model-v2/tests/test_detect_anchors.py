import numpy as np
from aslv2.detect.anchors import anchors_for, make_anchors, scales_for

def test_anchor_count_and_squareness():
    a = make_anchors(img=128, stride=16, scales=(32, 64, 96))
    assert a.shape == (8 * 8 * 3, 4)            # 192 anchors, xyxy
    w = a[:, 2] - a[:, 0]; h = a[:, 3] - a[:, 1]
    np.testing.assert_allclose(w, h)            # square
    # first cell center is at (8,8); smallest anchor spans 8±16
    np.testing.assert_allclose(a[0], [8 - 16, 8 - 16, 8 + 16, 8 + 16])


def test_scales_track_resolution():
    assert scales_for(128) == (32, 64, 96)               # baseline unchanged
    np.testing.assert_allclose(scales_for(192), (48, 96, 144))  # 1.5× input ⇒ 1.5× scales


def test_anchors_for_grid_grows_with_resolution():
    a128 = anchors_for(128)
    a192 = anchors_for(192)
    assert a128.shape == (8 * 8 * 3, 4)      # baseline matches make_anchors
    assert a192.shape == (12 * 12 * 3, 4)    # 192/16 = 12 grid
    np.testing.assert_allclose(a192[0, 2] - a192[0, 0], 48.0)  # smallest span 1.5×
