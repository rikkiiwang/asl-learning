import numpy as np
from aslv2.detect.anchors import make_anchors

def test_anchor_count_and_squareness():
    a = make_anchors(img=128, stride=16, scales=(32, 64, 96))
    assert a.shape == (8 * 8 * 3, 4)            # 192 anchors, xyxy
    w = a[:, 2] - a[:, 0]; h = a[:, 3] - a[:, 1]
    np.testing.assert_allclose(w, h)            # square
    # first cell center is at (8,8); smallest anchor spans 8±16
    np.testing.assert_allclose(a[0], [8 - 16, 8 - 16, 8 + 16, 8 + 16])
