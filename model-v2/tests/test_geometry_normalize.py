import numpy as np
from aslv2.geometry import normalize_geometry, GEOM_DIM


def _inputs():
    kps = np.zeros((2, 21, 2))
    kps[0] = np.tile([60.0, 50.0], (21, 1))     # left-hand keypoints clustered
    kps[1] = np.tile([140.0, 50.0], (21, 1))    # right-hand keypoints
    present = np.array([1.0, 1.0])
    head = np.array([90.0, 30.0, 110.0, 50.0])  # center (100,40), size 20
    return kps, present, head


def test_output_has_fixed_dim():
    kps, present, head = _inputs()
    v = normalize_geometry(kps, present, head, head_present=1.0)
    assert v.shape == (GEOM_DIM,)
    assert GEOM_DIM == 93


def test_translation_invariance():
    kps, present, head = _inputs()
    v1 = normalize_geometry(kps, present, head, 1.0)
    d = np.array([37.0, -12.0])
    v2 = normalize_geometry(kps + d, present, head + np.array([*d, *d]), 1.0)
    np.testing.assert_allclose(v1, v2, atol=1e-6)


def test_scale_invariance():
    kps, present, head = _inputs()
    v1 = normalize_geometry(kps, present, head, 1.0)
    v2 = normalize_geometry(kps * 3.0, present, head * 3.0, 1.0)
    np.testing.assert_allclose(v1, v2, atol=1e-6)


def test_missing_slot_is_zeroed_and_flagged():
    kps, present, head = _inputs()
    kps[1] = np.nan
    present[1] = 0.0
    v = normalize_geometry(kps, present, head, 1.0)
    # right-hand keypoint block (slot 1: indices 42..84) must be all zeros
    assert np.all(v[42:84] == 0.0)
    assert not np.isnan(v).any()


def test_present_slot_with_nan_keypoints_never_leaks_nan():
    kps, present, head = _inputs()
    present[0] = 1.0
    kps[1] = np.nan
    present[1] = 1.0                       # present, but keypoints unknown (occluded)
    v = normalize_geometry(kps, present, head, 1.0)
    assert not np.isnan(v).any()
