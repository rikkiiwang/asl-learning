import numpy as np
from asl.preprocess import compute_flow


def test_flow_shape_and_zero_first_frame():
    frames = (np.random.default_rng(0).random((16, 64, 64, 3)) * 255).astype(np.uint8)
    flow = compute_flow(frames)
    assert flow.shape == (16, 64, 64, 2)
    assert flow.dtype == np.float32
    assert np.allclose(flow[0], 0.0)            # first frame has no predecessor


def test_flow_zero_for_static_clip():
    frame = (np.random.default_rng(1).random((64, 64, 3)) * 255).astype(np.uint8)
    frames = np.repeat(frame[None], 8, axis=0)
    flow = compute_flow(frames)
    assert np.abs(flow).max() < 0.5            # no motion -> ~zero flow


def test_flow_detects_horizontal_shift():
    # A vertical bar shifted right by 4 px between frames -> positive dx where the bar is.
    base = np.zeros((64, 64, 3), dtype=np.uint8)
    base[:, 20:24] = 255
    shifted = np.zeros((64, 64, 3), dtype=np.uint8)
    shifted[:, 24:28] = 255
    frames = np.stack([base, shifted], 0)
    flow = compute_flow(frames)
    # mean dx over the central region should be clearly positive
    assert flow[1, 20:44, 18:30, 0].mean() > 0.3
