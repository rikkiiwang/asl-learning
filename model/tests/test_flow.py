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
    # A *textured* patch shifted right by 3 px produces a dominant horizontal flow
    # component. A textureless solid shape can't be tracked by Farneback (the
    # aperture problem — no gradients to follow), so the patch must have texture.
    rng = np.random.default_rng(0)
    patch = rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)   # one fixed patch
    def make(shift):
        img = np.full((64, 64, 3), 128, np.uint8)
        img[20:44, 20 + shift:44 + shift] = patch
        return img
    frames = np.stack([make(0), make(3)], 0)
    flow = compute_flow(frames)
    dx = flow[1, 24:40, 24:40, 0].mean()
    dy = flow[1, 24:40, 24:40, 1].mean()
    assert abs(dx) > 1.5         # clear horizontal motion detected (~3 px shift)
    assert abs(dx) > abs(dy)     # and the horizontal component dominates
