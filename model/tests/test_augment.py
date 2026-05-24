import numpy as np
from asl.dataset import augment_clip


def test_augment_preserves_shape_dtype_range():
    rng = np.random.default_rng(0)
    clip = rng.random((16, 100, 100, 3)).astype(np.float32)
    out = augment_clip(clip, size=112, rng=np.random.default_rng(0))
    assert out.shape == (16, 112, 112, 3)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_augment_is_deterministic_under_seed():
    clip = np.random.default_rng(1).random((16, 100, 100, 3)).astype(np.float32)
    a = augment_clip(clip, 112, rng=np.random.default_rng(42))
    b = augment_clip(clip, 112, rng=np.random.default_rng(42))
    assert np.allclose(a, b)


def test_augment_same_geometric_transform_across_frames():
    frame = np.random.default_rng(2).random((100, 100, 3)).astype(np.float32)
    clip = np.repeat(frame[None], 16, axis=0)
    out = augment_clip(clip, 112, rng=np.random.default_rng(7))
    assert np.allclose(out[0], out[1]) and np.allclose(out[0], out[15])
