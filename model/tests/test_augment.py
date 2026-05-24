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


def test_clipdataset_two_stream_returns_pair(tmp_path):
    import numpy as np, json, os
    from asl.dataset import ClipDataset
    n = 6
    X = (np.random.default_rng(0).random((n, 16, 112, 112, 3)) * 255).astype(np.uint8)
    Xflow = np.random.default_rng(1).random((n, 16, 112, 112, 2)).astype(np.float16)
    y = np.arange(n) % 3
    part = np.array(["P22"] * n)              # a known train signer
    split = np.array(["train"] * n)
    npz = os.path.join(tmp_path, "c.npz")
    np.savez(npz, X=X, Xflow=Xflow, y=y, participant=part, split=split,
             files=part)
    norm = os.path.join(tmp_path, "n.json")
    json.dump({"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]}, open(norm, "w"))
    fnorm = os.path.join(tmp_path, "n_flow.json")
    json.dump({"mean": [0.0, 0.0], "std": [1.0, 1.0]}, open(fnorm, "w"))
    ds = ClipDataset(npz, "train", norm, train=False, two_stream=True,
                     flow_norm_path=fnorm, signer_splits=None)
    (rgb, flow), label = ds[0]
    assert rgb.shape == (16, 3, 112, 112)
    assert flow.shape == (16, 2, 112, 112)
