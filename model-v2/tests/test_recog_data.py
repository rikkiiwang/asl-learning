"""Tests for the signer-held-out recognizer dataset (Plan 4 Task 3)."""
import numpy as np

from aslv2.recog.data import RecogDataset


def _make_cache(tmp_path):
    # 6 clips across 3 participants; tiny crops to keep the test light
    geom = np.random.rand(6, 16, 93).astype(np.float32)
    crops = np.random.rand(6, 16, 2, 3, 8, 8).astype(np.float32)
    y = np.array([0, 1, 2, 0, 1, 2])
    part = np.array(["P1", "P1", "P2", "P2", "P3", "P3"])
    p = tmp_path / "cache.npz"
    np.savez(p, geom=geom, crops=crops, y=y, participant=part)
    return str(p)


def test_splits_are_signer_disjoint_and_shaped(tmp_path):
    cache = _make_cache(tmp_path)
    policy = {"P1": "train", "P2": "val", "P3": "test"}

    tr = RecogDataset(cache, "train", policy)
    va = RecogDataset(cache, "val", policy)
    te = RecogDataset(cache, "test", policy)

    assert len(tr) == 2 and len(va) == 2 and len(te) == 2
    g, c, y = tr[0]
    assert tuple(g.shape) == (16, 93)
    assert tuple(c.shape) == (16, 2, 3, 8, 8)
    assert isinstance(y, int)

    # no participant appears in two splits
    assert tr.participants() & va.participants() == set()
    assert tr.participants() & te.participants() == set()
    assert va.participants() & te.participants() == set()


def test_train_jitter_perturbs_geometry_not_presence_flags(tmp_path):
    cache = _make_cache(tmp_path)
    policy = {"P1": "train", "P2": "val", "P3": "test"}
    ds = RecogDataset(cache, "train", policy, train=True, kp_jitter=0.05)
    g0, _, _ = ds[0]
    raw = ds._geom[0]
    # geometric dims changed, the 3 presence/head flags (last 3) are untouched
    assert not np.allclose(g0.numpy()[:, :90], raw[:, :90])
    np.testing.assert_allclose(g0.numpy()[:, 90:], raw[:, 90:])
