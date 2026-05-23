"""TDD tests for the FreiHAND adapter.

Tests:
  1. project_2d — pure function; verified against a hand-computable example.
  2. adapt_freihand — builds KpDataset-format records from a tiny synthetic
     FreiHAND directory (2 fake images + 1-entry xyz/K JSON).

No real FreiHAND data is used or downloaded here.
"""
import json
import os
import tempfile

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1.  project_2d — hand-computable unit test
# ---------------------------------------------------------------------------

class TestProject2D:
    """Verify project_2d with a known camera + xyz configuration.

    Camera: K = identity (f_x=f_y=1, c_x=c_y=0).
    Points: xyz = [[1, 2, 4]] (one landmark).

    Projection: uv = K @ xyz.T → uv[:2] / uv[2]
        u = 1/4 = 0.25,  v = 2/4 = 0.5

    With all 21 keypoints set to the same value we can assert all == 0.25, 0.5.
    """

    def test_identity_camera_known_output(self):
        from aslv2.landmark.freihand import project_2d

        # 21 identical 3-D points
        xyz = np.tile(np.array([1.0, 2.0, 4.0]), (21, 1))   # (21, 3)
        K   = np.eye(3, dtype=np.float64)

        uv = project_2d(xyz, K)    # (21, 2)

        assert uv.shape == (21, 2), f"Expected (21,2), got {uv.shape}"
        np.testing.assert_allclose(uv[:, 0], 0.25, atol=1e-6,
                                   err_msg="u-coordinate wrong")
        np.testing.assert_allclose(uv[:, 1], 0.50, atol=1e-6,
                                   err_msg="v-coordinate wrong")

    def test_scaled_camera_matrix(self):
        """K with focal length 200 and principal point (100, 80)."""
        from aslv2.landmark.freihand import project_2d

        fx, fy, cx, cy = 200.0, 200.0, 100.0, 80.0
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=np.float64)

        # Single 3-D point at (X=0, Y=0, Z=5) → should project to (cx, cy)
        xyz = np.zeros((21, 3), dtype=np.float64)
        xyz[:, 2] = 5.0   # depth = 5

        uv = project_2d(xyz, K)

        np.testing.assert_allclose(uv[:, 0], cx, atol=1e-6,
                                   err_msg="principal-point x wrong")
        np.testing.assert_allclose(uv[:, 1], cy, atol=1e-6,
                                   err_msg="principal-point y wrong")

    def test_returns_float32(self):
        from aslv2.landmark.freihand import project_2d

        xyz = np.random.randn(21, 3).astype(np.float32)
        xyz[:, 2] = np.abs(xyz[:, 2]) + 1.0   # positive depth
        K = np.eye(3, dtype=np.float32)

        uv = project_2d(xyz, K)
        assert uv.dtype == np.float32, f"Expected float32, got {uv.dtype}"


# ---------------------------------------------------------------------------
# 2.  adapt_freihand — synthetic FreiHAND directory fixture
# ---------------------------------------------------------------------------

def _make_fake_freihand_dir(tmp_path: str, n_base: int = 2) -> str:
    """Create a minimal synthetic FreiHAND directory.

    Layout mirrors real FreiHAND v2:
        training/rgb/%08d.jpg     (n_base images; 4 bg variants each)
        training_xyz.json         (n_base × 21 × 3 joints)
        training_K.json           (n_base × 3×3 intrinsics)
    """
    import cv2

    rgb_dir = os.path.join(tmp_path, "training", "rgb")
    os.makedirs(rgb_dir, exist_ok=True)

    n_images = n_base * 4   # 4 background versions per base
    for i in range(n_images):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        img[:, :] = (i * 30 % 256, 100, 200)
        cv2.imwrite(os.path.join(rgb_dir, f"{i:08d}.jpg"), img)

    # xyz: n_base entries, each (21, 3) — depth 0.5 so projection stays safe
    rng  = np.random.default_rng(0)
    xyz  = rng.standard_normal((n_base, 21, 3)).tolist()
    # ensure positive depth (Z component)
    for entry in xyz:
        for kp in entry:
            kp[2] = abs(kp[2]) + 0.5

    # K: n_base identity-ish 3×3 with fx=fy=150, cx=cy=112
    K_base = [[150.0, 0.0, 112.0],
              [0.0, 150.0, 112.0],
              [0.0,   0.0,   1.0]]
    Ks = [K_base for _ in range(n_base)]

    with open(os.path.join(tmp_path, "training_xyz.json"), "w") as f:
        json.dump(xyz, f)
    with open(os.path.join(tmp_path, "training_K.json"), "w") as f:
        json.dump(Ks, f)

    return tmp_path


class TestAdaptFreihand:
    def test_record_count_all_variants(self, tmp_path):
        """With 2 base images × 4 variants = 8 images → 8 records."""
        from aslv2.landmark.freihand import adapt_freihand

        root = _make_fake_freihand_dir(str(tmp_path), n_base=2)
        records = adapt_freihand(root)

        assert len(records) == 8, f"Expected 8 records, got {len(records)}"

    def test_record_schema(self, tmp_path):
        """Each record must have 'image', 'box', and 'keypoints' keys."""
        from aslv2.landmark.freihand import adapt_freihand

        root = _make_fake_freihand_dir(str(tmp_path), n_base=2)
        records = adapt_freihand(root)

        for rec in records:
            assert "image" in rec, "'image' key missing"
            assert "box" in rec, "'box' key missing"
            assert "keypoints" in rec, "'keypoints' key missing"

    def test_box_is_full_frame(self, tmp_path):
        """Box must be [0, 0, 224, 224] for every record."""
        from aslv2.landmark.freihand import adapt_freihand

        root = _make_fake_freihand_dir(str(tmp_path), n_base=2)
        records = adapt_freihand(root)

        for rec in records:
            assert rec["box"] == [0, 0, 224, 224], \
                f"Unexpected box: {rec['box']}"

    def test_keypoints_shape(self, tmp_path):
        """Each record's keypoints must be a list of 21 [x, y] pairs."""
        from aslv2.landmark.freihand import adapt_freihand

        root = _make_fake_freihand_dir(str(tmp_path), n_base=2)
        records = adapt_freihand(root)

        for rec in records:
            kps = rec["keypoints"]
            assert len(kps) == 21, f"Expected 21 keypoints, got {len(kps)}"
            for kp in kps:
                assert len(kp) == 2, f"Keypoint must have 2 coords, got {len(kp)}"

    def test_image_paths_relative_to_data_root(self, tmp_path):
        """When data_root is set, image paths should be relative to it."""
        from aslv2.landmark.freihand import adapt_freihand

        root = _make_fake_freihand_dir(str(tmp_path), n_base=2)
        records = adapt_freihand(root, data_root=root)

        # All paths must be relative (not absolute)
        for rec in records:
            assert not os.path.isabs(rec["image"]), \
                f"Expected relative path, got: {rec['image']}"

    def test_max_images_cap(self, tmp_path):
        """max_images limits the number of records returned."""
        from aslv2.landmark.freihand import adapt_freihand

        root = _make_fake_freihand_dir(str(tmp_path), n_base=2)
        records = adapt_freihand(root, max_images=3)

        assert len(records) == 3, f"Expected 3 records with cap, got {len(records)}"

    def test_label_index_wraps_correctly(self, tmp_path):
        """Images 0..3 all use xyz label 0; images 4..7 use label 1 (i % n_base)."""
        from aslv2.landmark.freihand import adapt_freihand

        root = _make_fake_freihand_dir(str(tmp_path), n_base=2)
        records = adapt_freihand(root)

        import json as _json
        xyz_all = _json.load(open(os.path.join(root, "training_xyz.json")))
        K_all   = _json.load(open(os.path.join(root, "training_K.json")))
        from aslv2.landmark.freihand import project_2d

        n_base = 2   # matches fixture
        for img_idx, rec in enumerate(records):
            label_idx = img_idx % n_base
            xyz = np.array(xyz_all[label_idx], dtype=np.float32)
            K   = np.array(K_all[label_idx],   dtype=np.float32)
            expected_uv = project_2d(xyz, K).tolist()

            np.testing.assert_allclose(
                np.array(rec["keypoints"]),
                np.array(expected_uv),
                atol=1e-4,
                err_msg=f"Keypoints wrong for image index {img_idx}",
            )
