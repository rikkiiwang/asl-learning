"""Tests for KpDataset — keypoint dataset loader."""
import json
import numpy as np
import pytest
import cv2


def test_kpdataset_synthetic(tmp_path):
    """Synthetic 128×128 PNG with 21 keypoints; loader returns (3,64,64) crop
    and (21,2) keypoints in [0,1].  A keypoint at the crop centre maps to ≈[0.5,0.5]."""
    from aslv2.landmark.data import KpDataset

    # Create a 128×128 solid-colour image
    img = np.full((128, 128, 3), 180, dtype=np.uint8)
    img_path = tmp_path / "hand0.png"
    cv2.imwrite(str(img_path), img)

    # Hand box: pixels [20,30,80,100] (x1,y1,x2,y2)
    box = [20.0, 30.0, 80.0, 100.0]   # w=60, h=70 before margin

    # 21 keypoints in pixel coords — all at crop centre (within the box)
    cx_px = (box[0] + box[2]) / 2  # = 50.0
    cy_px = (box[1] + box[3]) / 2  # = 65.0
    kps_px = [[cx_px, cy_px]] * 21  # 21 points at box centre

    manifest = [
        {
            "image": str(img_path),
            "box": box,          # xyxy in original pixel coords
            "keypoints": kps_px, # [[x,y], ...] × 21, pixel coords
        }
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    norm = {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]}
    ds = KpDataset(str(manifest_path), norm=norm, train=False)

    assert len(ds) == 1
    crop, kp = ds[0]

    # Image tensor shape
    assert crop.shape == (3, 64, 64), f"Expected (3,64,64), got {crop.shape}"

    # Keypoints shape and range
    assert kp.shape == (21, 2), f"Expected (21,2), got {kp.shape}"
    assert float(kp.min()) >= 0.0, "kp values must be >= 0"
    assert float(kp.max()) <= 1.0, "kp values must be <= 1"

    # Keypoints at box centre → normalised coords ≈ [0.5, 0.5]
    np.testing.assert_allclose(kp.numpy(), np.full((21, 2), 0.5), atol=0.05,
                               err_msg="Centre keypoints should map to ~[0.5,0.5]")


def test_kpdataset_val_framing_centers_and_spreads(tmp_path):
    """A spread, off-centre hand: val framing (scale 2.0 about the kp bbox) must
    re-centre it and yield keypoints that span a real range (not collapsed)."""
    from aslv2.landmark.data import KpDataset

    img = np.full((200, 200, 3), 150, dtype=np.uint8)
    p = tmp_path / "h.png"
    cv2.imwrite(str(p), img)
    # 21 keypoints on a grid spanning a 40×40 box near the top-left (off-centre)
    kps = [[30 + (i % 5) * 10, 40 + (i // 5) * 10] for i in range(21)]
    mp = tmp_path / "m.json"
    mp.write_text(json.dumps([{"image": str(p), "box": [0, 0, 200, 200], "keypoints": kps}]))

    norm = {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]}
    crop, kp = KpDataset(str(mp), norm=norm, train=False)[0]
    kp = kp.numpy()

    assert crop.shape == (3, 64, 64)
    assert kp.min() >= 0.0 and kp.max() <= 1.0
    # re-centred regardless of original off-centre position
    assert abs(kp[:, 0].mean() - 0.5) < 0.15 and abs(kp[:, 1].mean() - 0.5) < 0.15
    # real spread (scale 2.0 ⇒ hand fills ~half the crop), not a collapsed cluster
    assert kp[:, 0].std() > 0.1 and kp[:, 1].std() > 0.1


def test_kpdataset_data_root_resolves_relative_paths(tmp_path):
    """A relative manifest path + data_root must resolve to the real file."""
    from aslv2.landmark.data import KpDataset

    sub = tmp_path / "freihand" / "training" / "rgb"
    sub.mkdir(parents=True)
    cv2.imwrite(str(sub / "00000000.jpg"), np.full((128, 128, 3), 180, dtype=np.uint8))

    manifest = [{
        "image": "freihand/training/rgb/00000000.jpg",   # relative, like the real manifest
        "box": [20.0, 30.0, 80.0, 100.0],
        "keypoints": [[50.0, 65.0]] * 21,
    }]
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text(json.dumps(manifest))

    norm = {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]}
    # without data_root the relative path can't be read → FileNotFoundError
    with pytest.raises(FileNotFoundError):
        KpDataset(str(manifest_path), norm=norm, train=False)[0]

    # with data_root it resolves and returns a valid crop
    ds = KpDataset(str(manifest_path), norm=norm, train=False, data_root=str(tmp_path))
    crop, kp = ds[0]
    assert crop.shape == (3, 64, 64) and kp.shape == (21, 2)
