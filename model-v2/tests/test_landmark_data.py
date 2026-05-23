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
