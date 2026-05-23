"""Tests for DetDataset — detection dataset loader."""
import json
import numpy as np
import pytest
import cv2

def test_detdataset_synthetic(tmp_path):
    """Synthetic 256×256 PNG; box [0,0,128,128] should map to [0,0,64,64] at 128²."""
    from aslv2.detect.data import DetDataset

    # Create a 256×256 white PNG
    img = np.full((256, 256, 3), 200, dtype=np.uint8)
    img_path = tmp_path / "frame0.png"
    cv2.imwrite(str(img_path), img)

    # Unified manifest: box at top-left quarter in 256² space
    manifest = [
        {
            "image": str(img_path),
            "boxes": [[0.0, 0.0, 128.0, 128.0]],
            "labels": [0],
        }
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    norm = {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]}
    ds = DetDataset(str(manifest_path), norm=norm, train=False)

    assert len(ds) == 1
    tensor, boxes, labels = ds[0]

    # Tensor shape
    assert tensor.shape == (3, 128, 128), f"Expected (3,128,128), got {tensor.shape}"

    # Box should scale from 256² -> 128²: [0,0,128,128] -> [0,0,64,64]
    assert boxes.shape == (1, 4), f"Expected (1,4) boxes, got {boxes.shape}"
    np.testing.assert_allclose(boxes[0], [0.0, 0.0, 64.0, 64.0], atol=1.0)

    # Labels
    assert labels.shape == (1,)
    assert int(labels[0]) == 0
