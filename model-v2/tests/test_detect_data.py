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


def test_detdataset_relative_paths(tmp_path):
    """data_root + relative image path resolves to the correct file."""
    from aslv2.detect.data import DetDataset

    # Make a sub-directory under tmp_path to simulate data_root layout
    sub = tmp_path / "subdir"
    sub.mkdir()
    img = np.full((64, 64, 3), 100, dtype=np.uint8)
    img_path = sub / "frame.png"
    cv2.imwrite(str(img_path), img)

    # Manifest uses relative path ("subdir/frame.png"); data_root = tmp_path
    manifest = [
        {
            "image": "subdir/frame.png",
            "boxes": [[0.0, 0.0, 32.0, 32.0]],
            "labels": [1],
        }
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    norm = {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]}
    ds = DetDataset(str(manifest_path), norm=norm, train=False, data_root=str(tmp_path))

    assert len(ds) == 1
    tensor, boxes, labels = ds[0]
    assert tensor.shape == (3, 128, 128)
    assert int(labels[0]) == 1


def test_detdataset_absolute_path_ignores_data_root(tmp_path):
    """Absolute image paths work regardless of data_root (backward-compatible)."""
    from aslv2.detect.data import DetDataset

    img = np.full((64, 64, 3), 50, dtype=np.uint8)
    img_path = tmp_path / "abs_frame.png"
    cv2.imwrite(str(img_path), img)

    manifest = [
        {
            "image": str(img_path),  # absolute
            "boxes": [[0.0, 0.0, 32.0, 32.0]],
            "labels": [0],
        }
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    norm = {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]}
    # data_root is set to something irrelevant — absolute path must still resolve
    ds = DetDataset(str(manifest_path), norm=norm, train=False, data_root="/nonexistent/root")

    tensor, boxes, labels = ds[0]
    assert tensor.shape == (3, 128, 128)


def test_detdataset_drops_degenerate_box_after_clip(tmp_path):
    """A box that clips to zero width must be dropped; the valid box is kept."""
    from aslv2.detect.data import DetDataset

    # 128×128 image
    img = np.full((128, 128, 3), 128, dtype=np.uint8)
    img_path = tmp_path / "frame.png"
    cv2.imwrite(str(img_path), img)

    # Box 1: fully valid [10, 10, 60, 60] in 128² space (already matches 128² image)
    # Box 2: entirely past the right edge — after clipping x1=128, x2=128 → width=0
    manifest = [
        {
            "image": str(img_path),
            "boxes": [[10.0, 10.0, 60.0, 60.0], [200.0, 10.0, 250.0, 60.0]],
            "labels": [0, 1],
        }
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    norm = {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]}
    ds = DetDataset(str(manifest_path), norm=norm, train=False)

    tensor, boxes, labels = ds[0]

    # Only the valid box should survive — the degenerate one must be dropped
    assert boxes.shape[0] == 1, (
        f"Expected 1 box after dropping degenerate, got {boxes.shape[0]}"
    )
    assert int(labels[0]) == 0, "Surviving box should carry the first label (class 0)"
