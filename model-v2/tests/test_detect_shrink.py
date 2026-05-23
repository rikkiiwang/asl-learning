"""Tests for the image-shrinking helper (TDD: write RED first, then make it GREEN)."""
import numpy as np
import pytest

from aslv2.detect.shrink import resize_keep_aspect


def test_downscale_400x200():
    """400×200 image (W×H), max_side=200 -> scale 0.5, out 200×100, boxes halved."""
    img = np.zeros((200, 400, 3), dtype=np.uint8)  # H=200, W=400
    boxes = np.array([[100.0, 50.0, 300.0, 150.0]])
    img2, boxes2, scale = resize_keep_aspect(img, boxes, max_side=200)

    assert scale == pytest.approx(0.5)
    assert img2.shape == (100, 200, 3)  # H=100, W=200
    np.testing.assert_allclose(boxes2, [[50.0, 25.0, 150.0, 75.0]])


def test_no_upscale_small_image():
    """100×80 image, max_side=200 -> already smaller, scale=1.0, unchanged."""
    img = np.zeros((80, 100, 3), dtype=np.uint8)  # H=80, W=100
    boxes = np.array([[10.0, 5.0, 90.0, 70.0]])
    img2, boxes2, scale = resize_keep_aspect(img, boxes, max_side=200)

    assert scale == pytest.approx(1.0)
    assert img2.shape == (80, 100, 3)
    np.testing.assert_allclose(boxes2, [[10.0, 5.0, 90.0, 70.0]])


def test_square_image():
    """200×200 image, max_side=100 -> scale 0.5, out 100×100."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    boxes = np.array([[0.0, 0.0, 200.0, 200.0]])
    img2, boxes2, scale = resize_keep_aspect(img, boxes, max_side=100)

    assert scale == pytest.approx(0.5)
    assert img2.shape == (100, 100, 3)
    np.testing.assert_allclose(boxes2, [[0.0, 0.0, 100.0, 100.0]])


def test_empty_boxes():
    """Works correctly with an empty boxes array."""
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    boxes = np.zeros((0, 4), dtype=np.float32)
    img2, boxes2, scale = resize_keep_aspect(img, boxes, max_side=300)

    assert scale == pytest.approx(0.5)
    assert img2.shape == (200, 300, 3)
    assert boxes2.shape == (0, 4)
