"""Tests for Detector model — shapes and size budget."""
import torch
from aslv2.detect.model import Detector
from aslv2.size_budget import assert_within_budget


def test_detector_shapes_and_budget():
    m = Detector(n_classes=2, n_anchors=3).eval()
    cls, box = m(torch.randn(2, 3, 128, 128))
    assert cls.shape == (2, 8 * 8 * 3, 2)      # per-anchor 2-class logits
    assert box.shape == (2, 8 * 8 * 3, 4)
    assert_within_budget("detector", m)        # <= 2.0M params (spec §3.7)
