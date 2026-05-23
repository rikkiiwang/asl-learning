"""Tests for Landmark model — shape and size budget."""
import torch
from aslv2.landmark.model import Landmark
from aslv2.size_budget import assert_within_budget


def test_landmark_shape_and_budget():
    m = Landmark().eval()
    out = m(torch.randn(4, 3, 64, 64))
    assert out.shape == (4, 21, 2)
    assert_within_budget("landmark", m)        # <= 3.0M params (spec §3.7)
