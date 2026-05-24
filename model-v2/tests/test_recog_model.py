"""Tests for the recognizer models (Plan 4 Tasks 4 & 6)."""
import torch

from aslv2.geometry import GEOM_DIM
from aslv2.recog.model import RecognizerA
from aslv2.size_budget import assert_within_budget


def test_recognizer_a_shape_and_budget():
    m = RecognizerA(n_classes=75, head="attn").eval()
    logits = m(torch.randn(2, 16, GEOM_DIM))      # geometry only
    assert logits.shape == (2, 75)
    assert_within_budget("recognizer", m)         # ≤ 1.5M params (spec §3.7)


def test_recognizer_a_transformer_head():
    m = RecognizerA(n_classes=75, head="transformer").eval()
    logits = m(torch.randn(3, 16, GEOM_DIM))
    assert logits.shape == (3, 75)
    assert_within_budget("recognizer", m)


def test_recognizer_a_variable_clip_length():
    m = RecognizerA(n_classes=75).eval()
    assert m(torch.randn(2, 8, GEOM_DIM)).shape == (2, 75)   # robust to F != 16


def test_recognizer_a_velocity_option():
    m = RecognizerA(n_classes=75, use_velocity=True).eval()
    logits = m(torch.randn(2, 16, GEOM_DIM))                 # deltas computed in-model
    assert logits.shape == (2, 75)
    assert_within_budget("recognizer", m)
