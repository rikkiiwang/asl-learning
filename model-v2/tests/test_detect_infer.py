"""Tests for detect_frame inference function."""
import numpy as np
import torch
from aslv2.detect.model import Detector
from aslv2.detect.infer import detect_frame

_NORM = {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]}


def test_detect_frame_img_size_runs_and_stays_in_frame():
    """At img_size=192 the model's 12×12×3=432 anchors must align with anchors_for(192);
    decode then runs end-to-end and boxes land in original-frame pixel space."""
    torch.manual_seed(0)
    model = Detector(n_classes=2, n_anchors=3).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.data *= 5.0   # push some detections over threshold

    frame = np.full((300, 240, 3), 128, dtype=np.uint8)  # non-square original
    result = detect_frame(model, frame, _NORM, img_size=192)

    assert set(result) == {"hands", "head"}
    arrs = [result["hands"]] + ([result["head"][None]] if result["head"] is not None else [])
    for arr in arrs:
        assert (arr >= 0).all()
        assert (arr[:, [0, 2]] <= 240).all()   # x within original width
        assert (arr[:, [1, 3]] <= 300).all()   # y within original height


def test_detect_frame_output_structure():
    """detect_frame returns the expected dict structure."""
    model = Detector(n_classes=2, n_anchors=3).eval()
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    result = detect_frame(model, frame, _NORM)

    assert isinstance(result, dict), "Result must be a dict"
    assert "hands" in result, "Result must have 'hands' key"
    assert "head" in result, "Result must have 'head' key"


def test_detect_frame_hands_shape():
    """Hands must be an array with shape (<=2, 4)."""
    model = Detector(n_classes=2, n_anchors=3).eval()
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    result = detect_frame(model, frame, _NORM)

    hands = result["hands"]
    assert isinstance(hands, np.ndarray), "Hands must be an ndarray"
    assert hands.ndim == 2 and hands.shape[1] == 4, \
        f"Hands shape must be (N,4), got {hands.shape}"
    assert hands.shape[0] <= 2, \
        f"At most 2 hands allowed, got {hands.shape[0]}"


def test_detect_frame_head_is_1d_or_none():
    """Head must be a 1-D array of 4 coords or None."""
    model = Detector(n_classes=2, n_anchors=3).eval()
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    result = detect_frame(model, frame, _NORM)

    head = result["head"]
    assert head is None or (isinstance(head, np.ndarray) and head.shape == (4,)), \
        f"Head must be (4,) or None, got {head}"


def test_detect_frame_coords_in_original_space():
    """Box coordinates must be in original-frame pixel space (256²), not 128²."""
    # Use a large-weight random model to produce non-trivial activations
    torch.manual_seed(0)
    model = Detector(n_classes=2, n_anchors=3).eval()

    # Amplify head weights so some detections fire above threshold
    with torch.no_grad():
        for p in model.parameters():
            p.data *= 5.0

    frame = np.full((256, 256, 3), 128, dtype=np.uint8)
    result = detect_frame(model, frame, _NORM)

    # If any boxes are detected they must be in 256² pixel space (0..256 range),
    # NOT 128² space — i.e. some coordinate should potentially exceed 128 if scaled.
    # At minimum, check boxes are non-negative and within 256×256.
    for arr in [result["hands"]] + ([result["head"][None]] if result["head"] is not None else []):
        assert (arr >= 0).all(), "Box coords must be >= 0"
        assert (arr <= 256).all(), f"Box coords must be <= 256 (original frame size), got {arr}"


def test_detect_frame_scaling():
    """Verify boxes are actually scaled by orig/128 (factor 2 for 256² frame).

    We inject artificial logits into the model so that anchor 95 fires with
    high confidence for class 0 (hand). Anchor 95 = cell(4,4), scale=32px =>
    center (72,72), half=16 => xyxy (56,56,88,88) in 128² — all within bounds.
    The returned box in 256² space should be scaled ×2 = (112,112,176,176).
    """
    from aslv2.detect.anchors import make_anchors

    # anchor 95: cell index 31 (row 3, col 7), scale index 2 (96px)
    # Actually let's pick an anchor that's guaranteed to be in-bounds after scaling.
    # Anchor at cell (4,4) [grid center], scales=(32,64,96):
    #   cell center = (4+0.5)*16=72, anchors at indices 4*8*3+4*3 = 108
    anchors = make_anchors(img=128, stride=16, scales=(32, 64, 96))
    # Pick anchor 108 (cell row=4, col=4, scale=32): center (72,72), half=16
    # xyxy = (56, 56, 88, 88) — comfortably in [0,128]
    anchor_idx = 4 * 8 * 3 + 4 * 3    # = 108
    expected_128 = anchors[anchor_idx]
    assert (expected_128 >= 0).all() and (expected_128 <= 128).all(), \
        f"Test setup: anchor {anchor_idx} out of [0,128]: {expected_128}"

    model = Detector(n_classes=2, n_anchors=3).eval()

    class FakeDetector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._inner = model
        def forward(self, x):
            cls, box = self._inner(x)
            # Force all cls logits to -100, then target anchor, class 0 = +100
            cls = torch.full_like(cls, -100.0)
            box = torch.zeros_like(box)   # zero deltas → anchor box returned as-is
            cls[:, anchor_idx, 0] = 100.0
            return cls, box

    fake = FakeDetector().eval()
    frame = np.full((256, 256, 3), 100, dtype=np.uint8)
    result = detect_frame(fake, frame, _NORM)

    hands = result["hands"]
    assert hands.shape[0] >= 1, "Expected at least one hand detection"
    scale = 256 / 128   # = 2.0
    expected_256 = expected_128 * scale
    np.testing.assert_allclose(hands[0], expected_256, atol=2.0,
                                err_msg="Hand box should be scaled from 128² to 256²")
