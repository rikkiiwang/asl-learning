import torch
from aslv2.export_spike.dummy_models import DummyDetector, DummyLandmark, DummyRecognizer


def test_detector_emits_anchor_grid():
    det = DummyDetector()
    boxes, scores = det(torch.randn(16, 3, 128, 128))   # F frames
    assert boxes.shape[0] == 16 and boxes.shape[2] == 4  # (F, A, 4)
    assert scores.shape[:2] == boxes.shape[:2]
    assert scores.shape[2] == 2                          # hand, head classes


def test_landmark_regresses_21_keypoints():
    lm = DummyLandmark()
    out = lm(torch.randn(8, 3, 64, 64))                  # 8 crops
    assert out.shape == (8, 21, 2)


def test_recognizer_outputs_75_logits():
    rec = DummyRecognizer(geom_dim=93, n_classes=75)
    # per-frame fused vector = geom(93) + 2 hand appearance embeds(2*32)
    logits = rec(torch.randn(1, 16, 93 + 64))
    assert logits.shape == (1, 75)
