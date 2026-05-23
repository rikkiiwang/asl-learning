"""Tests for det_loss (focal cls + smooth-L1 box)."""
import numpy as np
import torch
import pytest
from aslv2.detect.loss import det_loss


def _perfect_inputs(n_anchors=100, n_pos=5, gt_class=0):
    """Build tensors where predictions are near-perfect.

    Returns cls_logits, box_pred, cls_tgt, box_t, pos, valid such that
    the correct class has logit +8 and the wrong class has -8, and
    box_pred == box_t on positives.
    """
    cls_logits = torch.full((n_anchors, 2), -8.0)
    box_pred = torch.zeros(n_anchors, 4)
    cls_tgt = torch.zeros(n_anchors, 2)
    box_t = torch.zeros(n_anchors, 4)
    pos = torch.zeros(n_anchors, dtype=torch.bool)
    valid = torch.ones(n_anchors, dtype=torch.bool)

    pos[:n_pos] = True
    cls_logits[:n_pos, gt_class] = 8.0   # strongly predict correct class
    cls_tgt[:n_pos, gt_class] = 1.0      # one-hot ground truth
    box_t[:n_pos] = torch.randn(n_pos, 4) * 0.5
    box_pred[:n_pos] = box_t[:n_pos]     # perfect box predictions

    return cls_logits, box_pred, cls_tgt, box_t, pos, valid


def test_perfect_predictions_give_near_zero_loss():
    """Near-perfect logits and matching boxes should yield loss close to 0."""
    cls_logits, box_pred, cls_tgt, box_t, pos, valid = _perfect_inputs()
    loss = det_loss(cls_logits, box_pred, cls_tgt, box_t, pos, valid)
    assert loss.item() < 0.1, f"Expected near-zero loss for perfect preds, got {loss.item():.4f}"


def test_wrong_class_gives_larger_loss():
    """Predicting the wrong class should give a clearly larger loss than correct."""
    cls_logits_good, box_pred, cls_tgt, box_t, pos, valid = _perfect_inputs(gt_class=0)
    loss_good = det_loss(cls_logits_good, box_pred, cls_tgt, box_t, pos, valid)

    # Flip: strongly predict class 1 instead of class 0
    cls_logits_bad = cls_logits_good.clone()
    cls_logits_bad[:5, 0] = -8.0
    cls_logits_bad[:5, 1] = 8.0   # wrong class
    loss_bad = det_loss(cls_logits_bad, box_pred, cls_tgt, box_t, pos, valid)

    assert loss_bad.item() > loss_good.item() + 0.5, (
        f"Wrong-class loss ({loss_bad.item():.4f}) should be clearly larger "
        f"than correct-class loss ({loss_good.item():.4f})"
    )


def test_accepts_numpy_inputs():
    """det_loss should accept numpy arrays and return a scalar tensor."""
    cls_logits, box_pred, cls_tgt, box_t, pos, valid = _perfect_inputs()
    # Convert to numpy
    loss = det_loss(
        cls_logits.numpy(), box_pred.numpy(),
        cls_tgt.numpy(), box_t.numpy(),
        pos.numpy(), valid.numpy(),
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0   # scalar


def test_no_positives_returns_finite_loss():
    """When there are no positives, loss should be finite (box term is zero)."""
    n = 50
    cls_logits = torch.zeros(n, 2)
    box_pred = torch.zeros(n, 4)
    cls_tgt = torch.zeros(n, 2)
    box_t = torch.zeros(n, 4)
    pos = torch.zeros(n, dtype=torch.bool)
    valid = torch.ones(n, dtype=torch.bool)

    loss = det_loss(cls_logits, box_pred, cls_tgt, box_t, pos, valid)
    assert torch.isfinite(loss), f"Loss should be finite with no positives, got {loss.item()}"
