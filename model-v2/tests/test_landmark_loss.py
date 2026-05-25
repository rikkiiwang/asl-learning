"""Tests for wing_loss."""
import torch
import pytest
from aslv2.landmark.loss import wing_loss


def test_wing_loss_perfect():
    """Perfect prediction → loss ≈ 0."""
    pred = torch.rand(4, 21, 2)
    gt   = pred.clone()
    loss = wing_loss(pred, gt)
    assert float(loss) < 1e-6, f"Expected ~0, got {loss}"


def test_wing_loss_positive():
    """A fixed non-zero error gives a known positive value."""
    pred = torch.zeros(1, 21, 2)
    gt   = torch.ones(1, 21, 2) * 0.5
    loss = wing_loss(pred, gt)
    assert float(loss) > 0.0, "Loss should be positive for non-zero error"


def test_wing_loss_ordering():
    """Near-correct prediction gives lower loss than far-off prediction."""
    gt   = torch.zeros(1, 21, 2)
    near = torch.full((1, 21, 2), 0.05)
    far  = torch.full((1, 21, 2), 0.5)
    loss_near = wing_loss(near, gt)
    loss_far  = wing_loss(far,  gt)
    assert float(loss_near) < float(loss_far), (
        f"Near loss ({loss_near:.4f}) should be < far loss ({loss_far:.4f})"
    )
