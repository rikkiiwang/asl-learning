"""Smoke test for the landmark training step.

Verifies that a single forward → wing_loss → backward → opt.step cycle:
  1. Produces a finite loss.
  2. Changes at least one parameter (gradients flowed, optimizer stepped).

No real data is loaded; everything is synthetic and runs on CPU.
"""
import torch
import pytest


def _make_synthetic_batch(n_images: int = 4, img_size: int = 64):
    """Return (imgs_tensor, kp_tensor) with random crops and random [0,1] keypoints."""
    imgs = torch.rand(n_images, 3, img_size, img_size)
    kps  = torch.rand(n_images, 21, 2)   # in [0, 1] — normalised crop space
    return imgs, kps


def test_single_train_step_finite_loss_and_param_update():
    """One forward → wing_loss → backward → step on CPU must be finite and update params."""
    from aslv2.landmark.model import Landmark
    from aslv2.landmark.loss import wing_loss

    torch.manual_seed(42)
    device = torch.device("cpu")

    model = Landmark(width=8).to(device)   # tiny width for speed
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3)

    imgs, kps = _make_synthetic_batch(n_images=4)
    imgs = imgs.to(device)
    kps  = kps.to(device)

    # Snapshot a parameter before the step
    param_before = next(model.parameters()).detach().clone()

    # Forward → loss → backward → step
    model.train()
    opt.zero_grad()
    pred = model(imgs)                       # (4, 21, 2)

    assert pred.shape == (4, 21, 2), f"Unexpected output shape: {pred.shape}"

    loss = wing_loss(pred, kps)

    assert torch.isfinite(loss), f"Loss is not finite: {loss}"

    loss.backward()
    opt.step()

    param_after = next(model.parameters()).detach().clone()
    assert not torch.allclose(param_before, param_after), \
        "Parameter did not change after optimizer step — gradients may not have flowed"
