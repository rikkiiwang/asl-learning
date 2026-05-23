"""Smoke test for the detector training step.

Verifies that a single encode→forward→det_loss→backward→opt.step cycle:
  1. Produces a finite loss.
  2. Changes at least one parameter (i.e. gradients flowed and the optimizer stepped).

No real data is loaded; everything is synthetic and runs on CPU.
"""
import numpy as np
import torch
import pytest


def _make_synthetic_batch(n_images: int = 2, img_size: int = 128):
    """Return (imgs_tensor, raw_targets) where raw_targets is a list of (boxes, labels)."""
    imgs = torch.rand(n_images, 3, img_size, img_size)
    # Each image gets 2 hand boxes + 1 head box (xyxy in [0, img_size])
    raw_targets = []
    for _ in range(n_images):
        boxes = np.array([
            [10., 10., 60., 60.],   # hand
            [70., 20., 110., 80.],  # hand
            [30., 5.,  90., 50.],   # head
        ], dtype=np.float32)
        labels = np.array([0, 0, 1], dtype=np.int64)
        raw_targets.append((boxes, labels))
    return imgs, raw_targets


def test_single_train_step_finite_loss_and_param_update():
    """One encode→forward→loss→backward→step on CPU must be finite and update params."""
    from aslv2.detect.model import Detector
    from aslv2.detect.anchors import make_anchors
    from aslv2.detect.encode import encode_targets
    from aslv2.detect.loss import det_loss

    torch.manual_seed(0)
    device = torch.device("cpu")

    model = Detector(n_classes=2, n_anchors=3, width=32).to(device)  # tiny for speed
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    imgs, raw_targets = _make_synthetic_batch(n_images=2)
    imgs = imgs.to(device)

    anchors = make_anchors()  # (192, 4)

    # Encode targets per image
    cls_tgts, box_ts, pos_masks, valid_masks = [], [], [], []
    for boxes, labels in raw_targets:
        ct, bt, pm, vm = encode_targets(anchors, boxes, labels)
        cls_tgts.append(ct)
        box_ts.append(bt)
        pos_masks.append(pm)
        valid_masks.append(vm)

    # Stack into batch tensors
    cls_tgt_batch  = torch.from_numpy(np.stack(cls_tgts)).to(device)   # (B, 192, 2)
    box_t_batch    = torch.from_numpy(np.stack(box_ts)).to(device)     # (B, 192, 4)
    pos_batch      = torch.from_numpy(np.stack(pos_masks)).to(device)  # (B, 192)
    valid_batch    = torch.from_numpy(np.stack(valid_masks)).to(device) # (B, 192)

    # Snapshot a parameter before the step
    param_before = next(model.parameters()).detach().clone()

    # Forward + loss
    model.train()
    opt.zero_grad()
    cls_logits, box_pred = model(imgs)  # (B, 192, 2), (B, 192, 4)

    # Aggregate loss over batch
    B = imgs.shape[0]
    total_loss = sum(
        det_loss(
            cls_logits[i], box_pred[i],
            cls_tgt_batch[i], box_t_batch[i],
            pos_batch[i], valid_batch[i],
        )
        for i in range(B)
    ) / B

    assert torch.isfinite(total_loss), f"Loss is not finite: {total_loss}"

    total_loss.backward()
    opt.step()

    param_after = next(model.parameters()).detach().clone()
    assert not torch.allclose(param_before, param_after), \
        "Parameter did not change after optimizer step — gradients may not have flowed"
