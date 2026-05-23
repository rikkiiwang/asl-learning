"""Focal classification + smooth-L1 box regression loss for SSD-style detector."""
import numpy as np
import torch
import torch.nn.functional as F


def det_loss(cls_logits, box_pred, cls_tgt, box_t, pos, valid, alpha=0.25, gamma=2.0):
    """Focal cls loss over valid anchors + smooth-L1 box loss over positives.

    Args:
        cls_logits : (n,2) float — raw class logits
        box_pred   : (n,4) float — predicted SSD deltas
        cls_tgt    : (n,2) float — one-hot for positives, all-zero for background
        box_t      : (n,4) float — target SSD deltas (only positives used)
        pos        : (n,)  bool  — positive anchor mask
        valid      : (n,)  bool  — anchors included in cls loss
        alpha      : focal loss alpha (class-balance weight)
        gamma      : focal loss focusing exponent

    Returns:
        Scalar torch.Tensor — combined loss.
    """
    # Accept numpy inputs
    def _t(x, dtype=torch.float32):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(dtype)
        return x.to(dtype) if x.dtype != dtype else x

    def _b(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x.astype(bool))
        return x.bool()

    cls_logits = _t(cls_logits)
    box_pred   = _t(box_pred)
    cls_tgt    = _t(cls_tgt)
    box_t      = _t(box_t)
    pos        = _b(pos)
    valid      = _b(valid)

    p  = torch.sigmoid(cls_logits)
    ce = F.binary_cross_entropy_with_logits(cls_logits, cls_tgt, reduction="none")
    pt = p * cls_tgt + (1 - p) * (1 - cls_tgt)
    a  = alpha * cls_tgt + (1 - alpha) * (1 - cls_tgt)
    focal = a * (1 - pt) ** gamma * ce          # (n, 2)

    n_pos    = pos.sum().clamp(min=1)
    cls_loss = focal[valid].sum() / n_pos

    if pos.any():
        box_loss = F.smooth_l1_loss(
            box_pred[pos], box_t[pos], reduction="sum"
        ) / n_pos
    else:
        box_loss = box_pred.sum() * 0.0

    return cls_loss + box_loss
