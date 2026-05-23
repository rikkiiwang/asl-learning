"""Wing loss for robust 2D keypoint regression.

Reference: Feng et al., "Wing Loss for Robust Facial Landmark Localisation
with Convolutional Neural Networks", CVPR 2018.

wing_loss(pred, gt, w=0.1, eps=0.02)
  - pred, gt : (..., 21, 2)  in [0,1]
  - w        : half-width of the non-linear region  (default 0.1)
  - eps      : smooth region width                  (default 0.02)

Returns a scalar mean loss over all coordinates.

Definition (per coordinate):
  |x| < w  →  w * ln(1 + |x|/eps)
  |x| >= w →  |x| - C           where C = w - w*ln(1 + w/eps)
"""
import math
import torch
import torch.nn as nn


def wing_loss(
    pred: torch.Tensor,
    gt:   torch.Tensor,
    w:    float = 0.1,
    eps:  float = 0.02,
) -> torch.Tensor:
    """Compute the mean wing loss between pred and gt keypoints.

    Args:
        pred: predicted keypoints  (..., 21, 2).
        gt:   ground-truth         (..., 21, 2).
        w:    wing half-width.
        eps:  smooth region width.

    Returns:
        Scalar tensor — mean loss over all (batch, keypoint, coord) elements.
    """
    C = w - w * math.log(1.0 + w / eps)
    diff = (pred - gt).abs()
    loss = torch.where(
        diff < w,
        w * torch.log(1.0 + diff / eps),
        diff - C,
    )
    return loss.mean()
