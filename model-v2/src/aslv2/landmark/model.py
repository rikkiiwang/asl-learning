"""Tiny depthwise-separable CNN for 21-keypoint hand landmark regression.

Input:  (N, 3, 64, 64) RGB crop, normalised.
Output: (N, 21, 2) keypoints in [0, 1] crop-space (sigmoid bounded).

Architecture: stem → 4 depthwise-separable blocks (stride 2 on first 3) →
GAP → Linear(·, 42) → sigmoid → reshape(-1, 21, 2).

Width = 32 keeps param count well under the 3.0 M cap (spec §3.7):
  stem:        3→32    (3×3 conv)
  block 1:     32→32   dw + 32→64  pw  stride=2   → 16×16
  block 2:     64→64   dw + 64→128 pw  stride=2   → 8×8
  block 3:     128→128 dw + 128→128 pw stride=2   → 4×4
  block 4:     128→128 dw + 128→128 pw stride=1   → 4×4
  GAP → 128 → Linear(128, 42) → sigmoid → (N, 21, 2)
Total params ≈ 95 k — tiny and well within budget.
"""
import torch
import torch.nn as nn


def _dw_sep_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    """Depthwise-separable conv block: DW → BN → ReLU → PW → BN → ReLU."""
    return nn.Sequential(
        # depthwise
        nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                  groups=in_ch, bias=False),
        nn.BatchNorm2d(in_ch),
        nn.ReLU(inplace=True),
        # pointwise
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class Landmark(nn.Module):
    """Tiny hand-landmark model: 64×64 crop → (21, 2) keypoints in [0, 1]."""

    def __init__(self, width: int = 32):
        super().__init__()
        w = width
        self.stem = nn.Sequential(
            nn.Conv2d(3, w, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            _dw_sep_block(w,      w * 2,  stride=2),   # 64→32
            _dw_sep_block(w * 2,  w * 4,  stride=2),   # 32→16
            _dw_sep_block(w * 4,  w * 4,  stride=2),   # 16→8
            _dw_sep_block(w * 4,  w * 4,  stride=1),   # 8→8
        )
        feat = w * 4
        self.head = nn.Linear(feat, 42)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N,3,64,64)  →  out: (N,21,2) in [0,1]."""
        x = self.stem(x)
        x = self.blocks(x)
        x = x.mean(dim=[2, 3])      # global average pool → (N, feat)
        x = self.head(x)            # (N, 42)
        x = torch.sigmoid(x)        # bound to [0,1]
        return x.view(-1, 21, 2)    # (N, 21, 2)
