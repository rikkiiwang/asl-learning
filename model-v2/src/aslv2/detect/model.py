"""Tiny depthwise-separable SSD-style detector.

Architecture (128×128 input → 8×8 feature map → 192 anchors):
  stem:  3→w, stride 2        → 64×64
  block1: w→w, stride 2       → 32×32
  block2: w→2w                → 32×32
  block3: 2w→2w, stride 2     → 16×16
  block4: 2w→2w, stride 2     →  8×8   (stride-16 from input)
  cls head: 1×1 conv → (A*n_classes) → reshape (B, 8*8*A, n_classes)
  box head: 1×1 conv → (A*4)         → reshape (B, 8*8*A, 4)

With default width=32 this yields ~16k params, well under the 2M cap (spec §3.7).
"""
import torch
import torch.nn as nn


class SepConv(nn.Module):
    """Depthwise-separable conv + BN + ReLU, optional stride."""
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.dw  = nn.Conv2d(cin, cin,  3, stride, 1, groups=cin, bias=False)
        self.pw  = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn  = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class Detector(nn.Module):
    """Tiny from-scratch single-shot detector for hand (0) and head (1) boxes.

    Args:
        n_classes: number of foreground classes (default 2: hand + head).
        n_anchors: anchors per cell (default 3: scales 32/64/96 px).
        width:     base channel width (default 32).
    """
    def __init__(self, n_classes: int = 2, n_anchors: int = 3, width: int = 32):
        super().__init__()
        w  = width
        self.n_anchors  = n_anchors
        self.n_classes  = n_classes

        # Backbone: 128×128 → 8×8  (stride 16 = 2^4)
        self.stem = nn.Sequential(
            nn.Conv2d(3, w, 3, stride=2, padding=1, bias=False),  # → 64×64
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            SepConv(w,    w,    stride=2),   # → 32×32
            SepConv(w,    2*w),              # → 32×32  (width expansion)
            SepConv(2*w,  2*w,  stride=2),   # → 16×16
            SepConv(2*w,  2*w,  stride=2),   # →  8×8
        )

        # Detection heads (1×1 convolutions)
        self.cls_head = nn.Conv2d(2*w, n_anchors * n_classes, 1)
        self.box_head = nn.Conv2d(2*w, n_anchors * 4,         1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, 128, 128) float tensor.
        Returns:
            cls_logits: (B, H*W*A, n_classes)
            box_pred:   (B, H*W*A, 4)
        """
        feat = self.body(self.stem(x))            # (B, 2w, 8, 8)
        B, _, H, W = feat.shape

        cls = self.cls_head(feat)                 # (B, A*C, H, W)
        box = self.box_head(feat)                 # (B, A*4, H, W)

        A, C = self.n_anchors, self.n_classes

        # Reshape to (B, H*W*A, ·)
        cls = cls.permute(0, 2, 3, 1).reshape(B, H * W * A, C)
        box = box.permute(0, 2, 3, 1).reshape(B, H * W * A, 4)

        return cls, box
