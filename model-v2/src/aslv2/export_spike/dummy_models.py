"""Shape-only stand-ins so the export spike exercises the real op chain.
No learned weights matter here. Real models arrive in Plans 2-4."""
import torch
import torch.nn as nn

A = 256  # anchors per frame (fixed grid -> static shapes for export)


class DummyDetector(nn.Module):
    """(F,3,128,128) -> boxes (F,A,4) xyxy, scores (F,A,2) [hand, head]."""
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(3, 8, 3, 2, 1), nn.ReLU(),
                                  nn.AdaptiveAvgPool2d(1))
        self.box = nn.Linear(8, A * 4)
        self.cls = nn.Linear(8, A * 2)

    def forward(self, x):
        f = self.body(x).flatten(1)             # (F,8)
        boxes = self.box(f).reshape(-1, A, 4).sigmoid() * 128.0
        scores = self.cls(f).reshape(-1, A, 2)
        return boxes, scores


class DummyLandmark(nn.Module):
    """(N,3,64,64) -> (N,21,2) keypoints in crop space."""
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(3, 8, 3, 2, 1), nn.ReLU(),
                                  nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(8, 21 * 2)

    def forward(self, x):
        return self.fc(self.body(x).flatten(1)).reshape(-1, 21, 2)


class DummyRecognizer(nn.Module):
    """(B,F,fused) -> (B,n_classes). Mean-pool temporal head (shape-only)."""
    def __init__(self, geom_dim=93, n_classes=75):
        super().__init__()
        self.fc = nn.Linear(geom_dim + 64, n_classes)

    def forward(self, x):
        return self.fc(x.mean(dim=1))
