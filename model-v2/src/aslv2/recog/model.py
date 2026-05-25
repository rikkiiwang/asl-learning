"""Stage-2 recognizers (Plan 4).

RecognizerA — geometry-only (the thesis test): per-frame pose-geometry MLP →
temporal head (attention pool, optional Transformer) → 75-class logits. No pixels
enter the model, so there is little appearance to overfit to.

RecognizerB (Task 6) adds a small shared appearance CNN over the hand crops; it is
only kept if it beats A on the signer-held-out val split (spec §4 discipline).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from aslv2.geometry import GEOM_DIM


class AttnPool(nn.Module):
    """Learned attention pooling over the frame dimension (same as v1)."""
    def __init__(self, emb: int):
        super().__init__()
        self.score = nn.Linear(emb, 1)

    def forward(self, x):                       # x: (B, F, emb)
        w = torch.softmax(self.score(x), dim=1)  # (B, F, 1)
        return (w * x).sum(1)                     # (B, emb)


class _TemporalHead(nn.Module):
    """Shared temporal aggregation: optional Transformer over frames + attn pool."""
    def __init__(self, emb: int, head: str, dropout: float, tf_layers: int, tf_heads: int):
        super().__init__()
        self.head_type = head
        if head == "transformer":
            self.pos = nn.Parameter(torch.zeros(1, 64, emb))
            layer = nn.TransformerEncoderLayer(emb, tf_heads, emb * 2, dropout,
                                               batch_first=True)
            self.tf = nn.TransformerEncoder(layer, tf_layers)
        self.pool = AttnPool(emb)

    def forward(self, e):                        # e: (B, F, emb) -> (B, emb)
        if self.head_type == "transformer":
            e = self.tf(e + self.pos[:, :e.shape[1]])
        return self.pool(e)


class RecognizerA(nn.Module):
    """Geometry-only recognizer.

    `use_velocity`: also feed the per-frame temporal delta (Δgeometry) — signs are
    defined by motion, so velocity is a strong, free signal. Doubles the per-frame
    input width (position ‖ velocity).
    """
    def __init__(self, n_classes: int = 75, emb: int = 192, head: str = "attn",
                 dropout: float = 0.3, tf_layers: int = 1, tf_heads: int = 4,
                 geom_dim: int = GEOM_DIM, use_velocity: bool = False):
        super().__init__()
        self.use_velocity = use_velocity
        in_dim = geom_dim * (2 if use_velocity else 1)
        self.frame = nn.Sequential(
            nn.Linear(in_dim, emb), nn.LayerNorm(emb), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(emb, emb), nn.ReLU(inplace=True),
        )
        self.temporal = _TemporalHead(emb, head, dropout, tf_layers, tf_heads)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(emb, n_classes)

    def forward(self, geom: torch.Tensor) -> torch.Tensor:
        """geom: (B, F, geom_dim) → logits (B, n_classes)."""
        b, f = geom.shape[:2]
        if self.use_velocity:
            vel = torch.zeros_like(geom)
            vel[:, 1:] = geom[:, 1:] - geom[:, :-1]
            x = torch.cat([geom, vel], dim=-1)
        else:
            x = geom
        e = self.frame(x.reshape(b * f, -1)).reshape(b, f, -1)
        return self.fc(self.drop(self.temporal(e)))
