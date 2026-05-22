"""End-to-end inference graph used ONLY to test exportability (spec §6).
16 raw frames -> detect -> topk select (2 hands + 1 head) -> roi_align crops ->
landmark -> tensorized geometry -> recognizer -> 75 logits.
Geometry here is a differentiable tensor approximation of aslv2.geometry; the
offline numpy version remains the source of truth for training-time caching."""
import torch
import torch.nn as nn
from torchvision.ops import roi_align

from .dummy_models import DummyDetector, DummyLandmark, DummyRecognizer

F = 16


class CombinedConstellation(nn.Module):
    def __init__(self):
        super().__init__()
        self.det = DummyDetector()
        self.lm = DummyLandmark()
        self.rec = DummyRecognizer()
        self.app = nn.Sequential(nn.Conv2d(3, 8, 3, 2, 1), nn.ReLU(),
                                 nn.AdaptiveAvgPool2d(1))   # tiny appearance -> 8
        self.app_proj = nn.Linear(8, 32)

    def _topk_box(self, boxes, scores, k):
        # boxes (F,A,4), scores (F,A) -> (F,k,4)
        idx = scores.topk(k, dim=1).indices                # (F,k)
        return torch.gather(boxes, 1, idx.unsqueeze(-1).expand(-1, -1, 4))

    def forward(self, frames):                              # (F,3,128,128)
        boxes, scores = self.det(frames)
        hands = self._topk_box(boxes, scores[..., 0], 2)    # (F,2,4)
        head = self._topk_box(boxes, scores[..., 1], 1)     # (F,1,4)

        # roi_align expects List[Tensor] of boxes per image, or (K,5) with batch idx
        batch_idx = torch.arange(F).repeat_interleave(2).float().unsqueeze(1)
        rois = torch.cat([batch_idx, hands.reshape(F * 2, 4)], dim=1)   # (F*2,5)
        crops = roi_align(frames, rois, output_size=(64, 64))          # (F*2,3,64,64)

        kps = self.lm(crops)                                # (F*2,21,2)
        app = self.app_proj(self.app(crops).flatten(1))     # (F*2,32)

        # tensor geometry: normalize keypoints by head size/center per frame
        hc = head.reshape(F, 4)
        hcx = (hc[:, 0] + hc[:, 2]) / 2
        hcy = (hc[:, 1] + hc[:, 3]) / 2
        hs = torch.clamp(torch.maximum(hc[:, 2] - hc[:, 0], hc[:, 3] - hc[:, 1]), min=1.0)
        center = torch.stack([hcx, hcy], dim=1).unsqueeze(1)            # (F,1,2)
        kps_f = kps.reshape(F, 2, 21, 2)
        norm_kps = (kps_f - center.unsqueeze(1)) / hs.reshape(F, 1, 1, 1)
        geom = norm_kps.reshape(F, 2 * 21 * 2)                          # (F,84)
        # pad geom to GEOM_DIM=93 (offsets/flags are zeros in the shape-only spike)
        geom = torch.cat([geom, torch.zeros(F, 9)], dim=1)             # (F,93)

        app_f = app.reshape(F, 2, 32).reshape(F, 64)
        fused = torch.cat([geom, app_f], dim=1).unsqueeze(0)            # (1,F,157)
        return self.rec(fused)
