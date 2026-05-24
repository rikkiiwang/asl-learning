"""From-scratch tiny video classifier: depthwise-separable 2D CNN frame encoder
shared across frames -> temporal head (attention pooling baseline; optional
1-2 layer Transformer) -> 75-way logits.

No pretrained weights, no external backbones — everything initialized from scratch.
Designed to quantize to 2-6 MB for ONNX Runtime Web.
"""
import torch
import torch.nn as nn


class SepConv(nn.Module):
    """Depthwise-separable conv + BN + ReLU, optional stride."""
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class FrameEncoder(nn.Module):
    """112x112xin_ch -> embedding of dim `emb`. `width` scales channel capacity."""
    def __init__(self, emb=384, width=48, in_ch=3):
        super().__init__()
        w = width
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, w, 3, 2, 1, bias=False),    # 56
            nn.BatchNorm2d(w), nn.ReLU(inplace=True))
        self.body = nn.Sequential(
            SepConv(w, 2 * w, stride=2),             # 28
            SepConv(2 * w, 2 * w),
            SepConv(2 * w, 4 * w, stride=2),         # 14
            SepConv(4 * w, 4 * w),
            SepConv(4 * w, 8 * w, stride=2),         # 7
            SepConv(8 * w, 8 * w),
            SepConv(8 * w, emb),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):                            # x: (B*F, 3, 112, 112)
        x = self.body(self.stem(x))
        return self.pool(x).flatten(1)               # (B*F, emb)


class AttnPool(nn.Module):
    """Learned attention pooling over the frame dimension."""
    def __init__(self, emb):
        super().__init__()
        self.score = nn.Linear(emb, 1)

    def forward(self, x):                            # x: (B, F, emb)
        w = torch.softmax(self.score(x), dim=1)      # (B, F, 1)
        return (w * x).sum(1)                         # (B, emb)


class SignClassifier(nn.Module):
    def __init__(self, num_classes, emb=384, head="attn",
                 tf_layers=1, tf_heads=4, dropout=0.3, width=48):
        super().__init__()
        self.encoder = FrameEncoder(emb, width)
        self.head_type = head
        if head == "transformer":
            self.pos = nn.Parameter(torch.zeros(1, 64, emb))
            layer = nn.TransformerEncoderLayer(emb, tf_heads, emb * 2,
                                               dropout, batch_first=True)
            self.tf = nn.TransformerEncoder(layer, tf_layers)
        self.pool = AttnPool(emb)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(emb, num_classes)

    def forward(self, x):                            # x: (B, F, C, H, W)
        x = x.contiguous()
        b, f = x.shape[:2]
        e = self.encoder(x.reshape(b * f, *x.shape[2:])).reshape(b, f, -1)
        if self.head_type == "transformer":
            e = self.tf(e + self.pos[:, :f])
        return self.fc(self.drop(self.pool(e)))


class TwoStreamClassifier(nn.Module):
    """RGB appearance stream + optical-flow motion stream, late-fused."""
    def __init__(self, num_classes, emb=384, width=48, dropout=0.3, flow_ch=2):
        super().__init__()
        self.encoder = FrameEncoder(emb, width, in_ch=3)        # name matches ckpt
        self.encoder_flow = FrameEncoder(emb, width, in_ch=flow_ch)
        self.pool = AttnPool(emb)
        self.pool_flow = AttnPool(emb)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * emb, num_classes)

    def forward(self, rgb, flow):                    # (B,F,3,H,W), (B,F,2,H,W)
        rgb, flow = rgb.contiguous(), flow.contiguous()
        b, f = rgb.shape[:2]
        er = self.encoder(rgb.reshape(b * f, *rgb.shape[2:])).reshape(b, f, -1)
        ef = self.encoder_flow(flow.reshape(b * f, *flow.shape[2:])).reshape(b, f, -1)
        fused = torch.cat([self.pool(er), self.pool_flow(ef)], dim=1)
        return self.fc(self.drop(fused))


def build(num_classes, two_stream=False, **kw):
    if two_stream:
        return TwoStreamClassifier(
            num_classes, emb=kw.get("emb", 384), width=kw.get("width", 48),
            dropout=kw.get("dropout", 0.3))
    return SignClassifier(num_classes, **kw)


if __name__ == "__main__":
    m = build(75)
    n = sum(p.numel() for p in m.parameters())
    x = torch.randn(2, 16, 3, 112, 112)
    print("params:", n, f"(~{n*4/1e6:.1f} MB fp32)")
    print("out:", m(x).shape)
