import torch
from asl.model import FrameEncoder, build, TwoStreamClassifier


def test_encoder_accepts_2ch_input():
    enc = FrameEncoder(emb=384, width=48, in_ch=2)
    out = enc(torch.randn(4, 2, 112, 112))
    assert out.shape == (4, 384)


def test_two_stream_forward_shape():
    m = build(75, emb=384, width=48, two_stream=True)
    assert isinstance(m, TwoStreamClassifier)
    rgb = torch.randn(2, 16, 3, 112, 112)
    flow = torch.randn(2, 16, 2, 112, 112)
    assert m(rgb, flow).shape == (2, 75)


def test_two_stream_under_10mb_fp32():
    m = build(75, two_stream=True)
    n = sum(p.numel() for p in m.parameters())
    assert n * 4 / 1e6 < 10.0          # fp32 under the 10MB browser cap
