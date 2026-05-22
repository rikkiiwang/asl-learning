import torch
from aslv2.export_spike.combined import CombinedConstellation


def test_combined_forward_outputs_logits():
    model = CombinedConstellation().eval()
    frames = torch.randn(16, 3, 128, 128)
    with torch.no_grad():
        logits = model(frames)
    assert logits.shape == (1, 75)
