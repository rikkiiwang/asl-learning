import os
import torch
from asl.pretrain import save_resume, load_resume
from asl.model import build


def test_resume_roundtrip(tmp_path):
    model = build(10, emb=384, head="attn", dropout=0.2, width=48)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = os.path.join(tmp_path, "resume.pt")
    save_resume(path, model, opt, epoch=5, best=0.42)
    model2 = build(10, emb=384, head="attn", dropout=0.2, width=48)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    ep, best = load_resume(path, model2, opt2, map_location="cpu")
    assert ep == 5 and abs(best - 0.42) < 1e-9
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)
