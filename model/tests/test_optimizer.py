import torch
from asl.model import build
from asl.train import make_optimizer


def test_default_scale_is_single_group_all_params():
    m = build(10, emb=384, width=48, head="attn", tf_layers=1, tf_heads=4)
    opt = make_optimizer(m, 1e-3, 0.05)            # default 1.0
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["base_lr"] == 1e-3
    assert all(p.requires_grad for p in m.parameters())


def test_frozen_encoder_trains_head_only():
    m = build(10, emb=384, width=48, head="attn", tf_layers=1, tf_heads=4)
    opt = make_optimizer(m, 1e-3, 0.05, 0.0)
    enc = [p for n, p in m.named_parameters() if n.startswith("encoder")]
    head = [p for n, p in m.named_parameters() if not n.startswith("encoder")]
    assert all(not p.requires_grad for p in enc)   # encoder frozen
    assert all(p.requires_grad for p in head)      # head trainable
    n_opt = sum(len(g["params"]) for g in opt.param_groups)
    assert n_opt == len(head)                       # optimizer holds head only
    assert all("base_lr" in g for g in opt.param_groups)


def test_discriminative_lr_two_groups():
    m = build(10, emb=384, width=48, head="attn", tf_layers=1, tf_heads=4)
    opt = make_optimizer(m, 1e-3, 0.05, 0.1)
    assert len(opt.param_groups) == 2
    assert abs(opt.param_groups[0]["base_lr"] - 1e-4) < 1e-12   # encoder 0.1x
    assert abs(opt.param_groups[1]["base_lr"] - 1e-3) < 1e-12   # head 1x
    assert all(p.requires_grad for p in m.parameters())
