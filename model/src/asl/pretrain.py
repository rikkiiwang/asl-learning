"""Pretrain the CNN encoder from scratch on the large (~500-class) ASL Citizen
slice, then save the encoder (+ temporal pool) weights for transfer to the
75-class fine-tune. No pretrained weights are used — this is our own encoder
learning general hand/motion features from more data.

Usage:
    python -m asl.pretrain --cache artifacts/cache/pretrain \
        --epochs 40 --batch-size 64 --out artifacts/checkpoints/pretrain
"""
import argparse
import json
import math
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from asl.dataset import MemmapClipDataset
from asl.model import build


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    c = t = 0
    for x, y in loader:
        c += (model(x.to(dev)).argmax(1).cpu() == y).sum().item()
        t += len(y)
    return c / max(t, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="artifacts/cache/pretrain")
    ap.add_argument("--norm", default="artifacts/manifest/norm.json")
    ap.add_argument("--emb", type=int, default=384)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.004)
    ap.add_argument("--weight-decay", type=float, default=0.02)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="artifacts/checkpoints/pretrain")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = device()
    meta = json.load(open(os.path.join(args.cache, "meta.json")))
    n, ncls = meta["n"], meta["num_classes"]
    print(f"device={dev}  pretrain clips={n}  classes={ncls}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = max(int(n * 0.04), ncls)          # small monitoring val
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    tr = MemmapClipDataset(args.cache, tr_idx, args.norm, train=True)
    va = MemmapClipDataset(args.cache, val_idx, args.norm, train=False)
    dl_tr = DataLoader(tr, args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    dl_va = DataLoader(va, args.batch_size, shuffle=False, num_workers=2)

    model = build(ncls, emb=args.emb, head="attn", dropout=0.2, width=args.width).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    crit = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    def lr_at(ep):
        if ep < args.warmup:
            return (ep + 1) / args.warmup
        p = (ep - args.warmup) / max(args.epochs - args.warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * p))

    os.makedirs(args.out, exist_ok=True)
    best = 0.0
    for ep in range(args.epochs):
        for g in opt.param_groups:
            g["lr"] = args.lr * lr_at(ep)
        model.train()
        tl = 0.0
        for x, y in dl_tr:
            opt.zero_grad()
            loss = crit(model(x.to(dev)), y.to(dev))
            loss.backward()
            opt.step()
            tl += loss.item() * len(y)
        tl /= len(tr)
        vacc = evaluate(model, dl_va, dev)
        print(f"ep {ep:02d}  loss {tl:.3f}  val_top1 {vacc:.3f}", flush=True)
        if vacc >= best:
            best = vacc
            torch.save({"encoder": model.encoder.state_dict(),
                        "pool": model.pool.state_dict(),
                        "emb": args.emb, "width": args.width,
                        "pretrain_classes": ncls, "val_acc": vacc, "epoch": ep},
                       os.path.join(args.out, "encoder.pt"))
    print(f"\nBest pretrain val_top1={best:.3f}. Saved encoder -> {args.out}/encoder.pt")


if __name__ == "__main__":
    main()
