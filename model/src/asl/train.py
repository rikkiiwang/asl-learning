"""Phase 2 — train the from-scratch sign classifier.

Reproducible: every run is driven by a YAML config; seeds fixed; train/val curves
and the best checkpoint are written to the config's out_dir. Early stopping on
signer-held-out val top-1.

Usage:
    python -m asl.train --config configs/baseline.yaml
    python -m asl.train --config configs/baseline.yaml --head transformer
"""
import argparse
import json
import math
import os
import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from asl.dataset import ClipDataset
from asl.model import build


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    correct = total = 0
    for x, y in loader:
        logits = model(x.to(dev))
        correct += (logits.argmax(1).cpu() == y).sum().item()
        total += len(y)
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--head", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.head:
        cfg["head"] = args.head
    if args.epochs:
        cfg["epochs"] = args.epochs

    set_seed(cfg["seed"])
    dev = device()
    print(f"device={dev}  head={cfg['head']}")

    n_classes = len(json.load(open(cfg["manifest"]))["labels"])
    tr = ClipDataset(cfg["cache"], "train", cfg["norm"], train=True)
    va = ClipDataset(cfg["cache"], "val", cfg["norm"], train=False)
    te = ClipDataset(cfg["cache"], "test", cfg["norm"], train=False)
    print(f"classes={n_classes}  train={len(tr)} val={len(va)} test={len(te)}")

    dl_tr = DataLoader(tr, cfg["batch_size"], shuffle=True, num_workers=4, drop_last=True)
    dl_va = DataLoader(va, cfg["batch_size"], shuffle=False, num_workers=2)
    dl_te = DataLoader(te, cfg["batch_size"], shuffle=False, num_workers=2)

    model = build(n_classes, emb=cfg["emb"], head=cfg["head"],
                  tf_layers=cfg["tf_layers"], tf_heads=cfg["tf_heads"],
                  dropout=cfg["dropout"], width=cfg.get("width", 48)).to(dev)
    pre = cfg.get("pretrained_encoder")
    if pre and os.path.exists(pre):
        ck = torch.load(pre, map_location=dev)
        model.encoder.load_state_dict(ck["encoder"])
        if "pool" in ck:
            model.pool.load_state_dict(ck["pool"])
        print(f"loaded pretrained encoder from {pre} "
              f"(pretrain val {ck.get('val_acc')}, {ck.get('pretrain_classes')} classes)")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    crit = torch.nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    warm, total = cfg["warmup_epochs"], cfg["epochs"]

    def lr_at(ep):
        if ep < warm:
            return (ep + 1) / warm
        p = (ep - warm) / max(total - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * p))

    os.makedirs(cfg["out_dir"], exist_ok=True)
    history, best, best_ep, patience = [], 0.0, -1, 0
    for ep in range(total):
        for g in opt.param_groups:
            g["lr"] = cfg["lr"] * lr_at(ep)
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
        history.append({"epoch": ep, "train_loss": tl, "val_acc": vacc,
                        "lr": opt.param_groups[0]["lr"]})
        print(f"ep {ep:02d}  loss {tl:.3f}  val_top1 {vacc:.3f}")
        if vacc > best:
            best, best_ep, patience = vacc, ep, 0
            torch.save({"state_dict": model.state_dict(), "cfg": cfg,
                        "val_acc": vacc, "epoch": ep},
                       os.path.join(cfg["out_dir"], "best.pt"))
        else:
            patience += 1
            if patience >= cfg["early_stop_patience"]:
                print(f"early stop at ep {ep} (best val {best:.3f} @ ep {best_ep})")
                break

    ck = torch.load(os.path.join(cfg["out_dir"], "best.pt"), map_location=dev)
    model.load_state_dict(ck["state_dict"])
    test_acc = evaluate(model, dl_te, dev)
    json.dump({"history": history, "best_val": best, "best_epoch": best_ep,
               "test_acc": test_acc},
              open(os.path.join(cfg["out_dir"], "history.json"), "w"), indent=2)
    print(f"\nBEST val_top1={best:.3f} @ ep{best_ep}   TEST top1={test_acc:.3f}")


if __name__ == "__main__":
    main()
