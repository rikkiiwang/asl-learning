"""Train the Stage-2 recognizer (Plan 4 Tasks 5/6) — the thesis test.

Geometry-only RecognizerA first, gated to beat v1's 16.4% signer-held-out val.
Config-driven; mirrors the detector/landmark trainers (AdamW, label smoothing,
warmup→cosine, early stop on val top-1).

    cd model-v2 && PYTHONPATH=src .venv/bin/python -m aslv2.recog.train \
        --config configs/recog_a.yaml 2>&1 | tee artifacts/checkpoints/recog_a/train.log
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aslv2.recog.data import RecogDataset
from aslv2.recog.model import RecognizerA

REPO = Path(__file__).resolve().parents[4]
DEFAULT_SPLITS = str(REPO / "model" / "artifacts" / "manifest" / "signer_splits.json")


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def lr_scale(ep, warm, total):
    if ep < warm:
        return (ep + 1) / max(warm, 1)
    p = (ep - warm) / max(total - warm, 1)
    return 0.5 * (1.0 + math.cos(math.pi * p))


@torch.no_grad()
def top1(model, loader, dev):
    model.eval()
    correct = n = 0
    for g, _c, y in loader:
        pred = model(g.to(dev)).argmax(1).cpu()
        correct += int((pred == y).sum()); n += len(y)
    return correct / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description="Train the Constellation recognizer.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if cfg.get("variant", "a") != "a":
        raise SystemExit("only variant 'a' is implemented; B arrives in Task 6")

    set_seed(cfg["seed"]); dev = device(); print(f"device={dev}")
    cache = cfg["cache"]; splits = cfg.get("signer_splits", DEFAULT_SPLITS)

    tr = RecogDataset(cache, "train", splits, train=True, kp_jitter=cfg["kp_jitter"])
    va = RecogDataset(cache, "val", splits, train=False)
    te = RecogDataset(cache, "test", splits, train=False)
    print(f"train={len(tr)}  val={len(va)}  test={len(te)}")

    nw = cfg.get("num_workers", 0)
    dl_tr = DataLoader(tr, batch_size=cfg["batch_size"], shuffle=True, num_workers=nw, drop_last=True)
    dl_va = DataLoader(va, batch_size=cfg["batch_size"], num_workers=nw)
    dl_te = DataLoader(te, batch_size=cfg["batch_size"], num_workers=nw)

    model = RecognizerA(n_classes=cfg["n_classes"], emb=cfg["emb"], head=cfg["head"],
                        dropout=cfg["dropout"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    lossf = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.1))

    out_dir = cfg["out_dir"]; os.makedirs(out_dir, exist_ok=True)
    warm, total, patience_lim = cfg["warmup_epochs"], cfg["epochs"], cfg["early_stop_patience"]
    history, best, best_ep, patience = [], -1.0, -1, 0

    for ep in range(total):
        for g in opt.param_groups:
            g["lr"] = cfg["lr"] * lr_scale(ep, warm, total)
        model.train(); ep_loss = nb = 0.0
        for geom, _c, y in dl_tr:
            geom, y = geom.to(dev), y.to(dev)
            opt.zero_grad()
            loss = lossf(model(geom), y)
            loss.backward(); opt.step()
            ep_loss += loss.item(); nb += 1
        val1 = top1(model, dl_va, dev)
        lr_now = opt.param_groups[0]["lr"]
        print(f"ep {ep:02d}  loss {ep_loss/max(nb,1):.4f}  val_top1 {val1:.4f}  lr {lr_now:.2e}")
        history.append({"epoch": ep, "train_loss": ep_loss / max(nb, 1), "val_top1": val1, "lr": lr_now})
        if val1 > best:
            best, best_ep, patience = val1, ep, 0
            torch.save({"state_dict": model.state_dict(), "cfg": cfg,
                        "val_top1": val1, "epoch": ep}, os.path.join(out_dir, "best.pt"))
        else:
            patience += 1
            if patience >= patience_lim:
                print(f"early stop @ ep {ep} (best {best:.4f} @ ep {best_ep})"); break

    # reload best, report test
    ck = torch.load(os.path.join(out_dir, "best.pt"), map_location=dev, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    test1 = top1(model, dl_te, dev)
    json.dump({"history": history, "best_epoch": best_ep, "best_val_top1": best,
               "test_top1": test1}, open(os.path.join(out_dir, "history.json"), "w"), indent=2)
    print(f"\nDONE  best val_top1={best:.4f} @ ep{best_ep}  |  test_top1={test1:.4f}")
    print(f"v1 baseline was 16.4% val / 18% test — beat it?  val {'YES' if best>0.164 else 'no'}")


if __name__ == "__main__":
    main()
