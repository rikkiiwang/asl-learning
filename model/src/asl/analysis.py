"""Error analysis for a trained checkpoint: per-split accuracy, per-class
accuracy, confusion matrix, and the most-confused class pairs.

Reusable by both the CLI and the story notebook.

Usage:
    python -m asl.analysis --ckpt artifacts/checkpoints/baseline/best.pt
"""
import argparse
import json
import re

import numpy as np
import torch
from torch.utils.data import DataLoader

from asl.dataset import ClipDataset
from asl.model import build


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(ckpt_path, num_classes, dev=None):
    dev = dev or _device()
    ck = torch.load(ckpt_path, map_location=dev)
    cfg = ck["cfg"]
    model = build(num_classes, emb=cfg["emb"], head=cfg["head"],
                  tf_layers=cfg["tf_layers"], tf_heads=cfg["tf_heads"],
                  dropout=cfg["dropout"], width=cfg.get("width", 48)).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, cfg


@torch.no_grad()
def predict_split(model, cache, norm, split, dev=None):
    dev = dev or _device()
    ds = ClipDataset(cache, split, norm, train=False)
    dl = DataLoader(ds, batch_size=64, shuffle=False)
    preds, trues = [], []
    for x, y in dl:
        preds.append(model(x.to(dev)).argmax(1).cpu().numpy())
        trues.append(y.numpy())
    if not preds:
        return np.array([]), np.array([])
    return np.concatenate(preds), np.concatenate(trues)


def run_analysis(ckpt="artifacts/checkpoints/baseline/best.pt",
                 manifest="artifacts/manifest/manifest.json",
                 cache="artifacts/cache/clips.npz",
                 norm="artifacts/manifest/norm.json"):
    labels = json.load(open(manifest))["labels"]
    n = len(labels)
    model, cfg = load_model(ckpt, n)

    acc, preds = {}, {}
    for sp in ("train", "val", "test"):
        p, t = predict_split(model, cache, norm, sp)
        acc[sp] = float((p == t).mean()) if len(t) else float("nan")
        preds[sp] = (p, t)

    # per-class accuracy + confusion matrix on TEST (held-out, stable-ish)
    p, t = preds["test"]
    per_class = np.full(n, np.nan)
    for c in range(n):
        m = t == c
        if m.any():
            per_class[c] = (p[m] == c).mean()
    cm = np.zeros((n, n), dtype=int)
    for ti, pi in zip(t, p):
        cm[ti, pi] += 1

    # top confused off-diagonal pairs (count true=i predicted=j)
    pairs = []
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], labels[i], labels[j]))
    pairs.sort(reverse=True)

    return {"labels": labels, "cfg": cfg, "acc": acc, "per_class": per_class,
            "cm": cm, "top_confused": pairs[:10]}


def parse_log(path):
    """Pull (epoch, train_loss, val_acc) tuples from a train log."""
    rows = []
    for line in open(path):
        m = re.match(r"ep (\d+)\s+loss ([\d.]+)\s+val_top1 ([\d.]+)", line)
        if m:
            rows.append((int(m[1]), float(m[2]), float(m[3])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/checkpoints/baseline/best.pt")
    args = ap.parse_args()
    r = run_analysis(ckpt=args.ckpt)
    print("accuracy:", {k: round(v, 3) for k, v in r["acc"].items()})
    pc = r["per_class"][~np.isnan(r["per_class"])]
    print(f"per-class test acc: mean {pc.mean():.3f}  "
          f"zero-acc classes {(pc == 0).sum()}/{len(pc)}")
    print("top-10 confused (count  true -> pred):")
    for cnt, a, b in r["top_confused"]:
        print(f"  {cnt:3d}  {a} -> {b}")


if __name__ == "__main__":
    main()
