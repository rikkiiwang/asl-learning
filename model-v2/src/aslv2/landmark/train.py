"""Landmark training script — device-agnostic (CUDA → MPS → CPU).

Usage:
    python -m aslv2.landmark.train --config configs/landmark.yaml
    python -m aslv2.landmark.train --config configs/landmark.yaml \\
        --data-root /mnt/gdrive/data/landmark_small
    python -m aslv2.landmark.train --config configs/landmark.yaml \\
        --max-train 200 --max-val 50 --epochs 1

Checkpoints are saved to cfg['out_dir']/best.pt (state_dict + cfg + metrics).
Training history is saved to cfg['out_dir']/history.json.

Val metric: PCK@0.2 with ref_size = crop size (64 px, per KpDataset spec).
            PCK is computed in [0,1] normalised crop-space, so the threshold
            is thr_frac * 1.0 = 0.2 (i.e. within 20% of the 64-px crop side).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from aslv2.landmark.data import KpDataset
from aslv2.landmark.loss import wing_loss
from aslv2.landmark.model import Landmark
from aslv2.metrics import pck


# ---------------------------------------------------------------------------
# Device selection: cuda → mps → cpu
# ---------------------------------------------------------------------------

def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


# ---------------------------------------------------------------------------
# Learning-rate schedule: warmup → cosine  (mirrors detect/train.py)
# ---------------------------------------------------------------------------

def lr_scale(epoch: int, warmup: int, total: int) -> float:
    if epoch < warmup:
        return (epoch + 1) / max(warmup, 1)
    p = (epoch - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * p))


# ---------------------------------------------------------------------------
# Validation: PCK@0.2 in normalised [0,1] crop-space
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: Landmark, loader: DataLoader, dev: torch.device) -> dict:
    """Compute PCK@0.1 (strict gate), PCK@0.2 (legacy), and mean per-keypoint
    error — all in normalised [0,1] crop-space. PCK@0.1 is the honest metric;
    PCK@0.2 was lenient enough to pass a near-mean-hand predictor."""
    model.eval()
    pck10: list[float] = []
    pck20: list[float] = []
    errs:  list[float] = []

    for imgs, kps_gt in loader:
        imgs   = imgs.to(dev)
        kps_gt = kps_gt.to(dev)

        pred = model(imgs)   # (B, 21, 2) in [0, 1]

        B = imgs.shape[0]
        for i in range(B):
            p_np = pred[i].cpu().float().numpy()     # (21, 2)
            g_np = kps_gt[i].cpu().float().numpy()   # (21, 2)
            pck10.append(pck(p_np, g_np, ref_size=1.0, thr_frac=0.1))
            pck20.append(pck(p_np, g_np, ref_size=1.0, thr_frac=0.2))
            errs.append(float(np.linalg.norm(p_np - g_np, axis=1).mean()))

    return {
        "pck_01":   float(np.mean(pck10)) if pck10 else 0.0,
        "pck_02":   float(np.mean(pck20)) if pck20 else 0.0,
        "mean_err": float(np.mean(errs))  if errs  else 1.0,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Train the ASL hand-landmark model.")
    ap.add_argument("--config",    required=True,  help="Path to YAML config")
    ap.add_argument("--data-root", default=None,   help="Override cfg data_root")
    ap.add_argument("--epochs",    type=int, default=None, help="Override cfg epochs")
    ap.add_argument("--max-train", type=int, default=None,
                    help="Cap training set to N images (smoke / CI use only)")
    ap.add_argument("--max-val",   type=int, default=None,
                    help="Cap val set to N images")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))

    # CLI overrides
    if args.data_root is not None:
        cfg["data_root"] = args.data_root
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    set_seed(cfg["seed"])
    dev = device()
    print(f"device={dev}")

    norm     = cfg["norm"]
    data_root = cfg.get("data_root", "")

    train_ds = KpDataset(cfg["train_manifest"], norm=norm, train=True,
                         data_root=data_root)
    val_ds   = KpDataset(cfg["val_manifest"],   norm=norm, train=False,
                         data_root=data_root)

    # Optional caps (smoke / sanity runs)
    if args.max_train is not None and args.max_train < len(train_ds):
        train_ds = Subset(train_ds, list(range(args.max_train)))

    if args.max_val is not None and args.max_val < len(val_ds):
        rng = random.Random(cfg["seed"])
        val_ds = Subset(val_ds, rng.sample(range(len(val_ds)), args.max_val))

    print(f"train={len(train_ds)}  val={len(val_ds)}")

    num_workers = cfg.get("num_workers", 4)
    dl_tr = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=num_workers, drop_last=True,
    )
    dl_va = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=num_workers,
    )

    model = Landmark(width=cfg["width"]).to(dev)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    warm           = cfg["warmup_epochs"]
    total          = cfg["epochs"]
    patience_limit = cfg["early_stop_patience"]

    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    history:    list[dict] = []
    best_score = -1.0
    best_ep    = -1
    patience   = 0

    for ep in range(total):
        # Set LR for this epoch
        scale = lr_scale(ep, warm, total)
        for g in opt.param_groups:
            g["lr"] = cfg["lr"] * scale

        # ---- train ----
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for imgs, kps_gt in dl_tr:
            imgs   = imgs.to(dev)
            kps_gt = kps_gt.to(dev)

            opt.zero_grad()
            pred = model(imgs)            # (B, 21, 2)
            loss = wing_loss(pred, kps_gt)

            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # ---- val ----
        val_metrics = evaluate(model, dl_va, dev)
        score = val_metrics["pck_01"]   # gate on the STRICT metric now

        lr_now = opt.param_groups[0]["lr"]
        print(
            f"ep {ep:02d}  loss {avg_loss:.4f}  "
            f"pck@0.1 {score:.3f}  pck@0.2 {val_metrics['pck_02']:.3f}  "
            f"err {val_metrics['mean_err']:.3f}  lr {lr_now:.2e}"
        )

        history.append({
            "epoch":      ep,
            "train_loss": avg_loss,
            "pck_01":     score,
            "pck_02":     val_metrics["pck_02"],
            "mean_err":   val_metrics["mean_err"],
            "lr":         lr_now,
        })

        # ---- checkpoint ----
        if score > best_score:
            best_score = score
            best_ep    = ep
            patience   = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "cfg":        cfg,
                    "metrics":    val_metrics,
                    "epoch":      ep,
                },
                os.path.join(out_dir, "best.pt"),
            )
        else:
            patience += 1
            if patience >= patience_limit:
                print(
                    f"early stop at ep {ep}  "
                    f"(best score {best_score:.3f} @ ep {best_ep})"
                )
                break

    json.dump(
        {"history": history, "best_epoch": best_ep, "best_score": best_score},
        open(os.path.join(out_dir, "history.json"), "w"),
        indent=2,
    )
    print(f"\nDONE  best_score={best_score:.3f} @ ep{best_ep}")


if __name__ == "__main__":
    main()
