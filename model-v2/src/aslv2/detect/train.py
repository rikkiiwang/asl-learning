"""Detector training script — device-agnostic (CUDA → MPS → CPU).

Usage:
    python -m aslv2.detect.train --config configs/detector.yaml
    python -m aslv2.detect.train --config configs/detector.yaml --data-root /mnt/gdrive/data/detect
    python -m aslv2.detect.train --config configs/detector.yaml --max-train 200 --max-val 100 --epochs 1

Checkpoints are saved to cfg['out_dir']/best.pt (state_dict + cfg + metrics).
Training history is saved to cfg['out_dir']/history.json.

Val metric: detection_rate@IoU0.5 for hand (class 0) and head (class 1), capped at
val_max_images (default 1500) for speed.
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

from aslv2.detect.anchors import make_anchors
from aslv2.detect.data import DetDataset
from aslv2.detect.encode import decode, encode_targets
from aslv2.detect.loss import det_loss
from aslv2.detect.model import Detector
from aslv2.metrics import detection_rate


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
# Collate: keep variable-length (boxes, labels) lists; stack images only
# ---------------------------------------------------------------------------

def _collate(batch):
    """Custom collate for variable-count bounding boxes.

    Args:
        batch: list of (tensor, boxes_np, labels_np)

    Returns:
        imgs:        (B, 3, H, W) float tensor
        raw_targets: list of (boxes_np, labels_np), length B
    """
    imgs = torch.stack([b[0] for b in batch])
    raw_targets = [(b[1], b[2]) for b in batch]
    return imgs, raw_targets


# ---------------------------------------------------------------------------
# Learning-rate schedule: warmup → cosine
# ---------------------------------------------------------------------------

def lr_scale(epoch: int, warmup: int, total: int) -> float:
    if epoch < warmup:
        return (epoch + 1) / max(warmup, 1)
    p = (epoch - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * p))


# ---------------------------------------------------------------------------
# Encode a whole batch and return stacked tensors
# ---------------------------------------------------------------------------

def encode_batch(raw_targets, anchors: np.ndarray, device: torch.device):
    """Encode per-image targets and stack into batch tensors.

    Returns:
        cls_tgt: (B, N, 2)
        box_t:   (B, N, 4)
        pos:     (B, N) bool
        valid:   (B, N) bool
    """
    cls_list, box_list, pos_list, val_list = [], [], [], []
    for boxes, labels in raw_targets:
        ct, bt, pm, vm = encode_targets(anchors, boxes, labels)
        cls_list.append(ct)
        box_list.append(bt)
        pos_list.append(pm)
        val_list.append(vm)

    cls_tgt = torch.from_numpy(np.stack(cls_list)).to(device)
    box_t   = torch.from_numpy(np.stack(box_list)).to(device)
    pos     = torch.from_numpy(np.stack(pos_list)).to(device)
    valid   = torch.from_numpy(np.stack(val_list)).to(device)
    return cls_tgt, box_t, pos, valid


# ---------------------------------------------------------------------------
# Validation: detection_rate for hand and head separately
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: Detector, loader: DataLoader,
             anchors: np.ndarray, dev: torch.device) -> dict:
    """Compute detection_rate@0.5 for hand and head over the val loader."""
    model.eval()
    hand_rates, head_rates = [], []

    for imgs, raw_targets in loader:
        imgs = imgs.to(dev)
        cls_logits, box_pred = model(imgs)  # (B, 192, 2), (B, 192, 4)

        B = imgs.shape[0]
        for i in range(B):
            cls_np = cls_logits[i].cpu().float().numpy()
            box_np = np.clip(box_pred[i].cpu().float().numpy(), -10.0, 10.0)

            pred_boxes, pred_scores, pred_labels = decode(
                anchors, box_np, cls_np, score_thr=0.3, iou_thr=0.45
            )

            boxes_gt, labels_gt = raw_targets[i]

            # Hand recall
            gt_hands = boxes_gt[labels_gt == 0]
            pred_hands = pred_boxes[pred_labels == 0] if len(pred_boxes) > 0 else np.zeros((0, 4))
            hand_rates.append(detection_rate(pred_hands, gt_hands, iou_thr=0.5))

            # Head recall
            gt_heads = boxes_gt[labels_gt == 1]
            pred_heads = pred_boxes[pred_labels == 1] if len(pred_boxes) > 0 else np.zeros((0, 4))
            head_rates.append(detection_rate(pred_heads, gt_heads, iou_thr=0.5))

    return {
        "hand_dr": float(np.mean(hand_rates)) if hand_rates else 0.0,
        "head_dr": float(np.mean(head_rates)) if head_rates else 0.0,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Train the ASL detector.")
    ap.add_argument("--config",    required=True,  help="Path to YAML config")
    ap.add_argument("--data-root", default=None,    help="Override cfg data_root")
    ap.add_argument("--epochs",    type=int, default=None, help="Override cfg epochs")
    ap.add_argument("--max-train", type=int, default=None,
                    help="Cap training set to N images (smoke / CI use only)")
    ap.add_argument("--max-val",   type=int, default=None,
                    help="Cap val set to N images (overrides cfg val_max_images)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))

    # CLI overrides
    if args.data_root is not None:
        cfg["data_root"] = args.data_root
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.max_val is not None:
        cfg["val_max_images"] = args.max_val

    set_seed(cfg["seed"])
    dev = device()
    print(f"device={dev}")

    # Anchors (shared across train/val)
    anchors = make_anchors(img=cfg["img"], stride=16, scales=(32, 64, 96))

    # Datasets
    norm = cfg["norm"]
    data_root = cfg.get("data_root", "")

    train_ds = DetDataset(cfg["train_manifest"], norm=norm, train=True,
                          data_root=data_root)
    val_ds   = DetDataset(cfg["val_manifest"],   norm=norm, train=False,
                          data_root=data_root)

    # Optional caps (smoke / sanity runs)
    if args.max_train is not None and args.max_train < len(train_ds):
        idxs = list(range(args.max_train))
        train_ds = Subset(train_ds, idxs)

    val_cap = cfg.get("val_max_images", 1500)
    if val_cap < len(val_ds):
        rng = random.Random(cfg["seed"])
        val_idxs = rng.sample(range(len(val_ds)), val_cap)
        val_ds = Subset(val_ds, val_idxs)

    print(f"train={len(train_ds)}  val={len(val_ds)}")

    # DataLoaders with custom collate (variable-length boxes)
    num_workers = cfg.get("num_workers", 4)
    dl_tr = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=num_workers, drop_last=True, collate_fn=_collate,
    )
    dl_va = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=num_workers, collate_fn=_collate,
    )

    # Model + optimizer
    model = Detector(
        n_classes=2,
        n_anchors=cfg["n_anchors"],
        width=cfg["width"],
    ).to(dev)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    warm  = cfg["warmup_epochs"]
    total = cfg["epochs"]
    patience_limit = cfg["early_stop_patience"]

    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    history: list[dict] = []
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

        for imgs, raw_targets in dl_tr:
            imgs = imgs.to(dev)
            cls_tgt, box_t, pos, valid = encode_batch(raw_targets, anchors, dev)

            opt.zero_grad()
            cls_logits, box_pred = model(imgs)

            B = imgs.shape[0]
            loss = sum(
                det_loss(
                    cls_logits[i], box_pred[i],
                    cls_tgt[i], box_t[i],
                    pos[i], valid[i],
                )
                for i in range(B)
            ) / B

            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # ---- val ----
        val_metrics = evaluate(model, dl_va, anchors, dev)
        # Combined score: mean of hand_dr + head_dr
        score = (val_metrics["hand_dr"] + val_metrics["head_dr"]) / 2.0

        lr_now = opt.param_groups[0]["lr"]
        print(
            f"ep {ep:02d}  loss {avg_loss:.4f}  "
            f"hand_dr {val_metrics['hand_dr']:.3f}  "
            f"head_dr {val_metrics['head_dr']:.3f}  "
            f"lr {lr_now:.2e}"
        )

        history.append({
            "epoch":    ep,
            "train_loss": avg_loss,
            "hand_dr":  val_metrics["hand_dr"],
            "head_dr":  val_metrics["head_dr"],
            "lr":       lr_now,
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
                print(f"early stop at ep {ep}  (best score {best_score:.3f} @ ep {best_ep})")
                break

    json.dump(
        {"history": history, "best_epoch": best_ep, "best_score": best_score},
        open(os.path.join(out_dir, "history.json"), "w"),
        indent=2,
    )
    print(f"\nDONE  best_score={best_score:.3f} @ ep{best_ep}")


if __name__ == "__main__":
    main()
