"""Gate the trained detector on the labeled ASL-frame audit slice (Plan 2 Task 9).

This is the BINDING domain gate (real ASL Citizen frames): detection-rate@0.5
≥ 0.80 head, ≥ 0.60 hand. Run after labeling asl_audit_slice.json and pulling
best.pt back from Colab.

  cd model-v2 && .venv/bin/python scripts/eval_detector_on_audit.py \
      --ckpt artifacts/checkpoints/detector/best.pt \
      --slice artifacts/audit/asl_audit_slice.json

Writes artifacts/checkpoints/detector/audit_eval.json and prints PASS/FAIL.
"""
import argparse
import json
from pathlib import Path

import torch

from aslv2.audit import load_audit_slice
from aslv2.detect.audit_eval import evaluate_on_audit
from aslv2.detect.model import Detector


def main() -> None:
    ap = argparse.ArgumentParser(description="ASL-audit domain gate for the detector.")
    ap.add_argument("--ckpt", default="artifacts/checkpoints/detector/best.pt")
    ap.add_argument("--slice", default="artifacts/audit/asl_audit_slice.json")
    ap.add_argument("--image-root", default=".",
                    help="dir that slice 'image' paths are relative to (default: cwd)")
    ap.add_argument("--out", default="artifacts/checkpoints/detector/audit_eval.json")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model = Detector(n_classes=2, n_anchors=cfg["n_anchors"], width=cfg["width"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    frames = load_audit_slice(args.slice)
    res = evaluate_on_audit(model, frames, cfg["norm"], args.image_root,
                            img_size=cfg["img"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))

    n_labeled = res["n_head_frames"] + res["n_hand_frames"]
    print(f"ASL-audit domain gate  (ckpt ep{ckpt.get('epoch')}, width={cfg['width']})")
    print(f"  labeled frames: {res['n_head_frames']} with head, {res['n_hand_frames']} with hand(s)")
    if n_labeled == 0:
        print("  WARNING: no labeled frames — run convert_makesense_to_audit.py first.")
    head_v = "PASS" if res["head_pass"] else "FAIL"
    hand_v = "PASS" if res["hand_pass"] else "FAIL"
    print(f"  head (anchor: center+scale) {res['head_anchor_dr']:.3f}  (gate ≥ {res['head_gate']})  {head_v}")
    print(f"  head (IoU@0.5, reference)   {res['head_iou_dr']:.3f}  — undercounts: face box vs whole-head label")
    print(f"  hand (detection-rate@0.5)   {res['hand_dr']:.3f}  (gate ≥ {res['hand_gate']})  {hand_v}")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
