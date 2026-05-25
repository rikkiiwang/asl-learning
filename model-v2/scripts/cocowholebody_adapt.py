"""Adapt COCO-WholeBody → landmark + detector manifests (Plan 6 Phase 1).

Run after downloading COCO-WholeBody (see Plan 6 / DATA.md):
  data/landmark/coco_wholebody/annotations/coco_wholebody_val_v1.0.json
  data/landmark/coco_wholebody/val2017/*.jpg

  cd model-v2 && PYTHONPATH=src .venv/bin/python scripts/cocowholebody_adapt.py \
      --wb-json data/landmark/coco_wholebody/annotations/coco_wholebody_val_v1.0.json \
      --image-prefix coco_wholebody/val2017

Writes (image paths relative to --data-root so KpDataset/DetDataset resolve them):
  artifacts/landmark/cocowb_landmark.json   {image, box, keypoints}
  artifacts/detect/cocowb_hands.json        {image, boxes, labels:[0,...]}
"""
import argparse
import json
import os
from pathlib import Path

from aslv2.landmark.cocowholebody import adapt_cocowholebody


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wb-json", required=True, help="coco_wholebody_*_v1.0.json")
    ap.add_argument("--image-prefix", required=True,
                    help="path (relative to data-root) to the images dir, e.g. coco_wholebody/val2017")
    ap.add_argument("--min-box", type=float, default=30.0)
    ap.add_argument("--lm-out", default="artifacts/landmark/cocowb_landmark.json")
    ap.add_argument("--det-out", default="artifacts/detect/cocowb_hands.json")
    args = ap.parse_args()

    coco = json.loads(Path(args.wb_json).read_text())
    lm, det = adapt_cocowholebody(coco, image_prefix=args.image_prefix, min_box=args.min_box)

    for path, data in [(args.lm_out, lm), (args.det_out, det)]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Path(path).write_text(json.dumps(data))

    n_hand_boxes = sum(len(d["boxes"]) for d in det)
    print(f"landmark records (all-21 hands): {len(lm)}  -> {args.lm_out}")
    print(f"detector images: {len(det)} with {n_hand_boxes} hand boxes  -> {args.det_out}")
    print("Next: I merge these with FreiHAND/100DOH, shrink, package for Colab.")


if __name__ == "__main__":
    main()
