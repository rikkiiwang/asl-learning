"""COCO-WholeBody adapter — in-the-wild 21-keypoint hand labels + hand boxes.

Each COCO-WholeBody person annotation carries up to two hands:
    {side}_valid : bool
    {side}_kpts  : 63 floats = 21 × [x, y, v]   (v>0 ⇒ labeled)
    {side}_box   : [x, y, w, h]
for side in ('lefthand', 'righthand').

`adapt_cocowholebody` emits two manifests from one parse, both in the schemas the
existing pipelines already consume:
  - landmark : {image, box(xyxy), keypoints:[[x,y]×21]}  — only hands with ALL 21
               keypoints labeled (clean regression targets) and a big-enough box.
  - detector : {image, boxes:[xyxy…], labels:[0,…]}      — hand class (0); any
               valid, big-enough hand box (occluded fingers OK — the box is fine).
"""
from __future__ import annotations

import numpy as np

_SIDES = ("lefthand", "righthand")


def _xywh_to_xyxy(b):
    x, y, w, h = b
    return [float(x), float(y), float(x + w), float(y + h)]


def adapt_cocowholebody(coco: dict, image_prefix: str = "",
                        min_box: float = 30.0):
    """Return (landmark_records, detector_records). See module docstring."""
    images = {im["id"]: im["file_name"] for im in coco.get("images", [])}
    lm: list[dict] = []
    det_by_img: dict[str, list] = {}

    for a in coco.get("annotations", []):
        fn = images.get(a.get("image_id"))
        if fn is None:
            continue
        img = f"{image_prefix}/{fn}" if image_prefix else fn
        for side in _SIDES:
            if not a.get(f"{side}_valid"):
                continue
            box = _xywh_to_xyxy(a[f"{side}_box"])
            if min(box[2] - box[0], box[3] - box[1]) < min_box:
                continue
            # detector hand box: valid + big enough (occlusion OK)
            det_by_img.setdefault(img, []).append(box)
            # landmark: only if all 21 keypoints are labeled (v>0) → clean targets
            kp = np.asarray(a[f"{side}_kpts"], dtype=float).reshape(21, 3)
            if int((kp[:, 2] > 0).sum()) == 21:
                lm.append({"image": img, "box": box, "keypoints": kp[:, :2].tolist()})

    det = [{"image": img, "boxes": boxes, "labels": [0] * len(boxes)}
           for img, boxes in det_by_img.items()]
    return lm, det
