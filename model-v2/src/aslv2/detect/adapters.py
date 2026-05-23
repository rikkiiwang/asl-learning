"""Dataset adapters: convert 100DOH + WIDER FACE into unified detector manifests.

Unified manifest format:
  [{"image": "<abs-path>", "boxes": [[x1,y1,x2,y2], ...], "labels": [0|1, ...]}]

Labels:  0 = hand (100DOH),  1 = head (WIDER FACE)
Boxes:   xyxy pixel coordinates
"""
from __future__ import annotations

import json
import os
from typing import List, Dict, Any


# ---------------------------------------------------------------------------
# 100DOH
# ---------------------------------------------------------------------------

def adapt_100doh(json_path: str, raw_dir: str) -> List[Dict[str, Any]]:
    """Convert 100DOH annotation JSON to unified manifest records.

    Args:
        json_path: path to 100DOH file/{train,val}.json
        raw_dir:   root directory where image files live (key resolves to raw_dir/key)

    Returns:
        List of manifest dicts; images with no valid boxes are excluded.
    """
    with open(json_path) as f:
        data = json.load(f)

    records: List[Dict[str, Any]] = []

    for key, annotations in data.items():
        boxes: List[List[float]] = []
        labels: List[int] = []

        for ann in annotations:
            w: float = ann["width"]
            h: float = ann["height"]
            x1 = ann["x1"] * w
            y1 = ann["y1"] * h
            x2 = ann["x2"] * w
            y2 = ann["y2"] * h

            # Skip degenerate boxes
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(0)  # hand

        if not boxes:
            continue

        records.append(
            {
                "image": os.path.join(raw_dir, key),
                "boxes": boxes,
                "labels": labels,
            }
        )

    return records


# ---------------------------------------------------------------------------
# WIDER FACE
# ---------------------------------------------------------------------------

# Minimum side length for a face box to be included (crowd-filtering heuristic)
_MIN_FACE_SIDE = 40


def adapt_widerface(gt_txt: str, images_dir: str) -> List[Dict[str, Any]]:
    """Convert WIDER FACE ground-truth text file to unified manifest records.

    Format (per image block):
        <relative/path/to/image.jpg>
        N
        x y w h blur expr illum invalid occl pose   (repeated N times)

    When N==0 the format still emits a dummy "0 0 0 0 0 0 0 0 0 0" line.

    Filters applied (face → kept only if ALL pass):
      - w > 0 and h > 0
      - invalid flag (field index 7) == 0
      - w >= 40 and h >= 40  (exclude tiny crowd faces)

    Images with no surviving boxes are dropped.

    Args:
        gt_txt:     path to wider_face_{train,val}_bbx_gt.txt
        images_dir: root directory for images (filename resolves to images_dir/filename)

    Returns:
        List of manifest dicts.
    """
    records: List[Dict[str, Any]] = []

    with open(gt_txt) as f:
        lines = [line.rstrip("\n") for line in f]

    i = 0
    n = len(lines)

    while i < n:
        # Skip blank lines
        if not lines[i].strip():
            i += 1
            continue

        filename = lines[i].strip()
        i += 1

        if i >= n:
            break

        num_faces = int(lines[i].strip())
        i += 1

        boxes: List[List[int]] = []
        labels: List[int] = []

        # Read num_faces annotation lines (always at least 1 line even when N==0)
        faces_to_read = max(num_faces, 1) if num_faces == 0 else num_faces
        for _ in range(faces_to_read):
            if i >= n:
                break
            parts = lines[i].strip().split()
            i += 1

            # When num_faces==0 there is a dummy "0 0 0 0 0 0 0 0 0 0" line — skip it
            if num_faces == 0:
                continue

            x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            invalid = int(parts[7])

            # Apply filters
            if w <= 0 or h <= 0:
                continue
            if invalid == 1:
                continue
            if w < _MIN_FACE_SIDE or h < _MIN_FACE_SIDE:
                continue

            boxes.append([x, y, x + w, y + h])
            labels.append(1)  # head

        if not boxes:
            continue

        records.append(
            {
                "image": os.path.join(images_dir, filename),
                "boxes": boxes,
                "labels": labels,
            }
        )

    return records


# ---------------------------------------------------------------------------
# write_manifest
# ---------------------------------------------------------------------------

def write_manifest(records: List[Dict[str, Any]], out_path: str) -> None:
    """Serialise manifest records to a JSON file, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f)
