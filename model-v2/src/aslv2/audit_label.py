"""Convert a makesense.ai annotation export into the labeled audit slice — fills
the `hands`/`head` boxes in each `asl_audit_slice.json` stub record.

Supports both export formats makesense offers:
  - COCO JSON  (single file; bbox is [x, y, w, h])           -> coco_to_audit
  - VOC XML    (zip / dir of per-image .xml; bbox is xyxy)    -> voc_to_audit

Matching is by image **basename**, so the export's path prefix (makesense often
stores just the filename, or a `frames/` prefix) does not have to match the
stub's `artifacts/audit/frames/...` path. Label names are matched loosely: any
name containing "hand" → hand, "head"/"face" → head.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


def _classify(name) -> str | None:
    name = str(name).strip().lower()
    if "hand" in name:
        return "hand"
    if "head" in name or "face" in name:
        return "head"
    return None


def _xywh_to_xyxy(b: list[float]) -> list[float]:
    x, y, w, h = b
    return [x, y, x + w, y + h]


def _fill_stub(boxes: dict[str, dict[str, list]], stub: list[dict]):
    """Project per-basename boxes onto the stub records.

    Returns (labeled, missing): `labeled` mirrors `stub` with hands/head filled
    (keypoints left None); `missing` lists basenames that got no boxes at all.
    """
    labeled: list[dict] = []
    missing: list[str] = []
    for fr in stub:
        base = Path(fr["image"]).name
        b = boxes.get(base, {"hand": [], "head": []})
        heads = b["head"]
        head = None
        if heads:
            # if multiple head boxes were drawn, keep the largest-area one
            head = max(heads, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
        labeled.append(
            {"image": fr["image"], "hands": b["hand"], "head": head, "keypoints": None}
        )
        if not b["hand"] and head is None:
            missing.append(base)
    return labeled, missing


def coco_to_audit(coco: dict, stub: list[dict]):
    """Fill stub records from a COCO-format annotation export (bbox = xywh)."""
    cat = {c["id"]: _classify(c["name"]) for c in coco.get("categories", [])}
    img_name = {im["id"]: Path(im["file_name"]).name for im in coco.get("images", [])}

    boxes: dict[str, dict[str, list]] = {}
    for a in coco.get("annotations", []):
        cls = cat.get(a["category_id"])
        if cls is None:
            continue
        name = img_name.get(a["image_id"])
        if name is None:
            continue
        boxes.setdefault(name, {"hand": [], "head": []})[cls].append(
            _xywh_to_xyxy(a["bbox"])
        )
    return _fill_stub(boxes, stub)


def voc_to_audit(xml_texts, stub: list[dict]):
    """Fill stub records from VOC XML exports (bndbox = xmin/ymin/xmax/ymax).

    Args:
        xml_texts: iterable of XML document strings (one per image).
        stub:      the existing asl_audit_slice.json list.
    """
    boxes: dict[str, dict[str, list]] = {}
    for text in xml_texts:
        root = ET.fromstring(text)
        fn = root.findtext("filename")
        name = Path(fn).name if fn else None
        if name is None:
            continue
        for obj in root.findall("object"):
            cls = _classify(obj.findtext("name"))
            if cls is None:
                continue
            bb = obj.find("bndbox")
            if bb is None:
                continue
            xyxy = [
                float(bb.findtext("xmin")),
                float(bb.findtext("ymin")),
                float(bb.findtext("xmax")),
                float(bb.findtext("ymax")),
            ]
            boxes.setdefault(name, {"hand": [], "head": []})[cls].append(xyxy)
    return _fill_stub(boxes, stub)
