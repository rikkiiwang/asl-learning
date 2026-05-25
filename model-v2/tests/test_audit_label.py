"""Tests for the makesense.ai COCO-JSON / VOC-XML → audit-slice converters."""
from aslv2.audit_label import coco_to_audit, voc_to_audit


def _voc(filename, objects):
    objs = "".join(
        f"<object><name>{n}</name><bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin>"
        f"<xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object>"
        for (n, x1, y1, x2, y2) in objects
    )
    return f"<annotation><filename>{filename}</filename>{objs}</annotation>"


def test_voc_fills_boxes_by_basename_xyxy():
    xmls = [
        _voc("a.png", [("hand", 158, 174, 270, 306),
                       ("hand", 10, 10, 50, 50),
                       ("head", 289, 91, 425, 277)]),
        _voc("frames/b.png", [("head", 0, 0, 30, 30)]),  # path prefix ignored
    ]
    stub = [
        {"image": "artifacts/audit/frames/a.png"},
        {"image": "artifacts/audit/frames/b.png"},
        {"image": "artifacts/audit/frames/c.png"},  # never annotated
    ]
    out, missing = voc_to_audit(xmls, stub)

    assert out[0]["hands"] == [[158, 174, 270, 306], [10, 10, 50, 50]]  # already xyxy
    assert out[0]["head"] == [289, 91, 425, 277]
    assert out[1]["hands"] == [] and out[1]["head"] == [0, 0, 30, 30]
    assert out[2]["hands"] == [] and out[2]["head"] is None
    assert missing == ["c.png"]


def test_voc_keeps_largest_head():
    xmls = [_voc("x.png", [("head", 0, 0, 10, 10), ("head", 0, 0, 50, 50)])]
    out, missing = voc_to_audit(xmls, [{"image": "x.png"}])
    assert out[0]["head"] == [0, 0, 50, 50]
    assert missing == []


def test_fills_boxes_by_basename_and_converts_xywh():
    coco = {
        "categories": [{"id": 1, "name": "hand"}, {"id": 2, "name": "head"}],
        "images": [
            {"id": 10, "file_name": "frames/a.png"},  # path prefix differs from stub
            {"id": 11, "file_name": "b.png"},
        ],
        "annotations": [
            {"image_id": 10, "category_id": 1, "bbox": [10, 10, 40, 40]},  # hand
            {"image_id": 10, "category_id": 1, "bbox": [60, 60, 40, 40]},  # 2nd hand
            {"image_id": 10, "category_id": 2, "bbox": [20, 5, 40, 40]},   # head
            {"image_id": 11, "category_id": 2, "bbox": [0, 0, 30, 30]},    # head only
        ],
    }
    stub = [
        {"image": "artifacts/audit/frames/a.png"},
        {"image": "artifacts/audit/frames/b.png"},
        {"image": "artifacts/audit/frames/c.png"},  # never annotated
    ]
    out, missing = coco_to_audit(coco, stub)

    assert len(out) == 3
    assert out[0]["hands"] == [[10, 10, 50, 50], [60, 60, 100, 100]]  # xywh→xyxy
    assert out[0]["head"] == [20, 5, 60, 45]
    assert out[0]["keypoints"] is None
    assert out[1]["hands"] == [] and out[1]["head"] == [0, 0, 30, 30]
    assert out[2]["hands"] == [] and out[2]["head"] is None
    assert missing == ["c.png"]


def test_largest_head_kept_when_multiple():
    coco = {
        "categories": [{"id": 2, "name": "Head"}],  # case-insensitive
        "images": [{"id": 1, "file_name": "x.png"}],
        "annotations": [
            {"image_id": 1, "category_id": 2, "bbox": [0, 0, 10, 10]},   # small
            {"image_id": 1, "category_id": 2, "bbox": [0, 0, 50, 50]},   # large
        ],
    }
    out, missing = coco_to_audit(coco, [{"image": "x.png"}])
    assert out[0]["head"] == [0, 0, 50, 50]
    assert missing == []
