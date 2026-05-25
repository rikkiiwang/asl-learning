"""Tests for the COCO-WholeBody adapter (Plan 6 Phase 1)."""
from aslv2.landmark.cocowholebody import adapt_cocowholebody


def _kpts(vis_count, x0=100, y0=100):
    """21×[x,y,v]; first `vis_count` keypoints labeled (v=2), rest v=0."""
    out = []
    for i in range(21):
        out += [x0 + i, y0 + i, 2 if i < vis_count else 0]
    return out


def _coco():
    return {
        "images": [{"id": 1, "file_name": "000000000001.jpg"}],
        "annotations": [{
            "image_id": 1,
            # left hand: valid, big box, all 21 visible  -> landmark + detector
            "lefthand_valid": True, "lefthand_box": [100, 100, 80, 80],
            "lefthand_kpts": _kpts(21, 100, 100),
            # right hand: valid, big box, only 10 visible -> detector ONLY
            "righthand_valid": True, "righthand_box": [300, 100, 70, 70],
            "righthand_kpts": _kpts(10, 300, 100),
        }],
    }


def test_landmark_needs_all_21_detector_takes_valid_boxes():
    lm, det = adapt_cocowholebody(_coco(), image_prefix="coco_wholebody/val2017")
    # landmark: only the fully-labeled left hand
    assert len(lm) == 1
    assert lm[0]["image"] == "coco_wholebody/val2017/000000000001.jpg"
    assert lm[0]["box"] == [100, 100, 180, 180]          # xywh -> xyxy
    assert len(lm[0]["keypoints"]) == 21
    # detector: both valid hands on the one image
    assert len(det) == 1 and det[0]["labels"] == [0, 0]
    assert det[0]["boxes"][0] == [100, 100, 180, 180]


def test_small_and_invalid_hands_filtered():
    coco = {
        "images": [{"id": 5, "file_name": "x.jpg"}],
        "annotations": [{
            "image_id": 5,
            "lefthand_valid": False, "lefthand_box": [0, 0, 80, 80], "lefthand_kpts": _kpts(21),
            "righthand_valid": True, "righthand_box": [0, 0, 10, 10],  # too small
            "righthand_kpts": _kpts(21),
        }],
    }
    lm, det = adapt_cocowholebody(coco)
    assert lm == [] and det == []      # invalid + too-small dropped from both
