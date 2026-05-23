"""Tests for 100DOH + WIDER FACE → unified manifest adapters.

TDD: tests written BEFORE implementation.
"""
import json
import os
import pytest


# ---------------------------------------------------------------------------
# 100DOH adapter
# ---------------------------------------------------------------------------

class TestAdapt100doh:
    def test_basic_box_denormalisation(self, tmp_path):
        """A hand at ratio (0.25,0.5,0.75,1.0) on a 200×100 image → pixel box [50,50,150,100]."""
        from aslv2.detect.adapters import adapt_100doh

        key = "trainval/study/frame_001.jpg"
        annotation = [
            {
                "x1": 0.25, "y1": 0.5, "x2": 0.75, "y2": 1.0,
                "width": 200, "height": 100,
                "contact_state": 0, "hand_side": "r",
                "obj_bbox": {"x1": 0, "y1": 0, "x2": 1, "y2": 1},
            }
        ]
        data = {key: annotation}
        json_path = tmp_path / "train.json"
        json_path.write_text(json.dumps(data))

        raw_dir = str(tmp_path / "raw")
        records = adapt_100doh(str(json_path), raw_dir)

        assert len(records) == 1
        rec = records[0]
        assert rec["boxes"] == [[50.0, 50.0, 150.0, 100.0]]
        assert rec["labels"] == [0]
        assert rec["image"].endswith(key)

    def test_image_path_uses_raw_dir(self, tmp_path):
        """image field = os.path.join(raw_dir, key)."""
        from aslv2.detect.adapters import adapt_100doh

        key = "trainval/cooking/vid_frame000.jpg"
        data = {
            key: [{"x1": 0.1, "y1": 0.1, "x2": 0.9, "y2": 0.9,
                   "width": 640, "height": 480,
                   "contact_state": 0, "hand_side": "l",
                   "obj_bbox": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}}]
        }
        json_path = tmp_path / "train.json"
        json_path.write_text(json.dumps(data))
        raw_dir = "/some/abs/raw"

        records = adapt_100doh(str(json_path), raw_dir)

        assert records[0]["image"] == os.path.join(raw_dir, key)

    def test_degenerate_box_skipped(self, tmp_path):
        """Boxes with x2<=x1 or y2<=y1 are skipped; images with no valid boxes are dropped."""
        from aslv2.detect.adapters import adapt_100doh

        key = "trainval/study/bad_frame.jpg"
        data = {
            key: [
                # degenerate: x2 == x1
                {"x1": 0.5, "y1": 0.2, "x2": 0.5, "y2": 0.8,
                 "width": 100, "height": 100,
                 "contact_state": 0, "hand_side": "l",
                 "obj_bbox": {}},
                # degenerate: y2 < y1
                {"x1": 0.1, "y1": 0.9, "x2": 0.5, "y2": 0.1,
                 "width": 100, "height": 100,
                 "contact_state": 0, "hand_side": "r",
                 "obj_bbox": {}},
            ]
        }
        json_path = tmp_path / "train.json"
        json_path.write_text(json.dumps(data))

        records = adapt_100doh(str(json_path), "/raw")
        # All boxes degenerate → image should be dropped
        assert records == []

    def test_multiple_hands_per_image(self, tmp_path):
        """Multiple hand boxes → all appear in one record."""
        from aslv2.detect.adapters import adapt_100doh

        key = "trainval/study/multi.jpg"
        data = {
            key: [
                {"x1": 0.0, "y1": 0.0, "x2": 0.5, "y2": 0.5,
                 "width": 100, "height": 100,
                 "contact_state": 0, "hand_side": "l", "obj_bbox": {}},
                {"x1": 0.5, "y1": 0.5, "x2": 1.0, "y2": 1.0,
                 "width": 100, "height": 100,
                 "contact_state": 0, "hand_side": "r", "obj_bbox": {}},
            ]
        }
        json_path = tmp_path / "train.json"
        json_path.write_text(json.dumps(data))

        records = adapt_100doh(str(json_path), "/raw")
        assert len(records) == 1
        assert len(records[0]["boxes"]) == 2
        assert records[0]["labels"] == [0, 0]


# ---------------------------------------------------------------------------
# WIDER FACE adapter
# ---------------------------------------------------------------------------

class TestAdaptWiderface:
    def _write_gt(self, tmp_path, content: str):
        gt = tmp_path / "wider_face_train_bbx_gt.txt"
        gt.write_text(content)
        return str(gt)

    def test_kept_face_converted_to_xyxy(self, tmp_path):
        """A valid face 60 60 50 50 → [60, 60, 110, 110] with label 1."""
        from aslv2.detect.adapters import adapt_widerface

        gt_content = (
            "0--Parade/0_Parade_test_0001.jpg\n"
            "1\n"
            "60 60 50 50 0 0 0 0 0 0\n"
        )
        gt_txt = self._write_gt(tmp_path, gt_content)
        images_dir = str(tmp_path / "images")

        records = adapt_widerface(gt_txt, images_dir)

        assert len(records) == 1
        rec = records[0]
        assert rec["boxes"] == [[60, 60, 110, 110]]
        assert rec["labels"] == [1]
        assert rec["image"] == os.path.join(images_dir, "0--Parade/0_Parade_test_0001.jpg")

    def test_tiny_face_filtered_out(self, tmp_path):
        """A tiny face (w<40 or h<40) is filtered; image with 0 kept faces is dropped."""
        from aslv2.detect.adapters import adapt_widerface

        gt_content = (
            "0--Parade/0_Parade_test_tiny.jpg\n"
            "2\n"
            "60 60 50 50 0 0 0 0 0 0\n"   # kept
            "10 10 5 5 0 0 0 0 0 0\n"     # too small → filtered
        )
        gt_txt = self._write_gt(tmp_path, gt_content)

        records = adapt_widerface(gt_txt, str(tmp_path / "images"))

        assert len(records) == 1
        assert len(records[0]["boxes"]) == 1
        assert records[0]["boxes"] == [[60, 60, 110, 110]]

    def test_all_tiny_faces_drops_image(self, tmp_path):
        """An image where all faces are tiny is dropped from the manifest."""
        from aslv2.detect.adapters import adapt_widerface

        gt_content = (
            "0--Parade/0_Parade_all_tiny.jpg\n"
            "1\n"
            "10 10 5 5 0 0 0 0 0 0\n"
        )
        gt_txt = self._write_gt(tmp_path, gt_content)

        records = adapt_widerface(gt_txt, str(tmp_path / "images"))
        assert records == []

    def test_invalid_flag_drops_face(self, tmp_path):
        """A face with invalid==1 (field index 7) is dropped."""
        from aslv2.detect.adapters import adapt_widerface

        # Format: x y w h blur expr illum invalid occl pose
        # invalid is field index 7
        gt_content = (
            "0--Parade/0_Parade_invalid.jpg\n"
            "2\n"
            "60 60 50 50 0 0 0 1 0 0\n"   # invalid=1 → dropped
            "100 100 60 60 0 0 0 0 0 0\n" # valid → kept
        )
        gt_txt = self._write_gt(tmp_path, gt_content)

        records = adapt_widerface(gt_txt, str(tmp_path / "images"))
        assert len(records) == 1
        assert records[0]["boxes"] == [[100, 100, 160, 160]]

    def test_zero_face_count_entry_skipped(self, tmp_path):
        """An image with face count=0 (and dummy 0-line) is skipped gracefully."""
        from aslv2.detect.adapters import adapt_widerface

        gt_content = (
            "0--Parade/0_Parade_no_face.jpg\n"
            "0\n"
            "0 0 0 0 0 0 0 0 0 0\n"        # dummy line
            "0--Parade/0_Parade_one_face.jpg\n"
            "1\n"
            "60 60 50 50 0 0 0 0 0 0\n"
        )
        gt_txt = self._write_gt(tmp_path, gt_content)

        records = adapt_widerface(gt_txt, str(tmp_path / "images"))
        # zero-face image dropped; one-face image kept
        assert len(records) == 1
        assert "one_face" in records[0]["image"]

    def test_nonpositive_wh_filtered(self, tmp_path):
        """Faces with w<=0 or h<=0 are filtered out."""
        from aslv2.detect.adapters import adapt_widerface

        gt_content = (
            "0--Parade/0_Parade_zero_wh.jpg\n"
            "2\n"
            "10 10 0 50 0 0 0 0 0 0\n"     # w=0 → filtered
            "60 60 50 50 0 0 0 0 0 0\n"    # valid
        )
        gt_txt = self._write_gt(tmp_path, gt_content)

        records = adapt_widerface(gt_txt, str(tmp_path / "images"))
        assert len(records) == 1
        assert records[0]["boxes"] == [[60, 60, 110, 110]]


# ---------------------------------------------------------------------------
# write_manifest
# ---------------------------------------------------------------------------

class TestWriteManifest:
    def test_writes_json_list(self, tmp_path):
        """write_manifest dumps a JSON list to the given path."""
        from aslv2.detect.adapters import write_manifest

        records = [
            {"image": "/a/b.jpg", "boxes": [[0, 0, 10, 10]], "labels": [0]},
        ]
        out = tmp_path / "manifest.json"
        write_manifest(records, str(out))

        loaded = json.loads(out.read_text())
        assert loaded == records

    def test_creates_parent_dirs(self, tmp_path):
        """write_manifest creates intermediate directories if needed."""
        from aslv2.detect.adapters import write_manifest

        out = tmp_path / "deep" / "nested" / "manifest.json"
        write_manifest([{"image": "/x.jpg", "boxes": [], "labels": []}], str(out))
        assert out.exists()
