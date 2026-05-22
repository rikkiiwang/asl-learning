import json
import numpy as np
import pytest
from aslv2.audit import load_audit_slice, assert_min_coverage, AuditFrame


def _good(tmp_path):
    data = [
        {"image": "f0.png", "hands": [[10, 10, 30, 40]], "head": [50, 5, 80, 45],
         "keypoints": [[[20, 20]] * 21]},
        {"image": "f1.png", "hands": [[10, 10, 30, 40], [60, 12, 82, 44]],
         "head": None, "keypoints": None},
    ]
    p = tmp_path / "slice.json"
    p.write_text(json.dumps(data))
    return p


def test_loads_frames_with_typed_arrays(tmp_path):
    frames = load_audit_slice(_good(tmp_path))
    assert len(frames) == 2
    assert isinstance(frames[0], AuditFrame)
    assert frames[0].hands.shape == (1, 4)
    np.testing.assert_allclose(frames[0].head, [50, 5, 80, 45])
    assert frames[1].head is None and frames[1].keypoints is None


def test_rejects_wrong_keypoint_count(tmp_path):
    bad = [{"image": "f.png", "hands": [[0, 0, 1, 1]], "head": None,
            "keypoints": [[[0, 0]] * 20]}]              # 20 != 21
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="21"):
        load_audit_slice(p)


def test_min_coverage_gate(tmp_path):
    frames = load_audit_slice(_good(tmp_path))
    with pytest.raises(AssertionError, match="at least 50"):
        assert_min_coverage(frames, min_frames=50)
