"""Labeled ASL-frame audit slice (spec §9 item 5) — REQUIRED ground truth for
detector/landmark metrics. JSON list of frames; loader returns typed arrays and
validates shapes so a malformed slice fails loudly before any metric is trusted."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class AuditFrame:
    image: str
    hands: np.ndarray                 # (H,4) xyxy
    head: np.ndarray | None           # (4,) xyxy or None
    keypoints: np.ndarray | None      # (H,21,2) or None


def load_audit_slice(path) -> list[AuditFrame]:
    raw = json.loads(Path(path).read_text())
    frames: list[AuditFrame] = []
    for i, e in enumerate(raw):
        hands = np.asarray(e["hands"], dtype=float)
        if hands.ndim == 1 and hands.size == 0:
            hands = hands.reshape(0, 4)          # [] stub → (0,4) unlabeled
        if hands.ndim != 2 or hands.shape[1] != 4:
            raise ValueError(f"frame {i}: hands must be (H,4) xyxy")
        head = None if e.get("head") is None else np.asarray(e["head"], float)
        if head is not None and head.shape != (4,):
            raise ValueError(f"frame {i}: head must be (4,) xyxy or null")
        kps = None
        if e.get("keypoints") is not None:
            kps = np.asarray(e["keypoints"], dtype=float)
            if kps.ndim != 3 or kps.shape[1] != 21 or kps.shape[2] != 2:
                raise ValueError(f"frame {i}: keypoints must be (H,21,2)")
            if kps.shape[0] != hands.shape[0]:
                raise ValueError(f"frame {i}: keypoints rows must match hands")
        frames.append(AuditFrame(e["image"], hands, head, kps))
    return frames


def assert_min_coverage(frames: list[AuditFrame], min_frames: int = 50) -> None:
    assert len(frames) >= min_frames, (
        f"audit slice has {len(frames)} frames; need at least {min_frames} "
        "for trustworthy per-stage metrics (spec §9 item 5)")
