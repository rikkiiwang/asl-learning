"""Per-stage quality metrics computed against the labeled audit slice (spec §3.6,
§8). Small-slice honest choices: detection-rate@IoU + mean-IoU (not full mAP),
head center/size jitter, and PCK for keypoints."""
import numpy as np

from .boxes import iou


def detection_rate(preds: np.ndarray, gts: np.ndarray, iou_thr: float = 0.5) -> float:
    """Fraction of GT boxes matched by at least one pred at >= iou_thr (recall)."""
    if len(gts) == 0:
        return 1.0
    matched = 0
    for g in gts:
        if any(iou(g, p) >= iou_thr for p in preds):
            matched += 1
    return matched / len(gts)


def head_stability(heads: np.ndarray) -> float:
    """heads: (F,4) xyxy across a clip. Returns center jitter normalized by median
    head size — 0.0 means perfectly static. Lower is better."""
    cx = (heads[:, 0] + heads[:, 2]) / 2
    cy = (heads[:, 1] + heads[:, 3]) / 2
    size = np.median(np.maximum(heads[:, 2] - heads[:, 0], heads[:, 3] - heads[:, 1]))
    size = max(float(size), 1.0)
    return float((cx.std() + cy.std()) / size)


def pck(pred_kps: np.ndarray, gt_kps: np.ndarray,
        ref_size: float, thr_frac: float = 0.2) -> float:
    """Percentage of Correct Keypoints: fraction within thr_frac*ref_size px."""
    d = np.linalg.norm(pred_kps - gt_kps, axis=1)
    return float((d <= thr_frac * ref_size).mean())
