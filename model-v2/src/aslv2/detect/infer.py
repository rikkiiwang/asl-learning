"""Per-frame inference: run Detector and return top-2 hands + top-1 head.

Usage:
    result = detect_frame(model, frame_rgb, norm)
    # result["hands"] : np.ndarray (<=2, 4) xyxy in original-frame pixels
    # result["head"]  : np.ndarray (4,) xyxy in original-frame pixels, or None
"""
import cv2
import numpy as np
import torch

from aslv2.detect.anchors import make_anchors
from aslv2.detect.encode import decode

_IMG_SIZE = 128


def detect_frame(
    model: torch.nn.Module,
    frame_rgb: np.ndarray,
    norm: dict,
    score_thr: float = 0.3,
    iou_thr: float = 0.45,
) -> dict:
    """Run the detector on a single RGB frame.

    Args:
        model:     Detector (or compatible) instance — must be in eval mode.
        frame_rgb: HxWx3 uint8 numpy array in RGB order.
        norm:      dict with "mean" and "std" lists of 3 floats (per-channel).
        score_thr: minimum sigmoid confidence to keep a detection.
        iou_thr:   NMS IoU threshold.

    Returns:
        {
          "hands": np.ndarray of shape (<=2, 4) — top-2 hands by score,
                   xyxy in *original-frame* pixel coordinates,
          "head":  np.ndarray of shape (4,) or None — top-1 head,
                   xyxy in *original-frame* pixel coordinates,
        }
    """
    orig_h, orig_w = frame_rgb.shape[:2]

    # ----- preprocess -----
    img = cv2.resize(frame_rgb, (_IMG_SIZE, _IMG_SIZE),
                     interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0        # (128, 128, 3)
    mean = np.array(norm["mean"], dtype=np.float32)
    std  = np.array(norm["std"],  dtype=np.float32)
    img  = (img - mean) / std                   # normalise

    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)  # (1, 3, H, W)

    # ----- forward pass -----
    with torch.no_grad():
        cls_logits, box_pred = model(tensor)    # (1, 192, 2), (1, 192, 4)

    cls_np  = cls_logits[0].cpu().numpy()       # (192, 2)
    box_np  = box_pred[0].cpu().numpy()         # (192, 4)

    # ----- decode with NMS (in 128² space) -----
    anchors = make_anchors(img=_IMG_SIZE, stride=16, scales=(32, 64, 96))
    # Clamp box deltas to prevent exp() overflow during decode
    box_np = np.clip(box_np, -10.0, 10.0)
    boxes, scores, labels = decode(anchors, box_np, cls_np,
                                   score_thr=score_thr, iou_thr=iou_thr)

    # Filter out any degenerate boxes (NaN / inf / negative size)
    if len(boxes) > 0:
        valid = (np.isfinite(boxes).all(axis=1) &
                 (boxes[:, 2] > boxes[:, 0]) &
                 (boxes[:, 3] > boxes[:, 1]))
        boxes  = boxes[valid]
        scores = scores[valid]
        labels = labels[valid]

    # ----- scale to original-frame pixel space and clip to image bounds -----
    sx = orig_w / _IMG_SIZE
    sy = orig_h / _IMG_SIZE
    if len(boxes) > 0:
        boxes = boxes * np.array([sx, sy, sx, sy], dtype=np.float32)
        # Clip to valid image region
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, orig_h)
        # Re-filter after clipping: remove degenerate (zero-area) boxes
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes  = boxes[valid]
        scores = scores[valid]
        labels = labels[valid]

    # ----- separate by class, cap hands at top-2, head at top-1 -----
    hand_mask = labels == 0
    head_mask = labels == 1

    # Hands: top-2 by score
    hand_boxes  = boxes[hand_mask]
    hand_scores = scores[hand_mask]
    if len(hand_boxes) > 0:
        order = np.argsort(hand_scores)[::-1][:2]
        hand_boxes = hand_boxes[order]
    else:
        hand_boxes = np.empty((0, 4), dtype=np.float32)

    # Head: top-1 by score
    head_result = None
    head_boxes  = boxes[head_mask]
    head_scores = scores[head_mask]
    if len(head_boxes) > 0:
        top = int(np.argmax(head_scores))
        head_result = head_boxes[top]             # (4,)

    return {"hands": hand_boxes, "head": head_result}
