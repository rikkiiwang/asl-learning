"""Image-shrinking helper for detector training data.

Provides a pure function that downscales an image (and its bounding boxes)
so the longer side equals *max_side* — useful for shrinking high-res frames
to a size still larger than the 128×128 training crop, making uploads much
smaller without discarding any information the model would see.

Never upscales: if the image is already smaller than max_side, it is returned
unchanged (scale = 1.0).
"""
from __future__ import annotations

import cv2
import numpy as np


def resize_keep_aspect(
    img_bgr: np.ndarray,
    boxes_xyxy: np.ndarray,
    max_side: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Downscale *img_bgr* so its longer side == *max_side*.

    Parameters
    ----------
    img_bgr:
        HxWx3 uint8 BGR image (as returned by ``cv2.imread``).
    boxes_xyxy:
        Nx4 float array of bounding boxes in [x1, y1, x2, y2] pixel
        coordinates relative to the *original* image.
    max_side:
        Target maximum dimension in pixels.  The aspect ratio is preserved.
        If the image's longer side is already ≤ *max_side*, the image and
        boxes are returned unchanged and ``scale`` is 1.0.

    Returns
    -------
    img2:
        Resized (or original) image.
    boxes2:
        Boxes scaled by *scale* (copy; original array is not mutated).
    scale:
        The multiplier applied to both dimensions.  Equal to 1.0 when no
        resize was needed.
    """
    h, w = img_bgr.shape[:2]
    longer = max(h, w)
    if longer <= max_side:
        return img_bgr, boxes_xyxy.copy() if len(boxes_xyxy) else boxes_xyxy, 1.0

    scale = max_side / longer
    new_w = round(w * scale)
    new_h = round(h * scale)
    img2 = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    boxes2 = boxes_xyxy * scale
    return img2, boxes2, scale
