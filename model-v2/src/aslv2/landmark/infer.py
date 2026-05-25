"""Landmark inference helpers.

landmark_hand(model, crop_rgb, norm) -> (21, 2) ndarray in [0, 1]
    Run the Landmark model on a 64×64 RGB crop (numpy uint8 H×W×3).

map_to_frame(kp01, box_xyxy) -> (K, 2) ndarray of pixel coords
    Map keypoints normalised to [0,1] in crop space back to original-frame pixels
    using the hand bounding box.

    kp01     : (K, 2) float — values in [0, 1]
    box_xyxy : (4,)   float — [x1, y1, x2, y2] in original-frame pixels

    x_frame = x1 + kp_x * (x2 - x1)
    y_frame = y1 + kp_y * (y2 - y1)
"""
import numpy as np
import cv2
import torch

_CROP_SIZE = 64


def landmark_hand(
    model: "torch.nn.Module",
    crop_rgb: np.ndarray,
    norm: dict,
) -> np.ndarray:
    """Run landmark model on a hand crop.

    Args:
        model:    Landmark model in eval mode.
        crop_rgb: uint8 RGB image of arbitrary size — will be resized to 64×64.
        norm:     dict with "mean" and "std", each a list of 3 floats.

    Returns:
        (21, 2) float32 ndarray with keypoints in [0, 1] crop-normalised coords.
    """
    # Resize to 64×64
    resized = cv2.resize(crop_rgb, (_CROP_SIZE, _CROP_SIZE),
                         interpolation=cv2.INTER_LINEAR)

    # Normalise: uint8 → float32 in [0,1], then per-channel standardise
    img = resized.astype(np.float32) / 255.0   # (64,64,3)
    mean = np.array(norm["mean"], dtype=np.float32)
    std  = np.array(norm["std"],  dtype=np.float32)
    img  = (img - mean) / std                  # (64,64,3)

    # To tensor (1, 3, 64, 64), on the model's device
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    tensor = tensor.to(next(model.parameters()).device)

    with torch.no_grad():
        out = model(tensor)   # (1, 21, 2)

    return out.squeeze(0).cpu().numpy()   # (21, 2)


def map_to_frame(
    kp01: np.ndarray,
    box_xyxy: np.ndarray,
) -> np.ndarray:
    """Map normalised crop keypoints to original-frame pixel coordinates.

    Args:
        kp01     : (K, 2) float — keypoints in [0, 1] crop space.
        box_xyxy : (4,)   float — [x1, y1, x2, y2] in frame pixels.

    Returns:
        (K, 2) float32 ndarray — keypoints in frame pixel coordinates.
    """
    kp01 = np.asarray(kp01, dtype=np.float32)
    box  = np.asarray(box_xyxy, dtype=np.float32)
    x1, y1, x2, y2 = box

    frame_kps = np.empty_like(kp01)
    frame_kps[:, 0] = x1 + kp01[:, 0] * (x2 - x1)
    frame_kps[:, 1] = y1 + kp01[:, 1] * (y2 - y1)
    return frame_kps
