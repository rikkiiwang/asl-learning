"""Geometry layer: the appearance-invariant primary signal (spec §3.3, §3.6).

All inputs are pixel-space xyxy boxes / xy keypoints. The head anchor is treated
as load-bearing: it is smoothed per clip and never trusted blind.
"""
import numpy as np

HEAD_FALLBACK_SCALE = 0.5   # fraction of frame height used as body-scale when no head


def resolve_head_anchor(heads: np.ndarray, frame_wh: tuple[int, int]):
    """heads: (F,4) xyxy with np.nan rows for frames where no head was detected.
    Returns (anchor_xyxy (4,), head_present_flag {0.0,1.0}).
    Head barely moves over ~3s, so a per-clip median is both stable and a valid
    no-jitter anchor."""
    valid = heads[~np.isnan(heads).any(axis=1)]
    if len(valid) > 0:
        return np.median(valid, axis=0), 1.0
    w, h = frame_wh
    half = HEAD_FALLBACK_SCALE * h / 2.0
    cx, cy = w / 2.0, h / 2.0
    return np.array([cx - half, cy - half, cx + half, cy + half]), 0.0


def _centroids(boxes: np.ndarray) -> np.ndarray:
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2,
                     (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)


def slot_hands(frames: list[np.ndarray], head_cx: float, gate: float = 80.0):
    """frames: list length F of (k,4) xyxy arrays, k in {0,1,2}.
    Associates boxes across frames into <=2 tracks by nearest-centroid (within
    `gate` px), then assigns each track a fixed slot: 0=left, 1=right, decided by
    the track's clip-median centroid x relative to `head_cx`. Stable to box-area
    swaps because slotting uses x-position, not size.
    Returns (slots (F,2,4) with np.nan for empty, presence (F,2) in {0,1})."""
    F = len(frames)
    tracks: list[dict] = []          # each: {"last": xy, "rows": {f: box}, "xs": [x..]}
    for f, boxes in enumerate(frames):
        if len(boxes) == 0:
            continue
        cents = _centroids(boxes)
        used = set()
        # match existing tracks to nearest unused box within gate
        for tr in tracks:
            d = np.linalg.norm(cents - tr["last"], axis=1)
            cand = [i for i in np.argsort(d) if i not in used and d[i] <= gate]
            if cand:
                i = cand[0]
                used.add(i)
                tr["rows"][f] = boxes[i]
                tr["last"] = cents[i]
                tr["xs"].append(cents[i, 0])
        # unmatched boxes start new tracks (cap 2 total, prefer most-supported)
        for i in range(len(boxes)):
            if i not in used:
                tracks.append({"last": cents[i], "rows": {f: boxes[i]},
                               "xs": [cents[i, 0]]})
    # cap is applied post-hoc: a 3rd object appearing mid-clip participates in
    # matching until the clip ends, then the two best-supported tracks are kept
    if len(tracks) > 2:
        tracks = sorted(tracks, key=lambda t: -len(t["rows"]))[:2]

    # assign slots by clip-median x relative to head
    def med_x(tr):
        return float(np.median(tr["xs"]))
    if len(tracks) == 2:
        order = sorted(tracks, key=med_x)            # smaller x -> slot 0 (left)
        slot_of = {id(order[0]): 0, id(order[1]): 1}
    elif len(tracks) == 1:
        slot_of = {id(tracks[0]): 0 if med_x(tracks[0]) < head_cx else 1}
    else:
        slot_of = {}

    slots = np.full((F, 2, 4), np.nan)
    present = np.zeros((F, 2))
    for tr in tracks:
        s = slot_of[id(tr)]
        for f, box in tr["rows"].items():
            slots[f, s] = box
            present[f, s] = 1.0
    return slots, present


GEOM_DIM = 2 * 21 * 2 + 2 * 2 + 2 + 2 + 1   # 84 + 4 + 2 + 2 + 1 = 93


def normalize_geometry(kps: np.ndarray, present: np.ndarray,
                       head: np.ndarray, head_present: float) -> np.ndarray:
    """kps: (2,21,2) keypoints per slot (np.nan where absent).
    present: (2,) slot presence. head: (4,) xyxy anchor. Returns (GEOM_DIM,).
    Translation/scale invariant: subtract head center, divide by head size."""
    hcx = (head[0] + head[2]) / 2.0
    hcy = (head[1] + head[3]) / 2.0
    hs = max(head[2] - head[0], head[3] - head[1])
    if hs <= 0:
        hs = 1.0
    center = np.array([hcx, hcy])

    kp_block = np.zeros((2, 21, 2))
    hand2head = np.zeros((2, 2))
    hand_centers = np.full((2, 2), np.nan)
    for s in range(2):
        if present[s] > 0 and not np.isnan(kps[s]).any():
            norm = (kps[s] - center) / hs
            kp_block[s] = norm
            hc = kps[s].mean(axis=0)
            hand_centers[s] = hc
            hand2head[s] = (hc - center) / hs
    if not np.any(np.isnan(hand_centers)):
        hand2hand = (hand_centers[1] - hand_centers[0]) / hs
    else:
        hand2hand = np.zeros(2)

    return np.concatenate([
        kp_block.reshape(-1),       # 84
        hand2head.reshape(-1),      # 4
        hand2hand,                  # 2
        present.astype(float),      # 2
        np.array([float(head_present)]),  # 1
    ])
