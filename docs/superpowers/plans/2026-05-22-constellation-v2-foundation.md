# Constellation v2 — Plan 1: Foundation & Export Spike

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the `model-v2/` (`aslv2`) package with test-first geometry utilities (the appearance-invariant primary signal's backbone) and resolve the project's hardest risk first — whether the three-stage inference graph can export and round-trip — before any real model is trained.

**Architecture:** Plan 1 of 5 for the Constellation spec (`docs/superpowers/specs/2026-05-22-constellation-v2-model-design.md`). It builds the pure-Python geometry layer (head smoothing/fallback §3.6, stable hand slotting §3.3, head-normalized feature vector) and an export spike (§6) using *dummy* tiny models with the real I/O shapes. Later plans add the real detector, landmark model, recognizer, and validation. No real training happens here.

**Tech Stack:** Python, PyTorch 2.7 (MPS), NumPy, torchvision.ops (roi_align/nms), onnx 1.21, onnxruntime 1.26, onnxruntime-web (node), pytest 8.4.

---

## Conventions (used by every task)

- **Box** = `np.ndarray` shape `(4,)` or `(N,4)`, format **xyxy**, float pixels.
- **Keypoints** = `np.ndarray` shape `(21,2)`, xy pixels in the original frame.
- **Per-clip frame count** `F = 16` (frozen, matches app capture).
- **Hand slots:** index `0 = left` (smaller x relative to head), `1 = right`.
- Missing data is represented by `np.nan`-filled arrays plus an explicit presence flag — never silently zeroed before the flag is computed.
- All angles/coords normalized by **head size** `hs = max(head_w, head_h)`.

## File Structure

```
model-v2/
  pyproject.toml                # package `aslv2` (src layout) + pytest config
  requirements.txt              # mirror model/requirements.txt + onnxruntime-web (dev)
  src/aslv2/__init__.py         # __version__
  src/aslv2/boxes.py            # iou, nms (numpy)
  src/aslv2/geometry.py         # smooth/fallback head anchor, slot hands, normalize
  src/aslv2/audit.py            # labeled ASL-frame audit-slice schema + loader/validator
  src/aslv2/metrics.py          # detection rate, head stability, keypoint PCK (vs audit slice)
  src/aslv2/size_budget.py      # per-stage param/quantized-size caps (spec §3.7) + asserts
  src/aslv2/export_spike/__init__.py
  src/aslv2/export_spike/dummy_models.py   # tiny detector/landmark/recognizer, real shapes
  src/aslv2/export_spike/combined.py       # 3-stage torch module (detect→crop→landmark→recognize)
  src/aslv2/export_spike/run_spike.py      # export + round-trip + write decision
  tests/test_version.py
  tests/test_boxes.py
  tests/test_geometry_head.py
  tests/test_geometry_slots.py
  tests/test_geometry_normalize.py
  tests/test_dummy_models.py
  tests/test_export_spike.py
  tests/test_audit.py
  tests/test_metrics.py
  tests/test_size_budget.py
  scripts/ort_web_probe.mjs     # node: load exported onnx in onnxruntime-web (WASM)
  artifacts/export_decision.md  # written by the spike (decision (a) vs (b))
```

---

## Task 1: Scaffold the `aslv2` package + pytest

**Files:**
- Create: `model-v2/pyproject.toml`
- Create: `model-v2/requirements.txt`
- Create: `model-v2/src/aslv2/__init__.py`
- Test: `model-v2/tests/test_version.py`

- [ ] **Step 1: Write the failing test**

`model-v2/tests/test_version.py`:
```python
def test_package_imports_and_has_version():
    import aslv2
    assert isinstance(aslv2.__version__, str)
    assert aslv2.__version__ != ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "model-v2" && python -m pytest tests/test_version.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aslv2'`

- [ ] **Step 3: Create the package and config**

`model-v2/src/aslv2/__init__.py`:
```python
__version__ = "0.0.1"
```

`model-v2/pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "aslv2"
version = "0.0.1"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.2", "numpy", "opencv-python", "pyyaml",
    "onnx", "onnxruntime", "torchvision", "tqdm",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

`model-v2/requirements.txt`:
```
torch>=2.2
numpy
opencv-python
pyyaml
onnx
onnxruntime
torchvision
tqdm
```

- [ ] **Step 4: Install editable + run the test to verify it passes**

Run: `cd "model-v2" && pip install -e . && python -m pytest tests/test_version.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add model-v2/pyproject.toml model-v2/requirements.txt model-v2/src/aslv2/__init__.py model-v2/tests/test_version.py
git commit -m "feat(aslv2): scaffold model-v2 package with pytest"
```

---

## Task 2: Box utilities — IoU + NMS

**Files:**
- Create: `model-v2/src/aslv2/boxes.py`
- Test: `model-v2/tests/test_boxes.py`

- [ ] **Step 1: Write the failing test**

`model-v2/tests/test_boxes.py`:
```python
import numpy as np
from aslv2.boxes import iou, nms


def test_iou_half_overlap():
    a = np.array([0.0, 0.0, 2.0, 2.0])
    b = np.array([1.0, 0.0, 3.0, 2.0])   # overlap area 2, union 6
    assert abs(iou(a, b) - (2.0 / 6.0)) < 1e-6


def test_iou_disjoint_is_zero():
    a = np.array([0.0, 0.0, 1.0, 1.0])
    b = np.array([5.0, 5.0, 6.0, 6.0])
    assert iou(a, b) == 0.0


def test_nms_suppresses_overlapping_keeps_best():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]], float)
    scores = np.array([0.9, 0.8, 0.7])
    keep = nms(boxes, scores, iou_thr=0.5)
    assert keep == [0, 2]            # box 1 suppressed by box 0; far box kept
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "model-v2" && python -m pytest tests/test_boxes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aslv2.boxes'`

- [ ] **Step 3: Implement `boxes.py`**

`model-v2/src/aslv2/boxes.py`:
```python
"""Numpy box helpers (xyxy). Used for offline detector post-processing and tests."""
import numpy as np


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area_a + area_b - inter))


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5) -> list[int]:
    order = list(np.argsort(-scores))
    keep: list[int] = []
    while order:
        i = order.pop(0)
        keep.append(int(i))
        order = [j for j in order if iou(boxes[i], boxes[j]) <= iou_thr]
    return keep
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "model-v2" && python -m pytest tests/test_boxes.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add model-v2/src/aslv2/boxes.py model-v2/tests/test_boxes.py
git commit -m "feat(aslv2): iou + nms box utilities"
```

---

## Task 3: Head anchor — per-clip smoothing + no-head fallback (spec §3.6)

**Files:**
- Create: `model-v2/src/aslv2/geometry.py`
- Test: `model-v2/tests/test_geometry_head.py`

- [ ] **Step 1: Write the failing test**

`model-v2/tests/test_geometry_head.py`:
```python
import numpy as np
from aslv2.geometry import resolve_head_anchor


def test_median_anchor_ignores_jitter_and_reports_present():
    heads = np.array([[10, 10, 30, 40],
                      [11, 9, 31, 41],
                      [9, 11, 29, 39]], float)            # 3 valid frames
    anchor, present = resolve_head_anchor(heads, frame_wh=(640, 480))
    assert present == 1.0
    np.testing.assert_allclose(anchor, [10, 10, 30, 40])  # per-coord median


def test_missing_frames_fall_back_to_median_of_valid():
    heads = np.array([[10, 10, 30, 40],
                      [np.nan] * 4,
                      [12, 12, 32, 42]], float)
    anchor, present = resolve_head_anchor(heads, frame_wh=(640, 480))
    assert present == 1.0
    np.testing.assert_allclose(anchor, [11, 11, 31, 41])  # median of the 2 valid


def test_all_missing_falls_back_to_centered_body_scale_anchor():
    heads = np.full((16, 4), np.nan)
    anchor, present = resolve_head_anchor(heads, frame_wh=(640, 480))
    assert present == 0.0
    cx, cy = (anchor[0] + anchor[2]) / 2, (anchor[1] + anchor[3]) / 2
    assert abs(cx - 320) < 1e-6 and abs(cy - 240) < 1e-6   # frame center
    assert (anchor[3] - anchor[1]) > 0                     # has a body-scale size
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "model-v2" && python -m pytest tests/test_geometry_head.py -v`
Expected: FAIL — `ImportError: cannot import name 'resolve_head_anchor'`

- [ ] **Step 3: Implement `resolve_head_anchor` in `geometry.py`**

`model-v2/src/aslv2/geometry.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "model-v2" && python -m pytest tests/test_geometry_head.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add model-v2/src/aslv2/geometry.py model-v2/tests/test_geometry_head.py
git commit -m "feat(aslv2): per-clip head-anchor smoothing + no-head fallback (spec 3.6)"
```

---

## Task 4: Stable hand slotting (spec §3.3 — replaces box-area ordering, fixes P2a)

**Files:**
- Modify: `model-v2/src/aslv2/geometry.py`
- Test: `model-v2/tests/test_geometry_slots.py`

- [ ] **Step 1: Write the failing test**

`model-v2/tests/test_geometry_slots.py`:
```python
import numpy as np
from aslv2.geometry import slot_hands


def _box(cx, cy, s):
    return [cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2]


def test_slots_are_stable_when_box_areas_swap():
    head_cx = 100.0
    # left hand stays at x=60, right at x=140; their SIZES swap across frames.
    frames = [
        np.array([_box(60, 50, 20), _box(140, 50, 40)]),   # right bigger
        np.array([_box(60, 50, 40), _box(140, 50, 20)]),   # left bigger (area swap)
    ]
    slots, present = slot_hands(frames, head_cx)
    # slot 0 (left) center.x must stay ~60 in BOTH frames despite the area swap
    for f in range(2):
        lcx = (slots[f, 0, 0] + slots[f, 0, 2]) / 2
        assert abs(lcx - 60) < 1e-6
    assert present.tolist() == [[1, 1], [1, 1]]


def test_single_hand_goes_to_side_relative_to_head():
    head_cx = 100.0
    frames = [np.array([_box(40, 50, 20)])]            # one hand, left of head
    slots, present = slot_hands(frames, head_cx)
    assert present.tolist() == [[1, 0]]                # left slot filled
    assert np.isnan(slots[0, 1]).all()                 # right slot empty

    frames = [np.array([_box(160, 50, 20)])]           # one hand, right of head
    slots, present = slot_hands(frames, head_cx)
    assert present.tolist() == [[0, 1]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "model-v2" && python -m pytest tests/test_geometry_slots.py -v`
Expected: FAIL — `ImportError: cannot import name 'slot_hands'`

- [ ] **Step 3: Add `slot_hands` to `geometry.py`**

Append to `model-v2/src/aslv2/geometry.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "model-v2" && python -m pytest tests/test_geometry_slots.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add model-v2/src/aslv2/geometry.py model-v2/tests/test_geometry_slots.py
git commit -m "feat(aslv2): stable left/right hand slotting via track association (fixes area-swap)"
```

---

## Task 5: Head-normalized geometry feature vector (spec §3.3)

**Files:**
- Modify: `model-v2/src/aslv2/geometry.py`
- Test: `model-v2/tests/test_geometry_normalize.py`

The feature layout per frame (fixed length **93**): per slot 21×2 normalized keypoints (84), per slot hand→head offset (4), hand→hand offset (2), per-slot presence (2), head_present (1).

- [ ] **Step 1: Write the failing test**

`model-v2/tests/test_geometry_normalize.py`:
```python
import numpy as np
from aslv2.geometry import normalize_geometry, GEOM_DIM


def _inputs():
    kps = np.zeros((2, 21, 2))
    kps[0] = np.tile([60.0, 50.0], (21, 1))     # left-hand keypoints clustered
    kps[1] = np.tile([140.0, 50.0], (21, 1))    # right-hand keypoints
    present = np.array([1.0, 1.0])
    head = np.array([90.0, 30.0, 110.0, 50.0])  # center (100,40), size 20
    return kps, present, head


def test_output_has_fixed_dim():
    kps, present, head = _inputs()
    v = normalize_geometry(kps, present, head, head_present=1.0)
    assert v.shape == (GEOM_DIM,)
    assert GEOM_DIM == 93


def test_translation_invariance():
    kps, present, head = _inputs()
    v1 = normalize_geometry(kps, present, head, 1.0)
    d = np.array([37.0, -12.0])
    v2 = normalize_geometry(kps + d, present, head + np.array([*d, *d]), 1.0)
    np.testing.assert_allclose(v1, v2, atol=1e-6)


def test_scale_invariance():
    kps, present, head = _inputs()
    v1 = normalize_geometry(kps, present, head, 1.0)
    v2 = normalize_geometry(kps * 3.0, present, head * 3.0, 1.0)
    np.testing.assert_allclose(v1, v2, atol=1e-6)


def test_missing_slot_is_zeroed_and_flagged():
    kps, present, head = _inputs()
    kps[1] = np.nan
    present[1] = 0.0
    v = normalize_geometry(kps, present, head, 1.0)
    # right-hand keypoint block (slot 1: indices 42..84) must be all zeros
    assert np.all(v[42:84] == 0.0)
    assert not np.isnan(v).any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "model-v2" && python -m pytest tests/test_geometry_normalize.py -v`
Expected: FAIL — `ImportError: cannot import name 'normalize_geometry'`

- [ ] **Step 3: Add `normalize_geometry` + `GEOM_DIM` to `geometry.py`**

Append to `model-v2/src/aslv2/geometry.py`:
```python
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
    if present[0] > 0 and present[1] > 0:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "model-v2" && python -m pytest tests/test_geometry_normalize.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add model-v2/src/aslv2/geometry.py model-v2/tests/test_geometry_normalize.py
git commit -m "feat(aslv2): head-normalized geometry feature vector (translation/scale invariant)"
```

---

## Task 6: Dummy models with the real I/O shapes (for the export spike)

**Files:**
- Create: `model-v2/src/aslv2/export_spike/__init__.py`
- Create: `model-v2/src/aslv2/export_spike/dummy_models.py`
- Test: `model-v2/tests/test_dummy_models.py`

These carry **no learned value** — only the correct tensor shapes so the export spike exercises the real op chain. Detector emits a fixed anchor grid (selection by `topk`, static shapes); landmark regresses 21×2; recognizer maps fused features over 16 frames to 75 logits.

- [ ] **Step 1: Write the failing test**

`model-v2/tests/test_dummy_models.py`:
```python
import torch
from aslv2.export_spike.dummy_models import DummyDetector, DummyLandmark, DummyRecognizer


def test_detector_emits_anchor_grid():
    det = DummyDetector()
    boxes, scores = det(torch.randn(16, 3, 128, 128))   # F frames
    assert boxes.shape[0] == 16 and boxes.shape[2] == 4  # (F, A, 4)
    assert scores.shape[:2] == boxes.shape[:2]
    assert scores.shape[2] == 2                          # hand, head classes


def test_landmark_regresses_21_keypoints():
    lm = DummyLandmark()
    out = lm(torch.randn(8, 3, 64, 64))                  # 8 crops
    assert out.shape == (8, 21, 2)


def test_recognizer_outputs_75_logits():
    rec = DummyRecognizer(geom_dim=93, n_classes=75)
    # per-frame fused vector = geom(93) + 2 hand appearance embeds(2*32)
    logits = rec(torch.randn(1, 16, 93 + 64))
    assert logits.shape == (1, 75)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "model-v2" && python -m pytest tests/test_dummy_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aslv2.export_spike'`

- [ ] **Step 3: Implement the dummy models**

`model-v2/src/aslv2/export_spike/__init__.py`:
```python
```

`model-v2/src/aslv2/export_spike/dummy_models.py`:
```python
"""Shape-only stand-ins so the export spike exercises the real op chain.
No learned weights matter here. Real models arrive in Plans 2-4."""
import torch
import torch.nn as nn

A = 256  # anchors per frame (fixed grid -> static shapes for export)


class DummyDetector(nn.Module):
    """(F,3,128,128) -> boxes (F,A,4) xyxy, scores (F,A,2) [hand, head]."""
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(3, 8, 3, 2, 1), nn.ReLU(),
                                  nn.AdaptiveAvgPool2d(1))
        self.box = nn.Linear(8, A * 4)
        self.cls = nn.Linear(8, A * 2)

    def forward(self, x):
        f = self.body(x).flatten(1)             # (F,8)
        boxes = self.box(f).reshape(-1, A, 4).sigmoid() * 128.0
        scores = self.cls(f).reshape(-1, A, 2)
        return boxes, scores


class DummyLandmark(nn.Module):
    """(N,3,64,64) -> (N,21,2) keypoints in crop space."""
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(3, 8, 3, 2, 1), nn.ReLU(),
                                  nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(8, 21 * 2)

    def forward(self, x):
        return self.fc(self.body(x).flatten(1)).reshape(-1, 21, 2)


class DummyRecognizer(nn.Module):
    """(B,F,fused) -> (B,n_classes). Mean-pool temporal head (shape-only)."""
    def __init__(self, geom_dim=93, n_classes=75):
        super().__init__()
        self.fc = nn.Linear(geom_dim + 64, n_classes)

    def forward(self, x):
        return self.fc(x.mean(dim=1))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "model-v2" && python -m pytest tests/test_dummy_models.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add model-v2/src/aslv2/export_spike/__init__.py model-v2/src/aslv2/export_spike/dummy_models.py model-v2/tests/test_dummy_models.py
git commit -m "feat(aslv2): dummy detector/landmark/recognizer with real I/O shapes"
```

---

## Task 7: Combined 3-stage torch module (detect → select → crop → landmark → recognize)

**Files:**
- Create: `model-v2/src/aslv2/export_spike/combined.py`
- Test: extend `model-v2/tests/test_export_spike.py` (created here)

This is the graph whose exportability is the project's hardest risk. It uses `torch.topk` for static selection (2 hands + 1 head per frame) and `torchvision.ops.roi_align` for cropping — the ops most likely to limit ONNX/ORT-Web export.

- [ ] **Step 1: Write the failing test (torch forward shape)**

`model-v2/tests/test_export_spike.py`:
```python
import torch
from aslv2.export_spike.combined import CombinedConstellation


def test_combined_forward_outputs_logits():
    model = CombinedConstellation().eval()
    frames = torch.randn(16, 3, 128, 128)
    with torch.no_grad():
        logits = model(frames)
    assert logits.shape == (1, 75)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "model-v2" && python -m pytest tests/test_export_spike.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aslv2.export_spike.combined'`

- [ ] **Step 3: Implement the combined module**

`model-v2/src/aslv2/export_spike/combined.py`:
```python
"""End-to-end inference graph used ONLY to test exportability (spec §6).
16 raw frames -> detect -> topk select (2 hands + 1 head) -> roi_align crops ->
landmark -> tensorized geometry -> recognizer -> 75 logits.
Geometry here is a differentiable tensor approximation of aslv2.geometry; the
offline numpy version remains the source of truth for training-time caching."""
import torch
import torch.nn as nn
from torchvision.ops import roi_align

from .dummy_models import DummyDetector, DummyLandmark, DummyRecognizer

F = 16


class CombinedConstellation(nn.Module):
    def __init__(self):
        super().__init__()
        self.det = DummyDetector()
        self.lm = DummyLandmark()
        self.rec = DummyRecognizer()
        self.app = nn.Sequential(nn.Conv2d(3, 8, 3, 2, 1), nn.ReLU(),
                                 nn.AdaptiveAvgPool2d(1))   # tiny appearance -> 8
        self.app_proj = nn.Linear(8, 32)

    def _topk_box(self, boxes, scores, k):
        # boxes (F,A,4), scores (F,A) -> (F,k,4)
        idx = scores.topk(k, dim=1).indices                # (F,k)
        return torch.gather(boxes, 1, idx.unsqueeze(-1).expand(-1, -1, 4))

    def forward(self, frames):                              # (F,3,128,128)
        boxes, scores = self.det(frames)
        hands = self._topk_box(boxes, scores[..., 0], 2)    # (F,2,4)
        head = self._topk_box(boxes, scores[..., 1], 1)     # (F,1,4)

        # roi_align expects List[Tensor] of boxes per image, or (K,5) with batch idx
        batch_idx = torch.arange(F).repeat_interleave(2).float().unsqueeze(1)
        rois = torch.cat([batch_idx, hands.reshape(F * 2, 4)], dim=1)   # (F*2,5)
        crops = roi_align(frames, rois, output_size=(64, 64))          # (F*2,3,64,64)

        kps = self.lm(crops)                                # (F*2,21,2)
        app = self.app_proj(self.app(crops).flatten(1))     # (F*2,32)

        # tensor geometry: normalize keypoints by head size/center per frame
        hc = head.reshape(F, 4)
        hcx = (hc[:, 0] + hc[:, 2]) / 2
        hcy = (hc[:, 1] + hc[:, 3]) / 2
        hs = torch.clamp(torch.maximum(hc[:, 2] - hc[:, 0], hc[:, 3] - hc[:, 1]), min=1.0)
        center = torch.stack([hcx, hcy], dim=1).unsqueeze(1)            # (F,1,2)
        kps_f = kps.reshape(F, 2, 21, 2)
        norm_kps = (kps_f - center.unsqueeze(1)) / hs.reshape(F, 1, 1, 1)
        geom = norm_kps.reshape(F, 2 * 21 * 2)                          # (F,84)
        # pad geom to GEOM_DIM=93 (offsets/flags are zeros in the shape-only spike)
        geom = torch.cat([geom, torch.zeros(F, 9)], dim=1)             # (F,93)

        app_f = app.reshape(F, 2, 32).reshape(F, 64)
        fused = torch.cat([geom, app_f], dim=1).unsqueeze(0)            # (1,F,157)
        return self.rec(fused)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "model-v2" && python -m pytest tests/test_export_spike.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add model-v2/src/aslv2/export_spike/combined.py model-v2/tests/test_export_spike.py
git commit -m "feat(aslv2): combined 3-stage inference module for export spike"
```

---

## Task 8: Export round-trip in Python ONNX Runtime (combined-graph (b) gate)

**Files:**
- Create: `model-v2/src/aslv2/export_spike/run_spike.py`
- Test: extend `model-v2/tests/test_export_spike.py`

- [ ] **Step 1: Write the failing test**

Append to `model-v2/tests/test_export_spike.py`:
```python
def test_combined_graph_exports_and_roundtrips(tmp_path):
    import numpy as np, onnxruntime as ort, torch
    from aslv2.export_spike.run_spike import export_combined

    onnx_path = export_combined(tmp_path / "combined.onnx")   # may raise -> (b) infeasible
    frames = torch.randn(16, 3, 128, 128)
    model = CombinedConstellation().eval()
    with torch.no_grad():
        ref = model(frames).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: frames.numpy()})[0]
    np.testing.assert_allclose(ref, got, atol=1e-3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "model-v2" && python -m pytest tests/test_export_spike.py::test_combined_graph_exports_and_roundtrips -v`
Expected: FAIL — `ImportError: cannot import name 'export_combined'`

- [ ] **Step 3: Implement `export_combined` (loads weights into the SAME instance is not needed — caller builds ref; here we export a fresh fixed-seed model and the test builds its own ref from the same seed)**

Adjust: to make ref match, `export_combined` must export a **seeded** model and the test must use the same seed. Update the test's model construction and the exporter to share a seed.

Replace the test body's model construction with a seeded factory and implement the exporter accordingly.

Append to `model-v2/tests/test_export_spike.py` a seeded helper and update the round-trip test:
```python
def _seeded_model():
    import torch
    torch.manual_seed(0)
    return CombinedConstellation().eval()
```
Change `model = CombinedConstellation().eval()` in `test_combined_graph_exports_and_roundtrips` to `model = _seeded_model()`.

`model-v2/src/aslv2/export_spike/run_spike.py`:
```python
"""Prove-first export spike (spec §6). Exports the combined graph and records the
(a)-vs-(b) decision. Run as a script to write artifacts/export_decision.md."""
from pathlib import Path

import torch

from .combined import CombinedConstellation

OPSET = 17


def _seeded_model():
    torch.manual_seed(0)
    return CombinedConstellation().eval()


def export_combined(out_path) -> Path:
    out_path = Path(out_path)
    model = _seeded_model()
    dummy = torch.randn(16, 3, 128, 128)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["frames"], output_names=["logits"],
        opset_version=OPSET, do_constant_folding=True,
    )
    return out_path


def main():
    art = Path(__file__).resolve().parents[3] / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    decision = art / "export_decision.md"
    try:
        export_combined(art / "combined_spike.onnx")
        import numpy as np, onnxruntime as ort
        m = _seeded_model()
        x = torch.randn(16, 3, 128, 128)
        with torch.no_grad():
            ref = m(x).numpy()
        sess = ort.InferenceSession(str(art / "combined_spike.onnx"),
                                    providers=["CPUExecutionProvider"])
        got = sess.run(None, {"frames": x.numpy()})[0]
        ok = np.allclose(ref, got, atol=1e-3)
        decision.write_text(
            f"# Export decision\n\nCombined-graph (b) Python-ORT round-trip: "
            f"{'PASS' if ok else 'MISMATCH'}.\n\n"
            "Next: run scripts/ort_web_probe.mjs to confirm ONNX Runtime Web "
            "(WASM) supports the ops. If web probe passes -> choose (b). "
            "If export raised or web probe fails -> choose (a) separate files.\n")
    except Exception as e:                       # noqa: BLE001 - we WANT to record failure
        decision.write_text(
            f"# Export decision\n\nCombined-graph (b) export FAILED: {e!r}.\n\n"
            "Decision: choose (a) separate ONNX files; coordinate §B with the app "
            "agent before training (spec §6).\n")
        print("Combined export failed -> decision (a). See", decision)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the round-trip test**

Run: `cd "model-v2" && python -m pytest tests/test_export_spike.py::test_combined_graph_exports_and_roundtrips -v`
Expected: **PASS** if roi_align + topk export cleanly at opset 17 (likely). If it **FAILS to export**, that is a *valid spike outcome* — record it: mark this test `xfail` with the exception reason, proceed to Task 9 which writes decision (a), and note the separate-files path for Plan 5.

- [ ] **Step 5: Commit**

```bash
git add model-v2/src/aslv2/export_spike/run_spike.py model-v2/tests/test_export_spike.py
git commit -m "feat(aslv2): combined-graph ONNX export + Python ORT round-trip gate"
```

---

## Task 9: ONNX Runtime **Web** op-support probe + write the decision

**Files:**
- Create: `model-v2/scripts/ort_web_probe.mjs`
- Create (generated): `model-v2/artifacts/export_decision.md`

Python ORT passing is necessary but not sufficient — the deployment target is ONNX Runtime **Web (WASM)**, which has a narrower op set. This probe loads the exported graph in `onnxruntime-web` under node and runs one inference. **This is the real (b)-vs-(a) gate.**

- [ ] **Step 1: Generate the combined graph + Python decision**

Run: `cd "model-v2" && python -m aslv2.export_spike.run_spike`
Expected: `artifacts/combined_spike.onnx` exists (or `export_decision.md` already records a (b) export failure → skip to Step 4 with decision (a)).

- [ ] **Step 2: Write the node probe**

`model-v2/scripts/ort_web_probe.mjs`:
```javascript
// Loads the exported combined graph in onnxruntime-web (WASM) and runs once.
// Usage: node scripts/ort_web_probe.mjs artifacts/combined_spike.onnx
import * as ort from "onnxruntime-web";
import { readFileSync } from "node:fs";

const path = process.argv[2] ?? "artifacts/combined_spike.onnx";
const bytes = readFileSync(path);
try {
  const sess = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  const frames = new ort.Tensor("float32", new Float32Array(16 * 3 * 128 * 128), [16, 3, 128, 128]);
  const out = await sess.run({ frames });
  const logits = out[Object.keys(out)[0]];
  console.log("ORT-Web OK; logits dims =", logits.dims);
  process.exit(logits.dims.join(",") === "1,75" ? 0 : 2);
} catch (e) {
  console.error("ORT-Web FAILED (op unsupported in WASM?):", String(e));
  process.exit(1);
}
```

- [ ] **Step 3: Run the web probe**

Run:
```bash
cd "model-v2" && npm init -y >/dev/null 2>&1 && npm i onnxruntime-web >/dev/null 2>&1 \
  && node scripts/ort_web_probe.mjs artifacts/combined_spike.onnx; echo "exit=$?"
```
Expected: `ORT-Web OK; logits dims = [1,75]` and `exit=0` → combined graph (b) is viable.
If `exit=1` (op unsupported) → decision (a).

- [ ] **Step 4: Finalize `artifacts/export_decision.md`**

Edit `model-v2/artifacts/export_decision.md` to state the final decision explicitly, e.g.:
```markdown
# Export decision (spec §6) — 2026-05-22

- Combined-graph (b) Python ORT round-trip: PASS
- Combined-graph (b) ONNX Runtime Web (WASM) probe: PASS (logits [1,75])

DECISION: (b) combined graph. App contract unchanged (frames in, logits out).
Plan 5 exports the real 3 models into one graph; no §B change needed.
```
(If either probe failed, state DECISION: (a) separate files, and add a note to coordinate the §B contract change with the app agent before Plan 4 training.)

- [ ] **Step 5: Commit**

```bash
git add model-v2/scripts/ort_web_probe.mjs model-v2/artifacts/export_decision.md model-v2/package.json
echo "node_modules/" >> model-v2/.gitignore && git add model-v2/.gitignore
git commit -m "feat(aslv2): ORT-Web export probe + recorded (a)/(b) decision (spec 6 prove-first)"
```

---

## Task 10: Labeled ASL-frame audit-slice schema + loader (spec §9 item 5)

**Files:**
- Create: `model-v2/src/aslv2/audit.py`
- Test: `model-v2/tests/test_audit.py`

The audit slice is the **required** ground truth for every "on real ASL frames" metric. Format is a JSON list; each entry labels one frame with hand boxes, an optional head box, and (optionally) 21 keypoints per hand. This task builds the schema + loader/validator so Plans 2–4 can populate and gate on it.

- [ ] **Step 1: Write the failing test**

`model-v2/tests/test_audit.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "model-v2" && python -m pytest tests/test_audit.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aslv2.audit'`

- [ ] **Step 3: Implement `audit.py`**

`model-v2/src/aslv2/audit.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "model-v2" && python -m pytest tests/test_audit.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add model-v2/src/aslv2/audit.py model-v2/tests/test_audit.py
git commit -m "feat(aslv2): labeled ASL-frame audit-slice schema + loader (required ground truth)"
```

---

## Task 11: Per-stage metrics — detection rate, head stability, keypoint PCK

**Files:**
- Create: `model-v2/src/aslv2/metrics.py`
- Test: `model-v2/tests/test_metrics.py`

Concrete metrics each component plan computes against the audit slice (Task 10). For the small slice we report **detection-rate@IoU** (recall) + mean-IoU rather than full mAP, **head stability** (per-clip center jitter / size), and **PCK** (fraction of keypoints within a fraction of a reference size).

- [ ] **Step 1: Write the failing test**

`model-v2/tests/test_metrics.py`:
```python
import numpy as np
from aslv2.metrics import detection_rate, head_stability, pck


def test_detection_rate_matches_by_iou():
    gt = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], float)
    pred = np.array([[1, 1, 11, 11]], float)            # matches box 0 only
    assert detection_rate(pred, gt, iou_thr=0.5) == 0.5


def test_head_stability_zero_for_static_head():
    heads = np.tile([10, 10, 30, 30], (16, 1)).astype(float)
    assert head_stability(heads) == 0.0                 # no jitter


def test_head_stability_positive_for_jitter():
    heads = np.tile([10, 10, 30, 30], (16, 1)).astype(float)
    heads[::2, 0] += 5                                   # wobble x
    assert head_stability(heads) > 0.0


def test_pck_counts_keypoints_within_threshold():
    gt = np.zeros((21, 2))
    pred = gt.copy()
    pred[0] = [100, 0]                                   # one bad keypoint
    # ref_size 50, thr 0.2 -> tolerance 10px; 20/21 within
    assert abs(pck(pred, gt, ref_size=50.0, thr_frac=0.2) - 20 / 21) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "model-v2" && python -m pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aslv2.metrics'`

- [ ] **Step 3: Implement `metrics.py`**

`model-v2/src/aslv2/metrics.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "model-v2" && python -m pytest tests/test_metrics.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add model-v2/src/aslv2/metrics.py model-v2/tests/test_metrics.py
git commit -m "feat(aslv2): detection-rate/head-stability/PCK metrics vs audit slice"
```

---

## Task 12: Per-stage size-budget caps + enforcement (spec §3.7)

**Files:**
- Create: `model-v2/src/aslv2/size_budget.py`
- Test: `model-v2/tests/test_size_budget.py`

Encodes the §3.7 per-component caps so each later plan can assert its model fits **before** export. Param caps assume ≈1 byte/param at int8.

- [ ] **Step 1: Write the failing test**

`model-v2/tests/test_size_budget.py`:
```python
import pytest
import torch.nn as nn
from aslv2.size_budget import BUDGETS, count_params, assert_within_budget, total_hard_cap_mb


def test_budgets_cover_all_four_stages():
    assert set(BUDGETS) == {"detector", "landmark", "recognizer", "appearance"}
    assert total_hard_cap_mb() == 20.0


def test_small_module_within_budget():
    tiny = nn.Linear(10, 10)                              # 110 params
    assert_within_budget("detector", tiny)               # no raise


def test_oversized_module_raises():
    big = nn.Linear(4000, 4000)                          # ~16M params > 2M cap
    with pytest.raises(AssertionError, match="detector"):
        assert_within_budget("detector", big)


def test_count_params_counts_trainable():
    assert count_params(nn.Linear(10, 10)) == 110
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "model-v2" && python -m pytest tests/test_size_budget.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aslv2.size_budget'`

- [ ] **Step 3: Implement `size_budget.py`**

`model-v2/src/aslv2/size_budget.py`:
```python
"""Per-stage size budget (spec §3.7). Caps are int8-quantized; param caps assume
~1 byte/param. Each component plan calls assert_within_budget() before export so
no single model (esp. the landmark model) can eat the whole budget."""
import torch.nn as nn

# (max_params, max_quantized_mb) per component
BUDGETS: dict[str, tuple[float, float]] = {
    "detector":    (2_000_000, 2.0),
    "landmark":    (3_000_000, 6.0),
    "recognizer":  (1_500_000, 3.0),
    "appearance":  (1_000_000, 2.0),
}
HARD_CAP_MB = 20.0      # inviolable total across all components
TARGET_MB = 13.0


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def assert_within_budget(name: str, module: nn.Module) -> None:
    max_params, _ = BUDGETS[name]
    n = count_params(module)
    assert n <= max_params, (
        f"{name} has {n:,} params > cap {int(max_params):,} (spec §3.7); "
        "shrink width/depth rather than borrowing from another stage's budget")


def total_hard_cap_mb() -> float:
    return HARD_CAP_MB
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "model-v2" && python -m pytest tests/test_size_budget.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Run the whole suite + commit**

Run: `cd "model-v2" && python -m pytest -q`
Expected: all tests pass (Tasks 1–12).

```bash
git add model-v2/src/aslv2/size_budget.py model-v2/tests/test_size_budget.py
git commit -m "feat(aslv2): per-stage size-budget caps + enforcement (spec 3.7)"
```

---

## Self-Review (done while writing)

**Spec coverage (this plan's slice):**
- §3.3 stable hand slotting → Task 4 ✓; head-normalized geometry → Task 5 ✓.
- §3.6 head-anchor smoothing + no-head fallback + present flag → Task 3 ✓; the head-stability *metric* → Task 11 ✓ (computed against the audit slice).
- §9 item 5 required labeled audit slice → Task 10 builds the schema/loader + min-coverage gate ✓; the actual hand-labeling of ~50–100 real ASL frames happens in Plans 2–3 using this loader.
- §3.6/§8 per-stage metrics (detection-rate, head stability, PCK) → Task 11 ✓ (foundation utilities; Plans 2–4 call them on the labeled slice).
- §3.7 per-stage size budgets → Task 12 ✓ (caps + `assert_within_budget`; each later plan asserts its real model before export).
- §6 prove-first export (combined graph, NMS/roi_align/grid_sample, ORT-Web) → Tasks 7-9 ✓, with the explicit (a)-fallback recorded.
- Detector/landmark/recognizer **real** training → Plans 2-4 (out of scope here, shapes stubbed in Task 6; budgets/metrics/audit utilities ready for them).
- ADR-0001 calibration/live-tester, WLASL eval → Plan 5 (no real model yet).

**Placeholder scan:** no TBDs; every code step has complete code; the one *intended* branch (export may legitimately fail → record (a)) is specified, not vague.

**Type consistency:** `GEOM_DIM=93` defined in Task 5 and reused in Task 6/7; `slot_hands`→`normalize_geometry` array shapes `(2,21,2)`/`(2,)`/`(4,)` consistent; `resolve_head_anchor` returns `(anchor, present)` used by `normalize_geometry(head_present=...)`.
