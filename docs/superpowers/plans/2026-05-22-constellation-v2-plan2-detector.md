# Constellation v2 — Plan 2: Stage-1 Detector (hand + head, from scratch)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Train, from scratch (no pretrained weights), a tiny single-shot detector that finds **hand** and **head** boxes per frame, and gate it on a manually-labeled ASL-frame audit slice.

**Architecture:** Plan 2 of 5. Anchor-based SSD-style detector on a single stride-16 feature map (8×8 grid for 128² input), **square anchors only for hands** (BlazePalm trick) and for head, 2 foreground classes. Reuses `aslv2.boxes` (iou/nms), `aslv2.metrics` (detection_rate), `aslv2.audit` (labeled slice), `aslv2.size_budget` (≤2 MB cap). Builds in `model-v2/`.

**Tech Stack:** PyTorch (MPS/CUDA), torchvision, NumPy, OpenCV, pytest. Datasets: 100DOH (hands, primary), EgoHands (optional), a head/face-box source for the head class.

---

## Conventions
- Detector input: **128×128 RGB**, per-channel normalized (reuse a `norm.json` mean/std; compute from the detection train set).
- Classes: `0 = hand`, `1 = head`. Boxes xyxy in pixels (model works in 128² space; scale to original at the call site).
- Grid: stride 16 → **8×8 cells**; **A = 3 square anchors/cell** (scales 32, 64, 96 px); total 192 anchors.
- All model code asserts its size with `aslv2.size_budget.assert_within_budget("detector", model)` in a test.

## File Structure
```
model-v2/src/aslv2/detect/
  __init__.py
  anchors.py        # square-anchor grid generation
  encode.py         # GT<->anchor target encoding/decoding + decode-with-nms
  data.py           # detection dataset (100DOH/EgoHands/head) -> (img, boxes, labels)
  model.py          # tiny depthwise-separable backbone + detection head
  loss.py           # focal cls loss + smooth-L1 box loss over matched anchors
  train.py          # config-driven training loop -> checkpoint
  infer.py          # run detector on frames -> per-frame hand(<=2)+head boxes
configs/detector.yaml
scripts/extract_asl_audit_frames.py   # pull N ASL Citizen frames to PNG for labeling
artifacts/audit/asl_audit_slice.json  # MANUALLY LABELED (hand+head boxes; keypoints added in Plan 3)
artifacts/checkpoints/detector/best.pt
tests/test_detect_anchors.py, test_detect_encode.py, test_detect_loss.py,
tests/test_detect_model.py, test_detect_infer.py, test_detect_data.py
```

---

## Task 1: Square-anchor grid

**Files:** Create `model-v2/src/aslv2/detect/__init__.py` (empty), `model-v2/src/aslv2/detect/anchors.py`; Test `model-v2/tests/test_detect_anchors.py`.

- [ ] **Step 1: Failing test**
```python
import numpy as np
from aslv2.detect.anchors import make_anchors

def test_anchor_count_and_squareness():
    a = make_anchors(img=128, stride=16, scales=(32, 64, 96))
    assert a.shape == (8 * 8 * 3, 4)            # 192 anchors, xyxy
    w = a[:, 2] - a[:, 0]; h = a[:, 3] - a[:, 1]
    np.testing.assert_allclose(w, h)            # square
    # first cell center is at (8,8); smallest anchor spans 8±16
    np.testing.assert_allclose(a[0], [8 - 16, 8 - 16, 8 + 16, 8 + 16])
```
- [ ] **Step 2: Run, expect FAIL** — `cd model-v2 && python -m pytest tests/test_detect_anchors.py -v`
- [ ] **Step 3: Implement**
```python
"""Square anchor grid for the from-scratch detector (BlazePalm-style: hands are
rigid blobs, so square anchors only)."""
import numpy as np

def make_anchors(img=128, stride=16, scales=(32, 64, 96)) -> np.ndarray:
    n = img // stride
    cs = (np.arange(n) + 0.5) * stride
    cx, cy = np.meshgrid(cs, cs)
    cx, cy = cx.reshape(-1), cy.reshape(-1)
    out = []
    for cxi, cyi in zip(cx, cy):
        for s in scales:
            out.append([cxi - s / 2, cyi - s / 2, cxi + s / 2, cyi + s / 2])
    return np.asarray(out, dtype=np.float32)
```
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** — `git add model-v2/src/aslv2/detect/__init__.py model-v2/src/aslv2/detect/anchors.py model-v2/tests/test_detect_anchors.py && git commit -m "feat(detect): square-anchor grid"`

---

## Task 2: Target encoding + decode-with-NMS

**Files:** Create `model-v2/src/aslv2/detect/encode.py`; Test `model-v2/tests/test_detect_encode.py`. Reuses `aslv2.boxes.iou`/`nms`.

- [ ] **Step 1: Failing test**
```python
import numpy as np
from aslv2.detect.anchors import make_anchors
from aslv2.detect.encode import encode_targets, decode

def test_encode_assigns_best_anchor_positive():
    anchors = make_anchors()
    gt = np.array([[0, 0, 64, 64]], float); labels = np.array([0])   # one hand
    cls_t, box_t, pos = encode_targets(anchors, gt, labels, iou_pos=0.5)
    assert pos.sum() >= 1                          # at least one positive anchor
    assert set(np.unique(cls_t[pos])) <= {0}       # positives labelled hand(0)
    assert (cls_t[~pos] == -1).all()               # background = -1 (ignored idx)

def test_decode_roundtrip_recovers_box():
    anchors = make_anchors()
    gt = np.array([[10, 12, 70, 78]], float); labels = np.array([1])
    cls_t, box_t, pos = encode_targets(anchors, gt, labels)
    # build perfect predictions: deltas = box_t, score=+inf on positives
    scores = np.full((len(anchors), 2), -9.0); scores[pos, 1] = 9.0
    boxes, scs, lbl = decode(anchors, box_t, scores, score_thr=0.5, iou_thr=0.5)
    assert len(boxes) == 1 and lbl[0] == 1
    np.testing.assert_allclose(boxes[0], gt[0], atol=1.0)
```
- [ ] **Step 2: Run, expect FAIL**
- [ ] **Step 3: Implement** (standard SSD encoding: deltas in center form; positives = anchors with IoU≥`iou_pos` to any GT, plus the best anchor per GT; background = IoU<`iou_neg`; in-between ignored)
```python
"""Anchor<->GT encoding (SSD-style) and inference decode with NMS."""
import numpy as np
from aslv2.boxes import iou, nms

def _xyxy_to_cwh(b):
    return np.stack([(b[:,0]+b[:,2])/2, (b[:,1]+b[:,3])/2, b[:,2]-b[:,0], b[:,3]-b[:,1]], 1)

def encode_targets(anchors, gt, labels, iou_pos=0.5, iou_neg=0.4):
    n = len(anchors)
    cls_t = np.full(n, -1, np.int64)      # -1 = ignore
    box_t = np.zeros((n, 4), np.float32)
    pos = np.zeros(n, bool)
    if len(gt) == 0:
        cls_t[:] = 2                      # all background (class index 2 = bg)
        return cls_t, box_t, pos
    ious = np.zeros((n, len(gt)))
    for i in range(n):
        for j in range(len(gt)):
            ious[i, j] = iou(anchors[i], gt[j])
    best_gt = ious.argmax(1); best_iou = ious.max(1)
    for j in range(len(gt)):              # force best anchor per GT positive
        best_iou[ious[:, j].argmax()] = 1.0; best_gt[ious[:, j].argmax()] = j
    pos = best_iou >= iou_pos
    cls_t[best_iou < iou_neg] = 2         # background
    a = _xyxy_to_cwh(anchors); g = _xyxy_to_cwh(gt)
    for i in np.where(pos)[0]:
        j = best_gt[i]; cls_t[i] = labels[j]
        box_t[i] = [(g[j,0]-a[i,0])/a[i,2], (g[j,1]-a[i,1])/a[i,3],
                    np.log(g[j,2]/a[i,2]), np.log(g[j,3]/a[i,3])]
    return cls_t, box_t, pos

def decode(anchors, deltas, scores, score_thr=0.5, iou_thr=0.5):
    a = _xyxy_to_cwh(anchors)
    cx = deltas[:,0]*a[:,2]+a[:,0]; cy = deltas[:,1]*a[:,3]+a[:,1]
    w = np.exp(deltas[:,2])*a[:,2]; h = np.exp(deltas[:,3])*a[:,3]
    boxes = np.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], 1)
    # scores: (n, 2) logits per foreground class
    prob = 1/(1+np.exp(-scores)); cls = prob.argmax(1); conf = prob.max(1)
    keep_b, keep_s, keep_l = [], [], []
    for c in (0, 1):
        m = (cls == c) & (conf >= score_thr)
        if m.sum() == 0: continue
        idx = nms(boxes[m], conf[m], iou_thr)
        bm = boxes[m]; cm = conf[m]
        for k in idx:
            keep_b.append(bm[k]); keep_s.append(cm[k]); keep_l.append(c)
    return (np.array(keep_b).reshape(-1,4), np.array(keep_s), np.array(keep_l, int))
```
- [ ] **Step 4: Run, expect PASS** — [ ] **Step 5: Commit** `feat(detect): anchor target encoding + decode/NMS`

---

## Task 3: Detection dataset loader

**Files:** Create `model-v2/src/aslv2/detect/data.py`; Test `model-v2/tests/test_detect_data.py`.

Loader reads a unified JSON manifest (`[{image, boxes:[[x1,y1,x2,y2]], labels:[0|1]}]`) produced by per-dataset adapters; resizes to 128² (scaling boxes), normalizes, returns `(tensor(3,128,128), boxes(N,4), labels(N,))`. Test against a tiny synthetic image+manifest written to `tmp_path`.

- [ ] **Step 1: Failing test** — write a 1-frame synthetic manifest + a 256×256 PNG; assert loaded tensor is `(3,128,128)` and a box at `[0,0,128,128]` in the 256 image maps to `[0,0,64,64]` at 128².
- [ ] **Step 2: FAIL** → **Step 3: Implement** `DetDataset(manifest_path, norm, train)` (OpenCV decode, resize, box-scale, augment: brightness/contrast + small scale/translate that also transforms boxes; **no horizontal flip** unless verified). **Step 4: PASS** → **Step 5: Commit** `feat(detect): detection dataset loader`.
- [ ] **Step 6 (data acquisition, documented run-step):** Write `scripts/adapters/` notes + run adapters to produce `artifacts/detect/{train,val}.json` from **100DOH** (hands) + **head/face-box source**. **Confirm dataset licenses** permit the pilot (spec §9 item 3) and record provenance in `artifacts/detect/PROVENANCE.md`. Gate: `python -c "import json;d=json.load(open('artifacts/detect/train.json'));print(len(d))"` ≥ a few thousand frames with both classes present. Commit the manifests + provenance (NOT the images — gitignored).

---

## Task 4: Detector model + size-budget assertion

**Files:** Create `model-v2/src/aslv2/detect/model.py`; Test `model-v2/tests/test_detect_model.py`.

- [ ] **Step 1: Failing test**
```python
import torch
from aslv2.detect.model import Detector
from aslv2.size_budget import assert_within_budget

def test_detector_shapes_and_budget():
    m = Detector(n_classes=2, n_anchors=3).eval()
    cls, box = m(torch.randn(2, 3, 128, 128))
    assert cls.shape == (2, 8 * 8 * 3, 2)      # per-anchor 2-class logits
    assert box.shape == (2, 8 * 8 * 3, 4)
    assert_within_budget("detector", m)        # <= 2.0M params (spec §3.7)
```
- [ ] **Step 2: FAIL** → **Step 3: Implement** a depthwise-separable backbone (reuse the `SepConv` pattern from v1 `model/src/asl/model.py`: stem stride2 → blocks to stride16, width ~32) → two 1×1 conv heads (cls: `A*2`, box: `A*4`), reshaped to `(B, H*W*A, ·)`. Keep width small to stay <2 M params. **Step 4: PASS** → **Step 5: Commit** `feat(detect): tiny detector model (<=2MB)`.

---

## Task 5: Loss

**Files:** Create `model-v2/src/aslv2/detect/loss.py`; Test `model-v2/tests/test_detect_loss.py`.

- [ ] **Step 1: Failing test** — perfect predictions (cls logits huge-correct, box deltas == targets) give ≈0 loss; wrong-class predictions give a clearly larger loss. (Numeric, small tensors.)
- [ ] **Step 2: FAIL** → **Step 3: Implement** `det_loss(cls_logits, box_pred, cls_t, box_t, pos)`: sigmoid-focal BCE over foreground/background for cls (ignore `cls_t==-1`), smooth-L1 on `box_pred[pos]` vs `box_t[pos]`, normalized by `max(pos.sum(),1)`. **Step 4: PASS** → **Step 5: Commit** `feat(detect): focal cls + smooth-L1 box loss`.

---

## Task 6: Decode→per-frame hand/head selection (inference)

**Files:** Create `model-v2/src/aslv2/detect/infer.py`; Test `model-v2/tests/test_detect_infer.py`.

- [ ] **Step 1: Failing test** — given a `Detector` + a frame, `detect_frame(model, frame_rgb, norm)` returns `{"hands": (<=2,4), "head": (4,) or None}` in **original-frame pixel coords** (test with a 256² frame, assert coords scaled back from 128²; cap hands at top-2 by score; head = top-1).
- [ ] **Step 2: FAIL** → **Step 3: Implement** using `make_anchors` + `decode`; scale boxes by `orig/128`. **Step 4: PASS** → **Step 5: Commit** `feat(detect): per-frame hand/head inference`.

---

## Task 7: Build + manually label the ASL audit slice (spec §9 item 5, REQUIRED)

**Files:** `scripts/extract_asl_audit_frames.py`; output `artifacts/audit/asl_audit_slice.json`.

- [ ] **Step 1:** Implement `extract_asl_audit_frames.py` to sample **~80 frames** across ≥8 ASL Citizen signers/lighting conditions (from the v1 subset videos) → write PNGs to `artifacts/audit/frames/` + a stub `asl_audit_slice.json` with `image` + empty `hands`/`head`.
- [ ] **Step 2 (HUMAN):** Manually label hand + head **boxes** in each frame (any box tool exporting xyxy). Keypoints are added in Plan 3. Fill `asl_audit_slice.json`.
- [ ] **Step 3:** Validate it loads + meets coverage:
```bash
cd model-v2 && python -c "from aslv2.audit import load_audit_slice, assert_min_coverage; f=load_audit_slice('artifacts/audit/asl_audit_slice.json'); assert_min_coverage(f, 50); print(len(f),'frames OK')"
```
- [ ] **Step 4: Commit** the JSON (+ extract script). Frames PNGs are gitignored under `artifacts/audit/frames/` — add that to ignore.

---

## Task 8: Train the detector (run-and-gate)

**Files:** Create `model-v2/src/aslv2/detect/train.py`, `model-v2/configs/detector.yaml`.

- [ ] **Step 1:** Implement `train.py` (config-driven; mirror v1 `model/src/asl/train.py` structure: seeds, AdamW, warmup+cosine, early stop on val). Each epoch: forward `Detector`, `det_loss`, eval val **detection-rate@0.5** (decode + `aslv2.metrics.detection_rate` against val GT). Save `artifacts/checkpoints/detector/best.pt`.
- [ ] **Step 2:** `configs/detector.yaml` — `img:128, anchors:3, width:32, lr:3e-3, weight_decay:0.02, epochs:60, batch_size:64, warmup_epochs:3, early_stop_patience:12, seed:1337`.
- [ ] **Step 3 (RUN):** `cd model-v2 && python -m aslv2.detect.train --config configs/detector.yaml 2>&1 | tee artifacts/checkpoints/detector/train.log`
- [ ] **Step 4 (GATE — val):** val detection-rate@0.5 **≥ 0.85 for head**, **≥ 0.70 for hand**. If below, iterate (more data/augmentation/anchor scales) before proceeding. Record the numbers.
- [ ] **Step 5: Commit** `train.py`, config, `train.log`, `history.json` (checkpoint `.pt` is gitignored).

---

## Task 9: Gate the detector on the ASL audit slice + provenance

- [ ] **Step 1 (RUN):** Script `scripts/eval_detector_on_audit.py`: run `detect_frame` on every audit-slice frame, compute `detection_rate` (hand + head separately) vs the manual labels, and head-box **stability** is N/A here (single frames). Write `artifacts/checkpoints/detector/audit_eval.json`.
- [ ] **Step 2 (GATE — domain):** ASL-audit detection-rate@0.5 **≥ 0.80 head, ≥ 0.60 hand**. If the from-100DOH detector underperforms on ASL frames (domain shift, spec §7), **self-label more ASL frames and fine-tune** (the §9 optional extension becomes active) before Plan 4 trusts the crops.
- [ ] **Step 3:** Write `artifacts/detect/PROVENANCE.md` (datasets, versions, licenses, from-scratch init evidence — Req 7/15). **Commit.**

---

## Self-Review
- Spec coverage: §3.1 detector (hand+head, square anchors, from scratch, ≤2 MB cap via Task 4 budget assert) ✓; §9 item 5 audit slice built+labeled (Tasks 7) ✓; per-stage metric gating on ASL frames (Task 9) ✓; Req-7 provenance (Task 9) ✓.
- TDD: anchors/encode/decode/loss/model/infer/data all have failing-first tests with real code; training is an explicit run-and-gate with numeric thresholds.
- Integration: reuses `aslv2.boxes`, `aslv2.metrics`, `aslv2.audit`, `aslv2.size_budget`. Output `detect_frame` is the contract Plan 4's caching consumes.
- Carry-forward: detector decode uses numpy NMS for offline caching; the *combined export* graph (Plan 5) keeps the static topk path proven in Plan 1, not this NMS.
