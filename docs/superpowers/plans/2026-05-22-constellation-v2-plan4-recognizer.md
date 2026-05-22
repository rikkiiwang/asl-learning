# Constellation v2 — Plan 4: Cache + Stage-2 Recognizer (A→B) + Calibration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Cache per-clip geometry + crops by running the trained detector+landmark over the ASL Citizen 75-sign subset, then train the landmark-primary recognizer from scratch — **A (geometry-only) first, gated to beat v1's 16%**, then **B (+ small appearance CNN)** — and calibrate per-class thresholds per ADR 0001.

**Architecture:** Plan 4 of 5. Per frame: pose-geometry (93-d, primary) + optional small hand-appearance embedding → fuse → temporal head (attn pool; optional 1-2 layer Transformer) → 75 logits. Reuses `aslv2.geometry`, `aslv2.detect.infer`, `aslv2.landmark.infer`, `aslv2.size_budget`. Validation/calibration follows **ADR 0001** (held-out datasets headline; live-tester separate).

**Tech Stack:** PyTorch (MPS/CUDA), NumPy, pytest. Reuses v1's frozen 75-sign vocab + signer splits in `model/artifacts/manifest/`.

---

## Conventions
- Clip = 16 frames (frozen contract). Cache stores, per clip: `geom (16,93)`, `hand_crops (16,2,3,64,64)` (zeros for absent slot), `y` (label), `participant` (for signer split).
- Recognizer budgets: `assert_within_budget("recognizer", rec)` (≤1.5 M) and `assert_within_budget("appearance", app)` (≤1.0 M).
- Calibration on the **ASL Citizen signer-held-out val split only** (ADR 0001). Output `meta.json` matches the app contract (`MODEL_WORKSTREAM.md §B`).

## File Structure
```
model-v2/src/aslv2/recog/
  __init__.py
  cache.py       # build per-clip geom+crops cache from detector+landmark over subset
  clip_features.py  # PURE: frames+detections+keypoints -> (geom(16,93), crops) [TDD]
  data.py        # RecogDataset over the cache, signer-held-out splits
  model.py       # RecognizerA (geom-only) + RecognizerB (+appearance), temporal head
  train.py       # config-driven training
  calibrate.py   # temperature scaling + per-class thresholds -> meta.json
configs/recog_a.yaml, configs/recog_b.yaml
artifacts/cache/constellation_clips.npz
artifacts/checkpoints/recog_a/best.pt, recog_b/best.pt
models/asl-v0.2.meta.json   # written here, finalized in Plan 5
tests/test_clip_features.py, test_recog_data.py, test_recog_model.py, test_calibrate.py
```

---

## Task 1: Per-clip feature builder (PURE, TDD)

**Files:** Create `recog/__init__.py` (empty), `recog/clip_features.py`; Test `tests/test_clip_features.py`. Reuses `aslv2.geometry` (resolve_head_anchor, slot_hands, normalize_geometry) + `aslv2.landmark.infer.map_to_frame`.

This isolates the cache logic from the heavy models so it is unit-testable with fake detections/keypoints.

- [ ] **Step 1: Failing test** — feed a synthetic clip: 16 frames of fake per-frame detections (hand boxes + head box) + fake per-hand keypoints. Assert `build_clip_features(...)` returns `geom (16,93)` (no NaN), `crops (16,2,3,64,64)`, and that a frame with no head still produces finite geometry (head fallback), and a one-hand frame zero-pads the empty slot.
- [ ] **Step 2: FAIL** → **Step 3: Implement** `build_clip_features(frames, dets_per_frame, kps_per_frame, crop_size=64)`:
  1. `head_anchor, head_present = resolve_head_anchor(stack_of_head_boxes, frame_wh)` (per-clip smoothed anchor).
  2. `slots, present = slot_hands(list_of_hand_boxes_per_frame, head_cx=center of head_anchor)`.
  3. Per frame/slot: map that slot's keypoints to frame coords; assemble `kps (2,21,2)` (NaN for empty slots).
  4. `geom[f] = normalize_geometry(kps, present[f], head_anchor, head_present)`.
  5. Crops: for each present slot, crop the hand box from the frame, resize to 64²; absent slot → zeros.
  Return `geom (16,93)`, `crops (16,2,3,64,64)`. **Step 4: PASS** → **Step 5: Commit** `feat(recog): pure per-clip feature builder`.

---

## Task 2: Build the cache (run-step)

**Files:** Create `recog/cache.py`.

- [ ] **Step 1:** Implement `cache.py`: load the v1 75-sign manifest (`model/artifacts/manifest/manifest.json` + `signer_splits.json`), for each subset clip decode 16 frames (reuse v1's temporal-resample logic / frame indices), run `aslv2.detect.infer.detect_frame` + `aslv2.landmark.infer.landmark_hand` per frame, then `build_clip_features`. Save `geom`, `crops`, `y`, `participant` to `artifacts/cache/constellation_clips.npz` (or memmap if RAM-bound, mirroring v1's `frames.dat`).
- [ ] **Step 2 (RUN):** `cd model-v2 && python -m aslv2.recog.cache --out artifacts/cache/constellation_clips.npz 2>&1 | tee artifacts/cache/cache.log`
- [ ] **Step 3 (GATE — sanity):** Print head-present rate and mean hands-present per clip; **head-present ≥ 0.9, ≥1 hand present in ≥0.95 of frames**. If low, the detector is failing on ASL frames → return to Plan 2 Task 9 (fine-tune) before training. Record. (Cache .npz is gitignored; commit `cache.py` + `cache.log`.)

---

## Task 3: Recognizer dataset (signer-held-out)

**Files:** Create `recog/data.py`; Test `tests/test_recog_data.py`.

- [ ] **Step 1: Failing test** — with a tiny synthetic cache npz (a few clips, 2 participants) + a signer-split policy, assert `RecogDataset(cache, "train"|"val"|"test", split_policy)` returns `(geom (16,93), crops (16,2,3,64,64), y)` and that **no participant appears in two splits**.
- [ ] **Step 2: FAIL** → **Step 3: Implement** mirroring v1 `model/src/asl/dataset.py` (split by `participant` via the signer-split policy; train-time geometry augmentation = small keypoint jitter + the existing crop augmentation). **Step 4: PASS** → **Step 5: Commit** `feat(recog): signer-held-out recognizer dataset`.

---

## Task 4: Recognizer A (geometry-only) + budget

**Files:** Create `recog/model.py`; Test `tests/test_recog_model.py`.

- [ ] **Step 1: Failing test**
```python
import torch
from aslv2.recog.model import RecognizerA
from aslv2.size_budget import assert_within_budget

def test_recognizer_a_shape_and_budget():
    m = RecognizerA(n_classes=75, head="attn").eval()
    logits = m(torch.randn(2, 16, 93))         # geom only
    assert logits.shape == (2, 75)
    assert_within_budget("recognizer", m)      # <= 1.5M params (spec §3.7)
```
- [ ] **Step 2: FAIL** → **Step 3: Implement** `RecognizerA`: per-frame geom MLP (93→D) → temporal head (reuse v1 `AttnPool`; optional `TransformerEncoder` flag) → `Linear(D,75)`. **Step 4: PASS** → **Step 5: Commit** `feat(recog): geometry-only recognizer A (<=1.5MB)`.

---

## Task 5: Train Recognizer A (run-and-gate — the thesis test)

**Files:** Create `recog/train.py`, `configs/recog_a.yaml`.

- [ ] **Step 1:** Implement `train.py` (mirror v1 `train.py`: AdamW, label smoothing, warmup+cosine, early stop on signer-held-out val top-1; logs history + best.pt). It accepts `--variant a|b`.
- [ ] **Step 2:** `configs/recog_a.yaml` — `variant:a, emb:192, head:attn, dropout:0.3, lr:2e-3, weight_decay:0.05, epochs:80, batch_size:64, warmup_epochs:3, label_smoothing:0.1, early_stop_patience:15, seed:1337`.
- [ ] **Step 3 (RUN):** `cd model-v2 && python -m aslv2.recog.train --config configs/recog_a.yaml 2>&1 | tee artifacts/checkpoints/recog_a/train.log`
- [ ] **Step 4 (GATE — the headline):** signer-held-out **val top-1 must beat v1's 16.4%** (target **≥ 35%** as a meaningful win for an appearance-invariant geometry-only model; if 16–35%, it still validates the thesis but flag for iteration). Record val + test top-1. If it does NOT beat v1, STOP and diagnose (cache quality? keypoint quality? geometry bug?) before adding capacity.
- [ ] **Step 5: Commit** train.py, config, logs, history.

---

## Task 6: Recognizer B (+ small appearance) + train (run-and-gate)

**Files:** Modify `recog/model.py` (add `RecognizerB`); add `configs/recog_b.yaml`.

- [ ] **Step 1: Failing test** (append to `tests/test_recog_model.py`) — `RecognizerB(n_classes=75)(geom (2,16,93), crops (2,16,2,3,64,64))` → `(2,75)`; `assert_within_budget("appearance", m.appearance)` (≤1.0 M) and `assert_within_budget("recognizer", m.recognizer_core)` (≤1.5 M).
- [ ] **Step 2: FAIL** → **Step 3: Implement** `RecognizerB`: a tiny shared appearance CNN on each hand crop → small embedding; concat per-frame with geom → fuse → same temporal head. **Step 4: PASS** → **Step 5: Commit** `feat(recog): recognizer B with small appearance stream`.
- [ ] **Step 6:** `configs/recog_b.yaml` (variant:b; otherwise like A). **RUN:** `python -m aslv2.recog.train --config configs/recog_b.yaml 2>&1 | tee artifacts/checkpoints/recog_b/train.log`.
- [ ] **Step 7 (GATE — earn the stream):** B's signer-held-out val top-1 must **beat A** to be kept (spec §4 discipline). If it doesn't beat A, ship A and record that the appearance stream didn't earn its place. Record both numbers.

---

## Task 7: Calibration + meta.json (ADR 0001)

**Files:** Create `recog/calibrate.py`; Test `tests/test_calibrate.py`.

- [ ] **Step 1: Failing test** — `temperature_scale(logits, labels)` returns T>0 reducing NLL on a synthetic miscalibrated set; `per_class_thresholds(probs, labels, target_fpr=0.05)` returns a threshold per class such that the val false-pass rate ≤ target.
- [ ] **Step 2: FAIL** → **Step 3: Implement** temperature scaling (1-param LBFGS on val logits) + per-class threshold + margin search on the **ASL Citizen val split** (ADR 0001). **Step 4: PASS** → **Step 5: Commit** `feat(recog): temperature scaling + per-class thresholds`.
- [ ] **Step 6 (RUN):** Run calibration on the winning recognizer; write `models/asl-v0.2.meta.json` = `{version, input:{frames:16,size:112,...}, labels:[...75], thresholds:{label:{t,margin}}, hints:{...}, temperature:T}` (labels/hints reuse v1 `proposed_vocab.json` + hint metadata). **Gate:** val false-pass rate ≤ 5% per class. **Commit** `calibrate.py` + `meta.json`.

---

## Task 8: Per-class metrics + confusion matrix

- [ ] **Step 1 (RUN):** `scripts/recog_report.py`: on signer-held-out **test**, compute per-class accuracy, top-1/top-2 confusion matrix (visually-similar signs), false-pass/false-fail per class after thresholds. Write `artifacts/checkpoints/recog_<winner>/test_report.json` + a confusion PNG.
- [ ] **Step 2:** Record the **v1 (16%/18%) → v2** comparison number prominently (feeds Plan 5's report). **Commit** report + script.

---

## Self-Review
- Spec coverage: §3.3 recognizer (pose-geometry primary A; small appearance B under the held-out gate) ✓; §4 A-first-then-B discipline (Tasks 5/6 gates) ✓; §3.7 budgets asserted (Tasks 4/6) ✓; ADR-0001 calibration on val split + meta.json (Task 7) ✓; v1→v2 comparison (Task 8) ✓.
- TDD: clip_features, data, model, calibrate have failing-first tests; caching + training + calibration are run-and-gate with numeric thresholds (beat 16%, B-beats-A, FPR≤5%).
- Integration: `build_clip_features` reuses geometry exactly; cache feeds `RecogDataset`; meta.json matches `MODEL_WORKSTREAM.md §B` so the app consumes it unchanged. The winning recognizer + detector + landmark are the inputs to Plan 5's combined export.
