# Constellation v2 — Plan 3: Stage-1.5 Hand-Landmark Model (21 keypoints, from scratch)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Train, from scratch, a small model that regresses **21 2D hand keypoints** from a hand crop, and gate it (PCK) on the labeled ASL-frame audit slice. These keypoints are the recognizer's *primary* (appearance-invariant) signal.

**Architecture:** Plan 3 of 5. Tiny CNN crop→`(21,2)` regressor. Trained on keypoint-labeled hand data (FreiHAND primary; optionally COCO-WholeBody hands for frontal variety). Reuses `aslv2.metrics.pck`, `aslv2.audit`, `aslv2.size_budget` (≤3 MB cap). Builds in `model-v2/`.

**Tech Stack:** PyTorch, NumPy, OpenCV, pytest.

---

## Conventions
- Landmark input: a hand crop resized to **64×64 RGB**, normalized. Output: `(21,2)` keypoints in **crop coords normalized to [0,1]** (multiply by crop size to get pixels; map to frame coords via the detector's hand box at the call site).
- Keypoint ordering follows the source dataset's 21-point convention; document it in `PROVENANCE.md`.
- Budget: `assert_within_budget("landmark", model)` (≤3.0 M params).

## File Structure
```
model-v2/src/aslv2/landmark/
  __init__.py
  data.py       # keypoint dataset (FreiHAND/COCO-WholeBody hands) -> (crop, kp[0,1])
  model.py      # tiny CNN -> 42 outputs (21x2)
  loss.py       # wing loss (robust keypoint regression)
  train.py      # config-driven training -> checkpoint
  infer.py      # landmark_hand(model, crop) -> (21,2) normalized; map_to_frame(kp, box)
configs/landmark.yaml
artifacts/checkpoints/landmark/best.pt
artifacts/landmark/PROVENANCE.md
tests/test_landmark_data.py, test_landmark_model.py, test_landmark_loss.py, test_landmark_infer.py
```

---

## Task 1: Keypoint dataset loader

**Files:** Create `model-v2/src/aslv2/landmark/__init__.py` (empty), `landmark/data.py`; Test `tests/test_landmark_data.py`.

- [ ] **Step 1: Failing test** — write a synthetic sample (a tiny PNG + a record with 21 xy keypoints in pixel coords + a hand box); assert the loader returns `crop` tensor `(3,64,64)` and `kp` `(21,2)` with values in `[0,1]` (normalized to the crop), and that a keypoint at the crop center maps to ≈`[0.5,0.5]`.
- [ ] **Step 2: FAIL** → **Step 3: Implement** `KpDataset(manifest, norm, train)`: crop the hand region (with small margin), resize to 64², normalize image, normalize keypoints to crop. Augment (train): brightness/contrast, small rotation/scale that also transforms keypoints, **no flip** unless verified. **Step 4: PASS** → **Step 5: Commit** `feat(landmark): keypoint dataset loader`.
- [ ] **Step 6 (data acquisition, run-step):** Produce `artifacts/landmark/{train,val}.json` from **FreiHAND** (+ optional COCO-WholeBody hands). **Confirm license** (spec §9 item 1/3); record in `PROVENANCE.md` incl. the 21-point ordering. Gate: ≥ a few thousand labeled hands, both train/val. Commit manifests + provenance (images gitignored).

---

## Task 2: Landmark model + budget

**Files:** Create `landmark/model.py`; Test `tests/test_landmark_model.py`.

- [ ] **Step 1: Failing test**
```python
import torch
from aslv2.landmark.model import Landmark
from aslv2.size_budget import assert_within_budget

def test_landmark_shape_and_budget():
    m = Landmark().eval()
    out = m(torch.randn(4, 3, 64, 64))
    assert out.shape == (4, 21, 2)
    assert_within_budget("landmark", m)        # <= 3.0M params (spec §3.7)
```
- [ ] **Step 2: FAIL** → **Step 3: Implement** depthwise-separable CNN (stem→blocks→GAP→`Linear(·, 42)`→`reshape(-1,21,2)`→`sigmoid` to bound in [0,1]). Width chosen to stay <3 M. **Step 4: PASS** → **Step 5: Commit** `feat(landmark): tiny keypoint model (<=3MB)`.

---

## Task 3: Wing loss

**Files:** Create `landmark/loss.py`; Test `tests/test_landmark_loss.py`.

- [ ] **Step 1: Failing test** — perfect prediction → loss ≈ 0; a fixed small error gives a known positive value; loss is lower for a near-correct than a far-off prediction.
- [ ] **Step 2: FAIL** → **Step 3: Implement** `wing_loss(pred, gt, w=0.1, eps=0.02)` (standard wing loss, robust to outliers; reduces over the 42 coords). **Step 4: PASS** → **Step 5: Commit** `feat(landmark): wing loss`.

---

## Task 4: Inference + frame mapping

**Files:** Create `landmark/infer.py`; Test `tests/test_landmark_infer.py`.

- [ ] **Step 1: Failing test** — `landmark_hand(model, crop_rgb, norm)` returns `(21,2)` in [0,1]; `map_to_frame(kp01, box_xyxy)` maps normalized keypoints into original-frame pixel coords (a keypoint `[0.5,0.5]` with box `[10,20,50,60]` → `[30,40]`).
- [ ] **Step 2: FAIL** → **Step 3: Implement** both. **Step 4: PASS** → **Step 5: Commit** `feat(landmark): inference + frame mapping`.

---

## Task 5: Add hand keypoints to the ASL audit slice (extends Plan 2 Task 7)

- [ ] **Step 1 (HUMAN):** In `artifacts/audit/asl_audit_slice.json`, add the 21-keypoint labels (per hand box, same ordering as the dataset) to each frame.
- [ ] **Step 2:** Validate it loads with keypoints:
```bash
cd model-v2 && python -c "from aslv2.audit import load_audit_slice as L; f=L('artifacts/audit/asl_audit_slice.json'); assert any(x.keypoints is not None for x in f); print('keypoints present')"
```
- [ ] **Step 3: Commit** the updated slice.

---

## Task 6: Train the landmark model (run-and-gate)

**Files:** Create `landmark/train.py`, `configs/landmark.yaml`.

- [ ] **Step 1:** Implement `train.py` (mirror the v1/Plan-2 training structure; AdamW, warmup+cosine, early stop on val PCK). Eval: **PCK@0.2** (`aslv2.metrics.pck`, ref_size = crop size) on the val split. Save `best.pt`.
- [ ] **Step 2:** `configs/landmark.yaml` — `img:64, width:32, lr:3e-3, weight_decay:0.02, epochs:80, batch_size:128, warmup_epochs:3, early_stop_patience:15, seed:1337`.
- [ ] **Step 3 (RUN):** `cd model-v2 && python -m aslv2.landmark.train --config configs/landmark.yaml 2>&1 | tee artifacts/checkpoints/landmark/train.log`
- [ ] **Step 4 (GATE — val):** val **PCK@0.2 ≥ 0.85**. Iterate (augmentation/width/lr) if below. Record.
- [ ] **Step 5: Commit** train.py, config, logs, history.

---

## Task 7: Gate landmark on the ASL audit slice + provenance

- [ ] **Step 1 (RUN):** `scripts/eval_landmark_on_audit.py`: for each audit-slice hand, crop via the labeled box, run `landmark_hand`, map to frame, compute **PCK** vs the manual keypoints (ref_size = hand-box size). Write `artifacts/checkpoints/landmark/audit_eval.json`.
- [ ] **Step 2 (GATE — domain):** ASL-audit **PCK@0.2 ≥ 0.70**. FreiHAND is lab/green-screen → if ASL PCK is weak (domain shift, spec §7), **self-label more ASL hands and fine-tune** before Plan 4 trusts keypoints. Record.
- [ ] **Step 3:** Finalize `PROVENANCE.md` (datasets, license, from-scratch evidence, keypoint ordering). **Commit.**

---

## Self-Review
- Spec coverage: §3.2 landmark (21 2D keypoints, from scratch, ≤3 MB via Task 2 budget assert) ✓; §9 item 5 keypoint labels added (Task 5) ✓; PCK gating on ASL frames (Task 7) ✓; Req-7 provenance ✓.
- TDD: data/model/loss/infer have failing-first tests; training is run-and-gate with numeric PCK thresholds.
- Integration: reuses `aslv2.metrics.pck`, `aslv2.audit`, `aslv2.size_budget`. `landmark_hand` + `map_to_frame` are the contract Plan 4 caching consumes.
- Carry-forward: combined-graph export (Plan 5) runs the landmark on roi_align crops (per the Plan-1 spike); keep crop size 64² consistent with the spike.
