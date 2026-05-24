# ASL v1 — Data Scaling + Optical-Flow Motion Stream

**Date:** 2026-05-24
**Status:** Design approved; revised 2026-05-24 after reading the codebase (see "Revisions" below)

## Revisions (2026-05-24, during planning)

Reading the code surfaced three facts that change the plan:

1. **Augmentation already exists and is enabled** (`dataset.py:augment_clip`, called
   with `train=True` in `train.py`): brightness/contrast + random-resized-crop. So
   Phase A is **strengthen** (add small rotation, widen ranges), not create. Expect a
   smaller, less certain gain than originally implied.
2. **Flow-pretrain on 1500 classes is storage-infeasible** (float32 flow memmap for
   ~38k clips ≈ 90GB, no clean Colab upload path). Replaced with: **warm-start the
   flow encoder from the pretrained RGB encoder's body + a fresh 2-channel stem.** No
   flow-pretrain cache needed.
3. **Temporal jitter conflicts with precomputed flow** (subsampling breaks the frame
   adjacency cached flow assumes). **Temporal jitter is deferred**; the cache stays at
   16 frames (no 24-frame re-cache). Flow is precomputed on the canonical 16 frames.
**Author:** model v1 workstream

## Context

The v1 isolated-sign recognizer (from-scratch CNN encoder + attention-pooled
temporal head, 16×112×112 clips, ONNX <10MB, signer-held-out splits) has climbed
through four iterations:

| Iteration | Test top-1 | Dead signs |
|---|---|---|
| Baseline (center crop) | 18.0% | 33/75 |
| ROI crop | 28.8% | 20/75 |
| Pretrain → fine-tune | 42.9% | 5/75 |
| Stacked (ROI + pretrain) | **45.8%** | 7/75 |

The stacked model is the shipping checkpoint. Its diagnosis — `train ≈ 0.99`,
best-val at **epoch 6** — says the binding constraint is now **data quantity for the
75-class fine-tune set**, not architecture or background. The remaining errors are
genuine look-alike sign pairs distinguished by *movement* (`SORRY↔HUNGRY`,
`WORK↔COOK`, `HELLO↔MAN`).

This iteration pulls three from-scratch-compliant levers to push past 46%.

## Hard constraints (unchanged)

- **From scratch only** — no pretrained weights, no learned backbones, no learned
  landmark/flow models. Farneback optical flow and the motion-ROI crop are
  *classical* algorithms with no trained weights, so they comply.
- **Browser-deployable** — PyTorch → ONNX → ONNX Runtime Web, quantized <10MB.
- **Signer-held-out splits** — val/test signers never appear in train or pretrain.
- Eval always via `analysis.run_analysis` (no DataLoader workers) on the held-out
  test split.

## Data reality (governs the plan)

Full ASL Citizen corpus: **83,399 clips · 2731 glosses · 52 signers**, median
**31 clips/gloss (min 21, max 45)**. The 75-class set already uses **47 of 52
signers** and 29–45 clips/sign — near the per-sign ceiling. ASL Citizen yields ~1
clip per signer per sign, so adding *real* training clips per sign is impossible
without either (a) folding in val/test signers (breaks held-out discipline) or
(b) augmentation. **Therefore "more clips/sign" is reframed as augmentation.**
Bigger pretrain, by contrast, has large headroom (500 → ~1500 glosses).

## Three levers

| Lever | Plan | Rationale |
|---|---|---|
| **A. Augmentation** | Train-only spatial+photometric aug in the dataset loader. | Attacks the ep-6 overfit on a fixed 52-signer set. Free generalization. |
| **B. Bigger pretrain** | Rebuild pretrain manifest to top ~1500 glosses (val/test signers excluded), ROI-cropped; re-pretrain encoder on Colab T4. | Proven lever; raises generalization floor. ~1500 caps cache size / Colab session time vs all 2731. |
| **C. Optical-flow motion stream** | Precompute Farneback (dx,dy); second FrameEncoder (in_ch=2); late fusion. | Targets movement-distinguished confusions that A/B do not specifically address. |

## Build order

Phases have a dependency chain and are each validated before the next.

### Phase A — Augmentation (no re-cache; fastest win)

- Add `augment_clip(frames, rng) -> frames` (pure function, unit-testable) invoked
  in `dataset.py` `__getitem__` only when `split == "train"` and a config flag is set.
- Transforms, applied **identically across all 16 frames** of a clip (a clip is one
  shot): random affine (scale 0.9–1.1, translate ±10%, rotate ±10°), brightness/
  contrast jitter.
- **No horizontal flip** — ASL is handedness-sensitive; flipping corrupts directional
  and one-handed signs.
- Operates on the existing `clips_roi.npz` (16×112×112 uint8) — **no re-preprocessing**.
- **Temporal jitter is NOT in Phase A**: random start/stride needs >16 cached frames,
  which the npz lacks. Deferred to Phase C's re-cache.
- **Validation:** A/B fine-tune (aug on vs off) on the stacked encoder; report test
  top-1 + dead signs. Visual sanity dump of augmented frames.

### Phase B — Bigger pretrain (Colab T4)

- `build_pretrain_manifest.py`: 500 → **top ~1500 glosses by clip count**, val/test
  signers excluded (already enforced).
- `preprocess_pretrain.py --roi`: produces a larger memmap (~23GB local; JPEG-packed
  ≈3GB upload via existing `pack_pretrain_jpeg.py`).
- Re-pretrain encoder (1500-way head) → `encoder.pt`; fine-tune 75-class head **with
  Phase-A augmentation enabled**.
- **Checkpoint/resume support** in `pretrain.py` (save+restore optimizer + epoch) so a
  free-T4 session disconnect (~12h cap) doesn't lose the run.
- **Validation:** encoder val_top1 on the 1500-way task; downstream fine-tune test
  top-1 vs the 500-class-pretrain baseline (42.9%/45.8%).

### Phase C — Optical-flow motion stream (architecture change)

- **Preprocess:** extend `load_clip` to also emit a **2-channel Farneback flow clip**
  `(dx,dy)` per consecutive frame-pair (`cv2.calcOpticalFlowFarneback`). New cache
  stores RGB at **24 frames** (to enable temporal jitter) + flow (2ch). Re-cache both
  the 75-class and 1500-class sets. Flow is computed on the ROI-cropped frames.
  **Frame-count alignment:** the 16 sampled RGB frames yield 15 consecutive flow
  fields; prepend a zero field so flow has 16 entries index-aligned with RGB (flow[t]
  = motion entering frame t, flow[0] = 0). Flow channels are normalized with their own
  mean/std in `norm.json` (separate from RGB).
- **Temporal jitter** now enabled: sample 16 of the 24 cached frames with random
  start/stride at train time (deterministic center sample for val/test).
- **Model (`model.py`):**
  - `FrameEncoder` gains `in_ch` arg (default 3) — minimal change.
  - `TwoStreamClassifier`: `encoder_rgb(in_ch=3)` + `encoder_flow(in_ch=2)`, each →
    `AttnPool` → **concat (2·emb) → dropout → fc**.
  - Each stream initialized from its own pretrained encoder; the **flow encoder is
    also pretrained** on the 1500-class set's flow, so both streams enter fine-tuning
    warm.
- **Export (`export.py`):** ONNX takes **two inputs** (rgb, flow). `meta.json` gains a
  flow-channel input spec.
- **Validation:** A/B fused vs RGB-only at equal data; confirm the target confusions
  (`SORRY↔HUNGRY`, `WORK↔COOK`) drop, not just aggregate accuracy.

## Data flow (Phase C end state)

```
video ─► ROI crop ─┬─► 24 RGB frames ──(temporal jitter → 16)──► aug ─► encoder_rgb ─► attn ─┐
                   └─► Farneback flow ─(15×2ch → 16)────────────────────► encoder_flow ─► attn ┴─► concat ─► fc ─► logits
```

## Component isolation

Each is one job, testable alone:

- `augment_clip(frames, rng)` — shape/dtype/range preserved; visual dump.
- `compute_flow(frames)` — returns `(F, 2, H, W)`; visual dump of flow fields.
- `FrameEncoder(in_ch)` — forward over a (B·F, in_ch, 112, 112) tensor.
- `TwoStreamClassifier` — forward over (rgb, flow) → logits; can run RGB-only by
  zeroing/omitting the flow stream (fallback path).

## Model ↔ App contract impact

- **New inference-contract item:** the app must compute Farneback `(dx,dy)` flow
  in-browser (WASM/JS) on the ROI-cropped capture frames before inference.
- **Fallback:** if the app cannot, the single-stream RGB checkpoint ships instead
  (the RGB encoder is trained standalone in Phase B, so this fallback exists for free).
- Both documented in `MODEL_WORKSTREAM.md` §B and §C.

## Risks & mitigations

1. **Colab session cap** at 1500-class pretrain → checkpoint/resume.
2. **Flow precompute cost** — Farneback over ~38k pretrain clips × frames is heavy
   (one-time, local M4). If too slow, pretrain the flow encoder on a subset of the
   1500-class set.
3. **Browser Farneback** is the deployment risk → RGB-only fallback checkpoint.
4. **Two encoders overfit** the 75-class set → mitigated by aug (A) + pretrained init (B).

## Success criteria

- Phase A: measurable test-top-1 gain or dead-sign reduction from augmentation alone.
- Phase B: 1500-class pretrain ≥ matches 500-class downstream; ideally lifts it.
- Phase C: fused model beats RGB-only at equal data **and** reduces the targeted
  movement-distinguished confusions.
- Overall stretch target: meaningfully past 46% test top-1, with ONNX export still
  <10MB and held-out discipline intact.

## Out of scope

- Transformer temporal head (revisit once data supports the capacity).
- Self-supervised pretraining.
- Adding val/test signers to train (would break held-out evaluation).
- Horizontal-flip augmentation (corrupts ASL handedness).
