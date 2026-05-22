# ASL Vision Model — Data Processing, Model Selection & Training Plan

Parallel workstream to the system/UI design. This plan turns the high-level `architecture.md` into an executable, time-boxed procedure for a **solo build in ~1–2 weeks**, and it bakes in the three risks surfaced in the architecture audit.

---

## 1. Objectives & Success Criteria

Build a **from-scratch** (no pretrained weights, backbones, or landmark/pose detectors) recognizer for **75–100 isolated beginner ASL signs** that:

- Accepts a **~3s capture window uniformly resampled to a fixed 16 frames @ 112×112** — matching the app's countdown-record capture — and returns logits over the vocabulary. The window is generous (beginner pace + reaction time); temporal resampling to a fixed count decouples input shape from sign tempo. (Frozen with the app stream 2026-05-21; source of truth `model/MODEL_WORKSTREAM.md §B`.)
- Produces a **conservative pass/fail** decision that is **biased against false passes** (a learner should not pass when the model is unsure — PRD Req 9).
- Runs **in the browser**, model file **2–6 MB quantized** (hard cap < 10 MB), smooth on a modern laptop.

**Concrete targets (to confirm/calibrate, PRD Req 8):**

| Metric | Target |
|---|---|
| Top-1 accuracy, signer-held-out val (dataset distribution) | ≥ 80% |
| Top-1 accuracy, self-collected webcam eval (deployment distribution) | ≥ 60% (honest stretch; see Risk 2) |
| False-pass rate after thresholding | ≤ 5% per class |
| Inference latency (clip → result), laptop | < 1s |

The deployment-set number is the one that actually predicts learner experience. Treat the gap between the two as the headline result, not a footnote.

---

## 2. Risks Driving This Plan

These three (from the audit) shape every choice below:

1. **Data sufficiency.** ASL Citizen is ~83k videos across ~2,731 signs ≈ **~30 videos/sign**. After a signer-held-out split, ~20 train clips/class — thin for a from-scratch video model. → Vocab is selected *by example count*, augmentation is aggressive, and the temporal head starts simple.
2. **Domain shift (dataset → laptop webcam).** Signer-held-out accuracy measures generalization *within ASL Citizen's distribution*, not to a college webcam. → We **collect a small self-recorded webcam set** for honest eval and threshold calibration.
3. **No hand localization.** Pretrained detectors are banned; full-frame at 96px makes hands tiny. → We baseline on full-frame and run a **classical (non-learned) ROI crop** experiment (allowed because it's not a trained model).

---

## 3. Data Processing Pipeline

### 3.1 Source & provenance
- **Primary dataset: ASL Citizen** (isolated signs, real signers, varied environments). Record dataset version, download date, and **confirm the license/data-use terms permit this pilot** (note in the validation report — PRD Req 15 wants provenance + evidence no pretrained models were used).
- Curating a subset satisfies PRD Req 6 ("curating *or* collecting"). The self-collected webcam set (§4) strengthens the "engineer-owned dataset" story.

### 3.2 Vocabulary selection (Phase 0 — do this *first*)
1. **Data audit:** count videos per sign and signers per sign across ASL Citizen.
2. Build a candidate list of **ASL-1-appropriate beginner signs** (common nouns/verbs/everyday vocab — align to a real ASL 1 word list).
3. Intersect candidates with the audit; **require a minimum per class** (target ≥ 25 videos and ≥ 4 distinct signers, tune to reality).
4. Drop or quarantine signs that are visually confusable unless separable; keep a documented exclusion list.
5. If fewer than 100 signs clear the bar, **ship fewer (but ≥ 75) and document it** rather than padding with data-starved classes.
6. Freeze the final vocabulary as a **manifest** (sign label → list of source video IDs → hint metadata).

### 3.3 Clip standardization
- Decode each video and **uniformly temporally resample to a fixed 16 frames spanning the sign**, regardless of the clip's wall-clock duration. This normalizes tempo — a slow beginner and a fluent dataset signer map to the same input shape — and directly mitigates the tempo dimension of domain shift (Risk 2).
- The app's live **~3s capture window** is resampled with the **identical** procedure. **Hard invariant:** training preprocessing and live capture must use the same temporal normalization, frame count, resize, and channel normalization — any divergence silently degrades live accuracy.
- Resize frames to **96×96 or 112×112 RGB**; normalize (per-channel mean/std from the training split).
- Browser inference cost is fixed by the frame count (not the window seconds), so a generous window is free at inference time.
- Cache preprocessed tensors to disk to keep training iterations fast.

### 3.4 Localization (experiment, Risk 3)
- **Baseline:** full upper-body frame, resized. Cheap; lets gross motion/location cues carry signs.
- **Experiment:** classical ROI crop — skin-color segmentation and/or frame-difference motion mask → crop to the active region → resize. **No learned weights**, so it's allowed under Req 7. Compare val accuracy; adopt if it helps handshape-dependent signs.

### 3.5 Augmentation
- Photometric: brightness/contrast/gamma jitter, mild blur, small Gaussian noise (bridges webcam variability).
- Geometric: small crops/shifts/scale, slight rotation.
- Temporal: frame-sampling offset jitter, mild speed variation (resample frame count).
- **Horizontal flip: OFF by default** — flipping can change a sign's meaning/handedness. Only enable per-sign if verified meaning-preserving.

### 3.6 Splits
- **Signer-held-out** train/val/test (no signer appears in two splits) — honest generalization to new signers.
- Plus the **self-collected webcam eval set (§4)** as a separate, deployment-distribution test set.
- Persist split assignments in the manifest.

### 3.7 Hint metadata
- For each sign, tag the **handshape, movement, location, orientation, timing, framing** descriptors and author a short rule-based hint. This feeds the app's hint engine (PRD Req 10); the model itself only supplies predicted class + competitors + margin.

---

## 4. Self-Collected Webcam Set (Risk 2 de-risk)

- Record **yourself (+ 1–2 others if at all possible)** performing each vocab sign through the **actual app capture path** (~3s window, same resampling) under the documented conditions (lighting, distance, framing).
- Volume: ~5–10 clips/sign is enough for evaluation + calibration.
- **Usage:** hold out a portion strictly for **eval + threshold calibration** (reported as the deployment number). Optionally mix a slice into *training* to bridge the domain gap — but never train and evaluate on the same clips.
- This set is also the empirical basis for the "Supported camera and lighting conditions" documentation (Req 8).

---

## 5. Model Selection

**Frame encoder:** shared **tiny depthwise-separable 2D CNN**, trained from scratch → per-frame embeddings (dim **128–192**). Outputs embeddings, not per-frame hard labels.

**Temporal head — decide empirically, start simple:**
1. **Baseline: mean / attention pooling** over frame embeddings → classifier. Fewest parameters, most robust at ~20 clips/class.
2. **Then try: 1–2 layer Transformer encoder** (2–4 heads, MLP 256–384) + positional encoding, as in `architecture.md`. Adopt only if it beats the pooling baseline on signer-held-out val (audit flag: Transformers overfit on tiny data).
3. (Optional) tiny GRU as a third comparison.

**Why this shape:** preserves temporal info (vs. hard frame voting, which discards it) and stays cheap/exportable (vs. a 3D CNN, which is heavier and harder to train from scratch on a small budget).

**Output path:** logits → softmax; expose top-k classes + top-1/top-2 margin for thresholding and hint selection.

**Size budget:** target **2–6 MB** quantized, **< 10 MB** hard cap.

**Framework / export:** pick one and verify the export path early —
- **PyTorch → ONNX → ONNX Runtime Web** (recommended if comfortable in PyTorch), or
- **TF/Keras → TensorFlow.js**.
Validate a tiny model round-trips and runs in-browser **before** investing in the full train (de-risks export surprises).

---

## 6. Training Plan (phased, ~1–2 weeks solo)

| Phase | Work | Est. |
|---|---|---|
| 0 | Data audit + vocabulary freeze (§3.2); confirm dataset license | 0.5–1 d |
| 1 | Preprocessing pipeline + tensor caching; **export round-trip smoke test** (§5) | 1–2 d |
| 2 | Baseline model (CNN + pooling head), train from scratch, get signer-held-out number | 1–2 d |
| 3 | Collect webcam eval set (§4); measure deployment gap; iterate: augmentation, classical-crop experiment, try Transformer head | 2–3 d |
| 4 | Per-class threshold calibration; false-pass/false-fail tuning | ~1 d |
| 5 | Export + quantize + in-browser latency check | ~1 d |
| 6 | Validation report + versioning bundle | ~1 d |

**Training details:**
- Loss: cross-entropy with **label smoothing**; class weighting if counts are uneven.
- Optimizer: AdamW; LR warmup + cosine decay; **early stopping on signer-held-out val**.
- Regularize hard (small data): dropout, weight decay, the augmentation above.
- Reproducibility: fixed seeds; every run driven by a checked-in config file; log train/val curves.

---

## 7. Pass/Fail Thresholding & Calibration

- **Rule:** `pass` iff `argmax == prompted_class` **and** `P(prompted) ≥ class_threshold` **and** `P(top1) − P(top2) ≥ class_margin`; else `fail`.
- **Per-class thresholds** calibrated on the val + webcam sets to hit the target **false-pass rate (≤ 5%)**.
- Apply **temperature scaling** so probabilities are calibrated before thresholding.
- **Out-of-vocab / "no sign":** handle the learner doing nothing or signing something off-list — either a low-confidence reject path or an explicit background/"none" class. Decide in Phase 4.
- **Live smoothing:** EMA over recent windows is an *inference-time* aid only; it does not replace the temporal head.

---

## 8. Validation & Reporting (PRD Req 8 & 15)

Report must document:
- Final vocabulary list + videos per class; train/val/test split method and **signer-overlap policy**.
- **Two accuracy numbers:** signer-held-out (dataset) vs. self-collected webcam (deployment) — and the gap.
- Per-class accuracy; **confusion matrix** for visually similar signs.
- Pass/fail thresholding method; **false-pass and false-fail rates** per class.
- Supported **camera/lighting/distance/framing** conditions (from §4).
- **Known limitations & failure cases** (e.g., handshape-confusable signs, off-distribution webcams).
- **Evidence no pretrained models were used:** from-scratch init, training logs/curves, dataset provenance, dependency list (frameworks/libraries only).

---

## 9. Versioning Strategy

Each model version bundles, together, and is content-hashed:
- dataset subset **manifest** (video IDs + splits),
- preprocessing config,
- training config,
- model weights (+ quantized export),
- validation report.

Use semantic versions (e.g., `v0.1`, `v0.2`). Keep a simple `models/` registry; the browser app pins a specific version.

---

## 10. Open Decisions (need your input)

1. **Framework:** PyTorch→ONNX Runtime Web, or TF/Keras→TF.js? (affects everything downstream)
2. **Webcam set helpers:** solo-only, or can you recruit 1–2 others? (changes how strongly we can claim generalization)
3. **Vocab count** if data-limited: hold at 75 with strong per-class data, or stretch to 100 with some thin classes?
4. **Flip augmentation:** confirm we keep it off unless per-sign verified.

---

## 11. Risks & Mitigations (summary)

| Risk | Mitigation |
|---|---|
| Too few clips/class → overfit | Select vocab by count; aggressive augmentation; simple temporal head first |
| Dataset→webcam domain shift | Self-collected webcam eval + calibration; report the gap; optional domain mixing |
| Hands too small at low res | Classical ROI crop experiment (non-learned, allowed) |
| Transformer overfits tiny data | Pooling baseline first; adopt Transformer only if it wins on val |
| Browser export surprises | Round-trip export smoke test in Phase 1, before full training |
| False passes erode trust | Conservative per-class thresholds + margin + temperature scaling; tune FPR ≤ 5% |
