# Constellation — ASL Recognizer v2 Design

**Date:** 2026-05-22
**Codename:** Constellation
**Workspace:** `model-v2/` (separate from `model/`, which is owned by another agent and stays untouched)
**Status:** design approved; ready for implementation plan
**Supersedes (architecturally, not on disk):** the single-stream recognizer in `model/src/asl/model.py`
**Binding constraint:** `docs/adr/0001-no-live-signer-validation-strategy.md` (Accepted 2026-05-21, revised 2026-05-22) governs all calibration/validation claims here — headline evidence is held-out datasets (ASL Citizen signer-held-out + WLASL); calibration runs on the ASL Citizen val split; the builder is a **documented live tester** under a predeclared protocol, reported **separately** as a sanity check (never tuned-against-and-reported, never the headline, never proof the system works for real users).

---

## 1. Motivation

### 1.1 What v1 is
The current recognizer (`model/`) is **one** shared depthwise-separable 2D CNN
run over each of 16 frames @112×112 → per-frame embeddings → a temporal head
(attention pooling, optional 1–2 layer Transformer) → 75-way logits. From
scratch, ~2–6 MB quantized. A pretrain→finetune path already exists.

### 1.2 The problem (measured)
`model/artifacts/checkpoints/baseline/history.json`: best **val top-1 = 16.4%**,
**test = 18.0%** on 75 classes. Train loss falls to ~1.75 while val flatlines at
~16% by epoch 20 — simultaneously **overfitting** (train improves, val stuck)
and a **weak fit** (train loss never nears zero).

### 1.3 Root cause
This is **Risk 3** from `vision-model-plan.md` realized: a single CNN sees the
whole 112×112 frame, where the hands are ~10–15% of the pixels. With ~20
clips/class over 75 classes, the cheapest signal is signer/background/clothing —
which does not transfer to held-out signers (hence 16% val). The failure was
**not** too little capacity; it was **capacity spent on the wrong, appearance-
correlated pixels**, plus the model having to learn the "location" parameter
from data it doesn't have.

### 1.4 v2 thesis
Recognize the sign from an **appearance-invariant geometric representation** of
the hands and their position relative to the head — keypoints and boxes, not raw
pixels. A keypoint skeleton has no background, skin tone, clothing, or face, so
the model **cannot** latch onto the cues that caused v1 to overfit. This is why
sign-language-recognition research leans heavily on pose/hand keypoints. A small
appearance CNN stays as a secondary cue to cover what keypoints miss.

---

## 2. Approach: detect → landmark → recognize (landmark-primary)

Constellation adopts the **method** (not the pretrained weights) of MediaPipe
Hands: a from-scratch detector localizes parts, a from-scratch landmark model
turns each hand into keypoints, and recognition runs **primarily on the
keypoint/box geometry**. Everything is trained by us — zero pretrained weights.

```
16 frames ─▶ [Stage-1 Detector]  per frame ─▶ hand×≤2 + head boxes
                                            │
                              crop each hand ▼
                       [Stage-1.5 Hand-Landmark model] ─▶ 21 keypoints per hand
                                            │
        ┌───────────────────────────────────┼──────────────────────────────┐
   POSE-GEOMETRY (primary)          small HAND-APPEARANCE (secondary)   (opt.) body context
   21×2 hand keypoints +            tiny CNN on hand crop @64–96²        low-res frame
   head anchor, head-normalized     → small embedding                   → small embedding
        │ → MLP/encoder                      │                               │
        └────────────────── per-frame fused embedding ─────────────────────┘
                                            │  (× 16 frames)
                              [Temporal head: attn-pool, opt. Transformer]
                                            │
                                  75-way logits → softmax / top-k / margin
```

### Why this addresses the 16% wall
1. **Appearance-invariant primary signal.** Keypoints + boxes carry no
   background/skin/clothing/face, so the dominant cue can no longer be the thing
   that overfit. Directly attacks the root cause.
2. **Handshape + location given as geometry, not learned from sparse pixels.**
   Keypoint configuration = handshape; keypoints normalized to the head anchor =
   location. Both handed to the model as numbers.
3. **Movement is the keypoint trajectory** through the temporal head — explicitly
   "the hand movement," the thing v1 failed to learn.
4. **Small appearance CNN** is a deliberately low-capacity secondary cue (contact,
   occlusion, handshapes the landmark model gets wrong) — sized to *not*
   reintroduce overfitting.

---

## 3. Components

### 3.1 Stage-1 detector (our own, from scratch)
- Tiny single-shot detector, YOLO/SSD-style, **no pretrained weights**.
- **BlazePalm trick:** square anchors for hands (rigid, easier).
- **Classes:** `hand`, `head` (head = position/scale reference only; never
  recognized as a face — see §3.4).
- Output: ≤2 hand boxes + 1 head box/frame. Input ~128². Size ~0.5–1 MB.
- **Data:** **100DOH** primary (frontal, varied), **EgoHands** optional; head
  boxes from a face/head-box source (finalize in plan).

### 3.2 Stage-1.5 hand-landmark model (our own, from scratch)
- Runs on each hand crop from Stage-1 → **21 2D keypoints** per hand
  (MediaPipe's keypoint count; **2D**, not 3D — simpler, sufficient for our
  geometry).
- From scratch; **no pretrained weights**.
- **Data:** a keypoint-labeled hand dataset — **FreiHAND** (~130k) candidate,
  possibly + COCO-WholeBody hands for in-the-wild frontal variety. *Domain-shift
  risk: lab/green-screen → webcam (see §7).*
- Size kept small; runs per detected hand.

### 3.3 Stage-2 recognizer — streams
- **Pose-geometry stream (PRIMARY).** Per frame: the 21×2 keypoints of each hand
  + head anchor (center + size). **Normalized**: translate so head center =
  origin, scale by head size → translation/scale invariant. Includes hand→head
  and hand→hand offsets, presence flags. **Hand slotting (stable, not by area):**
  associate boxes across the 16 frames into ≤2 tracks via nearest-center
  (greedy IoU/centroid) association, then assign each track to a **left/right
  slot by its x-position relative to the head anchor**, held constant for the
  whole clip. This avoids the fake motion that box-area ordering injects when
  hands swap size under occlusion/depth. One hand → its true slot filled, the
  other zero-padded + presence-flagged. *Fallback ablation:* a permutation-
  invariant two-hand encoder (max/mean over the two slots) if slotting proves
  unstable on held-out signers. → MLP/encoder. This is the main signal.
- **Hand-appearance stream (SECONDARY, small).** A *tiny* shared CNN on each hand
  crop @64–96² → small embedding. Deliberately low capacity to avoid
  reintroducing overfitting. Catches contact/occlusion/landmark errors.
- **Body/context stream (OPTIONAL).** Low-res @64² upper-body frame. Demoted to
  an **ablation experiment** — location is now carried by keypoint+head geometry,
  so this is kept only if it earns its place on signer-held-out val.
- **Fusion + temporal head:** concat per-frame → projection → temporal head
  (attention pooling baseline; 1–2 layer Transformer only if it wins on
  signer-held-out val) → 75 logits. **Output contract unchanged:** logits →
  softmax, top-k, P(prompted), top1–top2 margin.

### 3.4 No face-appearance recognition
The head box/anchor is a **spatial reference only**. For isolated beginner
vocabulary, facial non-manual markers carry grammar more than lexical identity; a
face CNN would mostly overfit to the signer.

### 3.5 Debug visualization (backend / test only — NOT user-facing)
Like the reference model the user saw: render, on test frames in the backend,
the **hand + head boxes**, the **21-point skeleton** per hand, and the
**geometry lines** (hand→head, hand→hand). Used to validate the detector +
landmark model are tracking correctly. **Never shown in the user app** (privacy +
UX; the app stays frames-in/logits-out).

### 3.6 Head-anchor robustness (load-bearing)
Because every geometry coordinate is normalized by the head box, the head anchor
must be stabilized, never trusted blind:
- **Per-clip smoothing.** The head barely moves across a ~3 s clip, so smooth the
  head box over the 16 frames (EMA or per-clip median of center + size) before
  using it to normalize. Removes per-frame jitter that would otherwise inject
  noise into the primary signal.
- **No-head fallback chain.** If the detector returns no head on a frame: use the
  last-good (or per-clip median) head box; if the head is missing for the whole
  clip, fall back to a frame-center anchor scaled by an estimated body scale
  (e.g. shoulder-width proxy from the frame), and **set a `head_present=0` flag**
  fed to the recognizer so it can down-weight geometry.
- **Gating metric.** Report **head detection-rate and box stability on real ASL
  frames**; if it falls below a threshold the geometry path is untrustworthy and
  the appearance stream must carry more weight (informs the §4 A→B decision).

### 3.7 Per-stage size budget (quantized)
Each component has its own cap so no single model (especially the landmark model)
can consume the whole budget. Caps are **int8-quantized** size; param counts
assume ≈1 byte/param at int8 (double for fp16). Enforced by a test in each
component's plan.

| Component | Param cap | Quantized size cap |
|---|---|---|
| Stage-1 detector | ≤ 2.0 M | ≤ 2 MB |
| Stage-1.5 hand-landmark | ≤ 3.0 M | ≤ 6 MB |
| Stage-2 recognizer (geometry MLP + fusion + temporal head) | ≤ 1.5 M | ≤ 3 MB |
| Hand-appearance CNN (secondary) | ≤ 1.0 M | ≤ 2 MB |
| **Total** | **≤ 7.5 M** | **target ≤ 13 MB, hard cap < 20 MB** |

If a component exceeds its cap, shrink it (width/depth) before borrowing from
another's budget; the total hard cap is the inviolable line.

---

## 4. Scope: build lean-first, geometry-first

The chosen center of gravity is **landmark-primary + small appearance**. Build
order, each gated on signer-held-out val (the §1.3 discipline — every added
stream must beat held-out signers or it's cut):

- **A — pose-geometry only (lean baseline).** Recognize from keypoints + box
  geometry alone, no appearance. Smallest, most appearance-invariant. **Must beat
  v1's 16%** before anything is added — and given it's appearance-invariant, it
  is the cleanest test of the thesis.
- **B — + small hand-appearance CNN (TARGET).** Add the secondary appearance cue.
- **C — optional experiments.** Body-context stream; richer hand-relationship
  encoder; motion stream. Adopt each only if it wins on held-out val.

---

## 5. Training & data flow (now three from-scratch models)

1. **Vocab:** reuse the frozen 75-sign vocab (`model/artifacts/manifest/`).
2. **Train detector** from scratch (100DOH/EgoHands); eval box quality.
3. **Train hand-landmark model** from scratch (FreiHAND/COCO-WholeBody hands);
   eval keypoint error (PCK) — including on a few real ASL frames.
4. **Cache (offline):** run detector + landmark over the ASL Citizen subset →
   precompute per-clip **keypoints + small hand crops + geometry** (same caching
   discipline as today's `clips.npz` / `frames.dat`). Detection+landmark cost
   paid once.
5. **Train recognizer** from scratch on the cache — **A first** (pose-geometry
   only), confirm > v1 16% on signer-held-out val; then **B** (add small
   appearance); ablate C only under the §4 gate.
6. **Calibration & evaluation — per ADR 0001 (revised 2026-05-22; supersedes
   `vision-model-plan.md` §4/§7):**
   - Per-class thresholds + temperature scaling calibrated on the **ASL Citizen
     signer-held-out val split only** (documented as dataset-distribution-
     calibrated).
   - Headline evidence = **signer-held-out test accuracy + WLASL cross-dataset
     accuracy** (the v1→v2 comparison). No ASL knowledge required to produce it.
   - **Documented live-tester check (the builder).** Live attempts recorded
     through the actual webcam capture path, after studying each target sign from
     authoritative references (Handspeak / Lifeprint-ASLU / Signing Savvy). To
     guard against p-hacking, **predeclare** the target signs, attempt counts, and
     lighting/framing conditions before these clips count as evidence (re-taking
     until pass is allowed for practice UX, not for reported quality).
   - **Separation rule:** live-tester clips are reported **separately** as a
     sanity check of the app path and the dataset→webcam gap — **never both tuned
     against and reported as an independent result**, and **never the headline**.
     The live loop is still **not** proof the system works for real learners (a
     single tester is not representative); a later student/signer pilot remains
     the only path to that.
7. **Export:** see §6.

Code is a plain Python package under `model-v2/` mirroring `model/`'s layout
(runs identically on M4/MPS and Colab GPU). `model-v2/data/` is git-ignored.

---

## 6. Export & app-contract impact

Capture window, 16-frame resampling, 112×112, normalization, vocab, and the
logits/threshold output all stay the same. Only *what runs inside inference*
changes — now three stages.

**Decision rule: prove first, then choose.** The export path is NOT pre-decided.
Before any real model is trained, **Phase 1 runs an export spike** with tiny
dummy 3-stage weights to learn whether the combined graph can round-trip in ONNX
Runtime Web. Only then do we commit to (b) or (a).

- **(b) Combined ONNX graph.** One graph: 16 raw frames in → detect →
  box-selection/NMS → crop (ROIAlign/`grid_sample`, 16 frames × ≤2 hands) →
  landmark → recognize → logits out. **App contract unchanged** (frames in,
  logits out). This is the **hardest export risk in the project** — NMS + dynamic
  per-frame cropping + `grid_sample` in ORT Web is exactly where it can break, so
  it must be *proven*, not assumed.
- **(a) Separate ONNX files.** Ship detector + landmark + recognizer; app
  orchestrates per frame in JS. More robust to export limits, but **changes §B**
  and pushes orchestration onto the app agent (coordinate the change).

The spike picks the path; if (a), coordinate the §B change with the app agent
**before** training, so M7 (export) cannot block after three models exist.

---

## 7. Constraints, budget, risks

| Concern | Decision / mitigation |
|---|---|
| **Req 7 "no pretrained / no landmark detector"** | **Reinterpreted, user-approved:** all three models trained **by us from scratch**, zero pretrained weights. We *do* now include a landmark model — document the reinterpretation + training logs + dataset provenance prominently in the validation report (Req 7/15). |
| **Scope / time** | This is a 3-model MediaPipe-style system — **materially larger** than the original ~1–2 week single-model plan. Plan must phase it and allow stopping at A or B. |
| **Error propagation** | detector → landmark → recognizer: errors compound. Mitigate by evaluating each stage on the **manually-labeled ASL-frame audit slice (§9 item 5)** *before* trusting it downstream; pose-geometry tolerates small jitter via head-normalization + temporal smoothing. |
| **Landmark domain shift** | FreiHAND/lab → ASL webcam. Pick the most frontal/in-the-wild source; consider self-labeling a slice of ASL Citizen (detector+landmark) to fine-tune on-domain. |
| **Size budget (per-stage, §3.7)** | **Hard cap < 20 MB total** quantized across all four components, **target ≤ 13 MB**. Per-component caps in §3.7 so the landmark model can't silently eat the whole budget. **Deliberate revision** from v1's `<10 MB` (single model) — a 3-model pipeline needs more; document the change like the Req-7 reinterpretation, and note the larger browser download. |
| **Latency <1 s / 16 frames** | Three models run per frame ×16. Keep all shallow; if latency bites, **detect+landmark on keyframes + track/interpolate** between frames. |
| **Overfitting (~20 clips/class)** | Landmark-primary is the core cure (appearance-invariant). A-first; held-out gate on every addition; small appearance CNN; aggressive augmentation (incl. keypoint jitter). |
| **Hand-ordering instability** | **Not** box-area (it swaps under occlusion/depth → fake motion in the *primary* signal). Use cross-frame track association → fixed left/right slot by x relative to head, held for the clip; permutation-invariant encoder as fallback ablation (§3.3). |
| **Head anchor is load-bearing** | Head center/size normalizes *all* geometry, so a missing/jittery head box corrupts every coordinate. Mitigations (§3.6): per-clip head smoothing (EMA/median — the head barely moves in 3 s), no-head fallback to last-good/median head then frame-center+body-scale, and report head detection-rate + stability as a gating metric on the **labeled ASL-frame audit slice (§9 item 5)**. |
| **2D vs 3D keypoints** | **2D** chosen — simpler model/data, sufficient for our normalized geometry. |
| **Export risk (hardest in project)** | **Prove first (§6):** Phase-1 export spike with dummy 3-stage weights decides combined-graph (b) vs separate-files (a) **before** any real training, so M7 can't block after all three models exist. |

---

## 8. Versioning & deliverables

- Ships as model **v0.2** (codename Constellation), bundled and content-hashed per
  `vision-model-plan.md §9`: detector + landmark + recognizer weights, dataset /
  crop / keypoint manifests, preprocessing + training configs, validation report,
  quantized export.
- Validation report adds, beyond v1: detector + landmark training
  logs/provenance + dataset licenses, the Req-7 reinterpretation note, the
  size-budget revision (§3.7), per-stage quality metrics (box mAP, head
  detection-rate/stability, keypoint PCK) measured on the **manually-labeled
  ASL-frame audit slice (§9 item 5)**, and the **v1→v2
  accuracy comparison** — headline = **signer-held-out test + WLASL cross-dataset**
  (per ADR 0001), plus a **separately-reported live-tester sanity check** under
  the predeclared protocol (not tuned against, not the headline).

---

## 9. Open items to resolve during planning
1. **Keypoint dataset + license** (FreiHAND / COCO-WholeBody hands / InterHand).
2. **Detector head-box label source** (face/head dataset vs. derived) — head is load-bearing for geometry (§3.6), so pick a source that detects heads reliably on frontal ASL frames.
3. **100DOH / EgoHands / keypoint-dataset license** confirmation.
4. **Export path** — the Phase-1 export spike (§6) decides combined-graph (b) vs separate-files (a) before training; not pre-decided.
5. **Manually-labeled ASL-frame audit slice — REQUIRED, not optional.** Hand-label a small slice of ASL Citizen frames (target ~50–100 frames spanning multiple signers/lighting) with hand + head **boxes** and 21 hand **keypoints**. This is the ground truth for every "on real ASL frames" metric (box mAP, head detection-rate + stability §3.6, keypoint PCK) — without it those metrics are unverifiable. The detector plan and landmark plan must each build/extend this slice and gate on it **before** the stage is trusted downstream. *Optional extension:* self-label a larger slice with the trained models to fine-tune on-domain.
6. Detector/landmark input **color** (RGB vs. grayscale) — size/robustness tradeoff.
