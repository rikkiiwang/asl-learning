# Model Workstream — Progress Log & App Contract

This file is owned by the **model-training agent/window**. The system-design/app
agent should READ this for the model↔app interface contract (§B) and avoid editing
§A; coordinate changes to §B here before depending on them.

Last updated: 2026-05-21 (coordination edit 2026-05-22 by the system/app window — §D.2 only)

> **Cross-stream banner (2026-05-22):** Validation/calibration strategy is now
> governed by `docs/adr/0001-no-live-signer-validation-strategy.md` (Accepted,
> revised 2026-05-22) — it supersedes the older webcam-eval decision in §D.2
> below. Separately, a v2 model redesign ("Constellation", landmark-primary) is
> specced in `docs/superpowers/specs/2026-05-22-constellation-v2-model-design.md`
> and built in `model-v2/` (does not touch this `model/` workspace).

---

## A. Progress Log (what the model side has done)

- **2026-05-20** — Read PRD (`ASL learning.pdf`, 15 reqs), `architecture.md`, `vision-model-plan.md`. Confirmed scope: 75-100 isolated ASL-1 signs, browser inference, from-scratch model, conservative pass/fail.
- **2026-05-20** — Toolchain check: PyTorch 2.7 + **MPS** available on Apple M4 / 24 GB. No TensorFlow installed. `cv2` 4.12 present, `onnx` not yet installed. → Strongly favors **PyTorch → ONNX → ONNX Runtime Web**.
- **2026-05-20** — Dataset was a truncated Safari download (24.0 GB of 45.9 GB). Moved partial zip to `model/data/ASL_Citizen.zip` and **resumed** via `curl -C-` from Microsoft source. (See `model/data/download.log`.)
- Workspace location for all model code/data/artifacts: **`ASL Learning/model/`**.
- **2026-05-21** — Dataset download completed + verified (45.9 GB, 83,399 videos). Extracted split CSVs. Ran data audit → **proposed balanced 75-sign vocab** across 10 categories (`artifacts/manifest/proposed_vocab.json`). Plan approved (`~/.claude/plans/what-is-our-traing-composed-thacker.md`).
- **2026-05-21** — Cross-stream freezes with app agent: **clip = 16 frames / ~3 s / resample @112** (app's 12 → 16); **vocab = 75 signs**. Reference clips: **cannot ship ASL Citizen video** (MSR license + Personal Data) → re-record. Scope: **non-commercial research/educational pilot**.

## B. Model ↔ App Interface Contract (CONFLICT SURFACE — read this)

These are the points where the two workstreams must agree. Proposed defaults below;
flagged items still need confirmation.

| Concern | Proposed contract | Status |
|---|---|---|
| **Runtime** | ONNX Runtime Web (WASM/WebGPU), model loaded client-side | proposed |
| **Model artifact** | `models/asl-<version>.onnx`, quantized, 2-6 MB (<10 MB cap) + sidecar `meta.json` | proposed |
| **Clip input** | **16 frames** spanning a **~3.0 s window** (≈5.3 fps), uniform temporal **resampling** to exactly 16 frames, **112×112 RGB**, per-channel normalized (mean/std in `meta.json`) | **FROZEN 2026-05-21** |
| **Capture window** | App's countdown-record produces the SAME clip shape: fixed ~3 s window → resample to **16 frames** @112×112 → identical normalization. **CHANGE: app proposed 12 → frozen at 16** (affects app A5). | **FROZEN 2026-05-21** |
| **Motion-ROI crop** | The shipped model is trained on a **classical motion-ROI crop** (per-pixel inter-frame abs-diff → robust bounding box `[5,95]` pct → squared + 20% margin, clamped to ≥0.55·min(h,w), center-crop fallback). The app **MUST replicate this crop in-browser** on the 16 captured frames *before* resize/normalize, or accuracy will degrade. No model/landmark detector needed — pure pixel diff, ports to JS/WASM. Ref impl: `motion_roi_box()` in `src/asl/preprocess.py`. | **NEW — pending app confirmation (2026-05-23)** |
| **Model output** | logits over the frozen vocab → softmax; expose top-k classes, P(prompted), top1-top2 margin | proposed |
| **Pass/fail** | done by app using thresholds in `meta.json`: pass iff argmax==prompt AND P(prompt)≥class_threshold AND (P1-P2)≥class_margin | proposed |
| **Vocabulary** | **frozen 75 labels** in `meta.json` (label → index → hint metadata); list = `model/artifacts/manifest/proposed_vocab.json`. App's prompts MUST come from this list. | **FROZEN 2026-05-21 (75 signs)** |
| **Hint metadata** | per-sign handshape/movement/location/orientation/timing/framing tags shipped in `meta.json`; app's hint engine consumes them | proposed |
| **Privacy** | inference fully local; no raw frame upload (PRD Req 13) | fixed by PRD |

### `meta.json` shape (proposed)
```json
{
  "version": "v0.1",
  "input": {"frames": 16, "window_seconds": 3.0, "fps": 5.3, "size": 112,
            "resample": "uniform", "layout": "NFCHW", "mean": [..], "std": [..]},
  "labels": ["BOOK", "EAT", ...],
  "thresholds": {"BOOK": {"t": 0.7, "margin": 0.2}, ...},
  "hints": {"BOOK": {"handshape": "...", "movement": "..."}, ...}
}
```

## C. Where conflicts could arise (flags for the app agent)

1. **Clip shape must match.** If the app fixes a capture duration / frame count before the model freezes its input spec, retraining may be needed. Treat §B "Clip input" as the source of truth.
2. **Vocabulary is data-driven.** Final 75-100 signs depend on the data audit (per-class counts), so the app should not hardcode a word list until the manifest is frozen (Phase 0).
3. **Threshold/margin live in `meta.json`**, not in app code — the app reads them so calibration updates don't require an app change.
4. **Model files** belong in a shared `models/` registry the app pins by version; the model agent writes there, the app reads.
5. **Motion-ROI crop is now a preprocessing step, not just a model.** The best v1 checkpoint (stacked ROI+pretrain, 46% test) expects ROI-cropped input. If the app cannot replicate the crop in-browser, the model agent ships a center-crop fallback checkpoint (43% test) instead — but the ROI crop is classical and cheap, so replicating it is preferred. Decide before ONNX export freezes.

## D. Decisions (resolved 2026-05-20)
1. **Framework: PyTorch → ONNX → ONNX Runtime Web.** ✅
2. **Validation strategy → SUPERSEDED 2026-05-22 by ADR 0001** (binding authority).
   The earlier "webcam eval set: user + 1-2 others" is dropped. Headline evidence =
   ASL Citizen signer-held-out + WLASL cross-dataset; threshold calibration on the
   ASL Citizen val split. The builder runs a **documented live-tester check** under
   a predeclared protocol, reported **separately** as a sanity check — never the
   headline, never both tuned-against and reported.
3. **Vocab: hold at 75 with strong per-class data** (all clear ≥25 videos & ≥4 signers). Don't stretch to 100 with thin classes.
4. **Layout: monorepo under `ASL Learning/`.** App agent builds in this folder (e.g. `ASL Learning/app`). Model lives in `ASL Learning/model`. Exported artifacts published to a shared `ASL Learning/models/` registry the app pins by version.
5. **Clip shape: 16 frames / ~3 s window / uniform resample / 112×112** (Alignment #1 resolved; app A5 must use 16, not 12). ✅
6. **License scope: non-commercial / controlled / research-educational pilot** (ASL Citizen MSR license). Reference clips (Alignment #2) must be **re-recorded**, not taken from ASL Citizen. WLASL added as Tier-2 cross-dataset eval, **internal-only**. ✅

## E. Compute & data flow (Colab training)
- **Training runs on Google Colab (GPU).** Dataset stored in Google Drive + locally.
- Efficiency decision: full dataset is ~46 GB / ~83k videos but we only need the ~75-sign subset (~2k videos). Plan:
  1. **Local (M4):** extract zip → data audit on metadata CSVs → freeze vocab → preprocess ONLY the subset videos into cached frame tensors.
  2. Upload the **compact preprocessed subset** (tensors + manifest, few GB) to Drive — NOT the full 46 GB.
  3. **Colab:** mount Drive, train from the compact cache, export ONNX.
- Code is a plain Python package (`model/src`) + a thin Colab notebook wrapper, so it runs identically locally or on Colab. No notebook-only logic.
- `model/data/` (raw dataset + tensors) is git-ignored; never committed.
