# ASL Learning — from-scratch sign recognizer + practice app

A browser-based app that teaches **75 beginner ASL signs**: you sign into your webcam,
a **from-scratch** computer-vision model recognizes the sign, and you get pass/fail
feedback with hints. Built for a non-commercial research/educational pilot on the
[ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/) dataset.

**Live demo:** https://willowy-cactus-5db06d.netlify.app/

## Result

A from-scratch CNN (no pretrained weights, no external backbones or landmark detectors),
evaluated **signer-held-out** (no signer appears in two splits):

| Metric (held-out test) | Value |
|---|---|
| top-1 accuracy | **73.3%** |
| top-3 accuracy | **85.7%** |
| top-5 accuracy | 90.2% |
| signs never recognized | 1 / 75 |
| ONNX size (quantized) | ~0.5 MB (browser-deployable, <10 MB cap) |

The accuracy ladder and the full story are in **`model/notebooks/model_story.ipynb`**:
baseline 18% → motion-ROI crop 29% → 500-class pretrain 43% → stacked 46% →
**1500-class pretrain 73%**. The decisive lever was *data* (scaling the encoder's
self-supervised-style pretraining corpus), not architecture.

## How it works

- **Model** (`model/`): a depthwise-separable 2D CNN frame encoder + attention pooling
  over 16 frames (112×112), trained from scratch on ASL Citizen. A classical
  **motion-ROI crop** (inter-frame motion → bounding box, no learned model) frames the
  signer. Exported to ONNX.
- **App** (`app/`): React + Vite + Supabase. Captures a ~3 s / 16-frame clip, replicates
  the motion-ROI crop in-browser, runs the ONNX model via ONNX Runtime Web, and passes a
  sign if it lands in the model's **top-3**. Conservative-gating infra is retained for a
  stricter production bar.

## Repository layout

```
model/        from-scratch model: data audit, preprocessing, training, ONNX export
              -> notebooks/model_story.ipynb  (the showcase narrative)
              -> MODEL_WORKSTREAM.md           (model <-> app interface contract)
app/          React/Vite/Supabase practice app (see app/README.md)
model-v2/     separate "Constellation" landmark-primary redesign (independent track)
docs/         design specs, ADRs, privacy
implementation.md   living status tracker across both workstreams
architecture.md     system architecture
```

## Quickstart

**App (dev):**
```bash
cd app
cp .env.example .env.local   # fill in Supabase URL + anon key
npm install
npm run seed                 # seed the signs table from the model manifest
npm run dev                  # http://localhost:5173
```
Requires a Supabase project with **Anonymous sign-ins** enabled. See `app/README.md`
and `app/SUPABASE_SETUP.md`.

**Model (training):** see `model/README.md`. Training runs on Google Colab (T4);
the local M4 handles preprocessing and ONNX export.

## Constraints (self-imposed by the brief)

- **From scratch** — every weight trained by us; no pretrained models/backbones/landmark
  detectors. (OpenCV ROI crop and optical flow are classical, no learned weights.)
- **Browser inference** — ONNX Runtime Web, model <10 MB, fully client-side (no raw frame
  upload; see `docs/privacy.md`).
- **Honest evaluation** — signer-held-out splits; conservative pass/fail.
