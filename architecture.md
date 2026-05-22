# ASL Learning — System Architecture

Browser-based ASL-1 vocabulary practice pilot. This document is the **overall system architecture** (app + how the vision model integrates). It is owned by the **system/app workstream**.

- Product/design detail → `docs/superpowers/specs/2026-05-21-asl-learning-pilot-design.md`
- Vision model internals (dataset, training, validation) → `vision-model-plan.md`
- **Model↔app interface contract (source of truth)** → `model/MODEL_WORKSTREAM.md §B` (model-agent-owned)

---

## 1. Repository Layout (monorepo)

```
ASL Learning/
  app/      # React + Vite SPA (system/app workstream)
  model/    # PyTorch training + ONNX export (model workstream)
  models/   # shared artifact registry: asl-<version>.onnx + meta.json (app pins a version)
  docs/     # specs
```

## 2. Components

**Client SPA (`app/`, React + Vite):**
- **Auth** — Supabase email login; accounts persist across devices.
- **Onboarding / camera setup** — permission request, brightness + framing check, persistent body-outline guide; handles denied/unavailable/unsupported (Req 4).
- **Practice screen** — floating-overlay layout: top-center prompt pill, centered camera stage, slim left control rail (Start / Retry); feedback (result + hint + Show example) overlay.
- **Capture pipeline** — `getUserMedia` → countdown → **~3s window** → **uniform resample to 16 frames @ 112×112** → identical normalization to training.
- **Inference** — **ONNX Runtime Web** (WASM/WebGPU); loads `models/asl-<ver>.onnx` + `meta.json` once, cached. Logits → softmax.
- **Pass/fail + hints** — app-side decision using per-class thresholds from `meta.json`; rule-based hints from per-sign metadata.
- **Session orchestrator** — builds mastery-prioritized deck, runs the per-word loop, writes attempts.

**Supabase (BaaS):**
- Auth; Postgres (`signs`, `sessions`, `attempts`, `sign_mastery`) with RLS; mastery trigger; static hosting (or Vercel/Netlify).

**Model artifact:** versioned `models/asl-<ver>.onnx` + `meta.json` (labels, input spec, per-class thresholds, hint metadata). Produced by the model workstream; consumed read-only by the app.

## 3. Data Flow & Privacy Invariant

```
camera → browser capture → in-browser ONNX inference → pass/fail + hint
                                                  │
                          outcome metadata only ──┘──► Supabase Postgres
```

**Camera frames + inference never leave the browser** (Req 5 & 13) — there is no upload code path. Only `attempts` metadata (result, confidence, margin, ids, timestamps) is persisted. See spec §7.

## 4. Data Model

See spec §5. Tables: `signs` (seeded from model `meta.json` — not hand-authored), `sessions`, `attempts` (metadata only), `sign_mastery` (rollup). Mastery = 2 first-try passes across 2 distinct sessions, maintained by a Postgres trigger. RLS per `auth.uid()`.

## 5. Key Interaction Rules

- **Capture:** fixed ~3s countdown window, resampled to the model input (decouples sign tempo).
- **Retry:** up to **2 retries (3 attempts total)** per encounter; hint after each fail; reference clip available from first fail and shown after the final fail; skip anytime. Only the **first attempt** counts toward mastery.
- **Deck:** mastery-prioritized ~10 words/session.

## 6. Cross-Stream Alignments — RESOLVED 2026-05-21

Truth for the interface is `model/MODEL_WORKSTREAM.md §B`.

1. **Clip input shape — FROZEN: 16 frames @ 112×112, uniform resample of the ~3s window to 16.** (App A5 uses 16, not the earlier 12 proposal.)
2. **Reference clips — re-recorded** (NOT ASL Citizen — MSR license + personal data); produced by the model/dataset side, served from `models/refs/`.
3. **Scope** — non-commercial / research-educational pilot (ASL Citizen license).

## 7. Vision Model (summary)

From-scratch (no pretrained) tiny depthwise-separable 2D CNN frame encoder → temporal head (pooling baseline, Transformer if it wins) → logits over the frozen 75-sign vocab; quantized ONNX 2–6 MB. Full plan: `vision-model-plan.md`. Trained on a curated ASL Citizen subset + a self-collected webcam eval set for honest deployment numbers.
