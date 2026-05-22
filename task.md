# ASL Learning — Task Assignment & Responsibilities

Two parallel workstreams. This file defines **who owns what**, the **shared interface** they must agree on, and the **dependency order** so the streams don't collide. Status lives in `implementation.md`.

- **Stream M — Model & Dataset** — owner: *model-training agent*. Detailed log/contract: `model/MODEL_WORKSTREAM.md`. Plan: `vision-model-plan.md`.
- **Stream A — Application & System** — owner: *system/app agent*. Spec: `docs/superpowers/specs/2026-05-21-asl-learning-pilot-design.md`. Arch: `architecture.md`.

---

## 0. Shared Interface Contract (FREEZE BEFORE DEEP WORK)

Source of truth for the **interface** (shapes/artifacts): **`model/MODEL_WORKSTREAM.md §B`**. Both streams depend on these; changes get coordinated there before anyone hard-codes against them.

**Validation & calibration strategy is governed by `docs/adr/0001-no-live-signer-validation-strategy.md`** (Accepted 2026-05-21, **revised 2026-05-22**). The ADR is the binding authority and **supersedes any webcam-eval decision** in `MODEL_WORKSTREAM.md §D` or `vision-model-plan.md §4`. ⚠️ `MODEL_WORKSTREAM.md §D.2` (model-agent-owned) still records the old "user + 1–2 others webcam eval set" — needs the model agent to reconcile it with ADR 0001.

| Artifact | Produced by | Consumed by | Status |
|---|---|---|---|
| `meta.json` (labels, input spec, thresholds, hints) | M | A | shape agreed |
| `models/asl-<ver>.onnx` (quantized, 2–6 MB) | M | A | format agreed |
| Vocabulary (75 signs) | M (`proposed_vocab.json`) | A (seeds `signs`) | frozen (proposed) |
| Clip input shape (16 frames @112) | M | A | **FROZEN 2026-05-21** |
| Per-class thresholds | M (calibration) | A (pass/fail) | format agreed |
| Hint metadata per sign | M | A (hint engine) | agreed |
| Reference clip per sign (re-recorded) | M/dataset | A (failure UI) | resolved 2026-05-21 |

### Alignments — RESOLVED 2026-05-21
1. **Clip input shape — FROZEN: 16 frames @ 112×112.** App captures ~3s and **uniformly resamples to 16 frames** (`meta.json.input` = {frames:16, window_seconds:3.0, size:112}). **App A5 uses 16, not the earlier 12 proposal.**
2. **Reference clips — re-recorded.** Cannot redistribute ASL Citizen video (MSR license + personal data). Model/dataset side records one short clip per sign → `models/refs/<gloss>.mp4`, served as a static asset.
3. **Scope — non-commercial / research-educational pilot** (ASL Citizen license).

---

## Stream M — Model & Dataset (owner: model agent)

High-level only; Stream A does **not** drive these. Detail in `vision-model-plan.md` / `MODEL_WORKSTREAM.md`.

> **v2 (Constellation):** a separate landmark-primary redesign is specced in `docs/superpowers/specs/2026-05-22-constellation-v2-model-design.md` (workspace `model-v2/`, plans under `docs/superpowers/plans/`). M2–M7 below describe v1; v2 plans supersede them as they land.

- M1. Data audit + **freeze 75-sign vocab** → `proposed_vocab.json`. *(done — proposed)*
- M2. Preprocessing pipeline (resample to fixed frames, resize, normalize); cache subset tensors.
- M3. Train from-scratch CNN+temporal model on Colab; signer-held-out validation.
- M4. **Validation per ADR 0001** (revised 2026-05-22): headline = ASL Citizen signer-held-out + WLASL cross-dataset; threshold calibration on the ASL Citizen val split. **No "user + 1–2 others" webcam eval set.** The builder runs a **documented live-tester check** under a predeclared protocol (signs/attempts/conditions fixed up front), reported **separately** as a sanity check — never the headline, never both tuned-against and reported.
- M5. **Threshold calibration** (per-class, false-pass-averse) → into `meta.json`.
- M6. Author **hint metadata** + **reference clips** per sign.
- M7. Export + quantize **ONNX**; publish `models/asl-<ver>.onnx` + `meta.json`; validation report.

## Stream A — Application & System (owner: app agent)

- A1. **Scaffold** `app/` (React + Vite + TypeScript); ONNX Runtime Web dependency.
- A2. **Supabase project**: schema (`signs`, `sessions`, `attempts`, `sign_mastery`), RLS, mastery trigger, auth. Seed `signs` from `meta.json`.
- A3. **Auth + routing**: login, progress dashboard shell.
- A4. **Camera/onboarding**: permission, brightness/framing check, framing-guide overlay, denied/unavailable handling.
- A5. **Capture pipeline**: countdown → ~3s window → **uniform resample to 16 frames @ 112×112** → normalization (mirror M2 exactly).
- A6. **Inference integration**: load ONNX + `meta.json`, run, softmax. *(stub with a fake model until M7)*
- A7. **Pass/fail + hint engine**: thresholds + margin from `meta.json`; rule-based hints.
- A8. **Practice screen**: floating-overlay layout; states (ready/countdown/recording/result/fail); 2-retry loop; reference-clip viewer; skip.
- A9. **Session orchestrator**: mastery-prioritized deck; per-word loop; write attempts.
- A10. **Progress dashboard**: mastery overview + recent history.
- A11. **Privacy doc** (Req 15) + offline/error handling polish.

## Dependency Order (lets A run before M finishes)

- **A can build A1–A5, A8–A10 against the frozen contract + a STUB model immediately** — it does not need the trained model.
- **A6/A7 finalize after M7** (real ONNX) and M5 (thresholds), but can be developed against a stub `meta.json` matching the agreed shape.
- **A2 `signs` seeding** needs the frozen vocab (M1 done) — can proceed now using `proposed_vocab.json`; re-seed if vocab changes.
- **Critical path:** Alignment #1 (clip shape) **resolved → 16 frames**; A5 and M2 unblocked.
- Integration milestone: M7 artifact dropped into `models/` → A swaps stub for real model → end-to-end test.

## Coordination Rules
- Interface changes go through `model/MODEL_WORKSTREAM.md §B` (model-owned) + this file's Alignment list.
- App reads vocab/thresholds/hints from `meta.json` at runtime — never hard-codes them.
- `model/data/` and `models/*.onnx` weights are gitignored; manifests/configs/reports are committed.
