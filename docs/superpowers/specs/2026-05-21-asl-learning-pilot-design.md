# ASL Learning Pilot — Design Spec

**Date:** 2026-05-21
**Status:** Reviewed 2026-05-21; approved to begin Stream A (A1–A2 underway).
**Scope:** Browser-based ASL-1 vocabulary practice app (the *application/system* workstream). The vision-model workstream is specified in `vision-model-plan.md`; the interface between them is `model/MODEL_WORKSTREAM.md §B` (model-agent-owned, source of truth).

---

## 1. Product Scope

A browser app where a college ASL-1 learner is prompted with a beginner vocabulary word, signs it to their webcam, and gets an immediate **pass/fail** plus a **targeted hint** on failure, with progress saved across sessions. Controlled pilot, not a public product. Implements PRD requirements 1–15.

**Pedagogical role:** **practice / recall only** — assumes the learner already met the signs in class. No upfront teaching content; a reference clip is offered only as failure-recovery help.

**Vocabulary:** the **frozen 75-sign list** produced by the model workstream (`proposed_vocab.json` → `meta.json`). The app **does not hardcode** the word list — it reads labels (and hint metadata) from the model artifact's `meta.json`. 75 satisfies the PRD's 75–100 range with strong per-class data (≥25 videos, ≥4 signers each).

## 2. Target User

College ASL-1 beginners. New to ASL; need repeated practice, clear feedback, progress tracking. No assumptions about linguistics, model confidence, or CV limitations. No teacher/admin roles (out of scope).

## 3. Core Workflow

1. **Login** (Supabase auth) → **Progress dashboard** (mastery overview + recent history).
2. **Start session** → create a `sessions` row → **camera setup**: request permission, brightness/framing check, persistent body-outline framing guide. Handle denied/unavailable/unsupported camera clearly (Req 4).
3. **Build deck**: mastery-prioritized ~10 words (not-mastered first, least-recently-practiced, plus a couple mastered for review).
4. **Per word** — practice screen (floating-overlay layout): floating top-center prompt pill; centered camera as the stage; slim left control rail (Start / Retry).
   - Learner clicks **Start** → **3-2-1 countdown** → **~3s capture window** → frames resampled to the model input → in-browser inference → **pass/fail** (Req 9).
   - **Pass** → log attempt → advance.
   - **Fail** → targeted hint → **retry (up to 2 retries; 3 attempts total)**. A **"Show example"** reference clip (re-recorded, **not** from ASL Citizen — licensing) is available from the first fail. After the 3rd failed attempt, show the reference clip and advance. **Skip** available anytime.
   - Only the **first attempt** of an encounter counts toward mastery.
5. **End session** → set `ended_at` → summary (words mastered, accuracy). Word resurfaces in a later session's deck if not mastered.

## 4. Capture & Inference (app side of the contract)

- Capture window **~3s**; app **uniformly resamples to the model's frozen 16 frames at 112×112**, applies the **identical** normalization the model trained on (mean/std from `meta.json`).
- **Invariant:** the app's preprocessing MUST match the model's training preprocessing exactly. Drift here silently degrades live accuracy.
- Inference via **ONNX Runtime Web** (WASM/WebGPU); model + `meta.json` loaded once, cached. Output = logits → softmax.
- **Pass/fail done in the app** using per-class thresholds from `meta.json`: `pass` iff `argmax == prompted` AND `P(prompted) ≥ class_threshold` AND `P1 − P2 ≥ class_margin`. Conservative — avoids false passes (Req 9).
- **Hints** (Req 10): rule-based, selected from the prompted sign's hint metadata in `meta.json` (handshape / movement / location / orientation / timing / framing), optionally steered by the top competing class.

## 5. Data Model (Supabase / Postgres)

- `auth.users` — Supabase-managed accounts. Optional `profiles`.
- `signs` — catalog **seeded from the model's authoritative manifest** (label, gloss, model_class_index, category), **bootstrapped now from `model/artifacts/manifest/manifest.json`** (has `label_idx` 0–74 that MUST match training) and **enriched** with hint metadata + `reference_clip_ref` when `meta.json` ships (M7). Not hand-authored in the app; `model_class_index` is never invented app-side.
- `sessions` — `id, user_id, started_at, ended_at`.
- `attempts` — one row per capture: `id, user_id, session_id, sign_id, attempt_number, is_first_try, result, predicted_sign_id, confidence, margin, created_at`. **Metadata only — never frames/video** (Req 13).
- `sign_mastery` — per-user-per-sign rollup: `total_attempts, total_passes, first_try_pass_session_count, mastery_status (not_started/learning/mastered), last_practiced_at, last_result`.
- **Mastery rule:** **2 first-try passes in 2 distinct sessions** → `mastered`. Maintained by a Postgres trigger on `attempts` insert.
- **RLS** keyed to `auth.uid()` on `sessions`/`attempts`/`sign_mastery`.

## 6. System Architecture

Client SPA (**React + Vite**) does capture + inference + session orchestration + pass/fail; **Supabase** provides auth + Postgres (progress) + hosting. The trained **model artifact** (`models/asl-<version>.onnx` + `meta.json`) is a versioned static asset loaded into the browser. **Only outcome metadata** crosses to Supabase; **camera frames and inference never leave the browser** (Req 5 & 13), enforced by the absence of any upload path. Monorepo: `app/`, `model/`, shared `models/` registry.

## 7. Privacy (Req 13 & 15)

Raw video never uploaded by default; processed locally. Supabase stores only outcome metadata; standard request logs (timestamps/IP) are the only side data and touch no frames. Any future data collection would be an explicit, separately-documented, consented path. This flow is written into the privacy deliverable. **Pilot scope is non-commercial / research-educational** (per the ASL Citizen MSR license); reference clips are re-recorded rather than redistributed from the dataset.

## 8. Error / Edge Handling

- Camera denied/unavailable/unsupported → clear blocking message + retry guidance; can't enter practice without it.
- Out-of-vocab / no-sign attempt → low confidence fails the threshold (no false pass); decide explicit abstain vs. "none" handling during model calibration.
- Model artifact load failure → graceful error, retry.
- Offline / Supabase write failure → queue attempt writes locally, retry; never block the practice loop on the network.

## 9. Success Criteria (PRD §6)

Account → login → complete a session; 75 prompts available; camera grant + attempts; conservative pass/fail under documented conditions; targeted hints on fail; progress saved + visible across sessions; model explainable as trained-from-scratch/validated/versioned/limited; raw video stays local.

## 10. Cross-Stream Interface (summary; truth = `model/MODEL_WORKSTREAM.md §B`)

| Concern | Contract | Status |
|---|---|---|
| Runtime | ONNX Runtime Web | agreed |
| Artifact | `models/asl-<ver>.onnx` + `meta.json`, 2–6 MB | agreed |
| Clip input | **16 frames @ 112×112; app uniformly resamples ~3s window to 16** | **FROZEN 2026-05-21** |
| Vocabulary | 75 labels from `meta.json`; app never hardcodes | agreed |
| Pass/fail | app-side, thresholds from `meta.json` | agreed |
| Hints | per-sign metadata in `meta.json` | agreed |
| Reference clips | one short **re-recorded** clip per sign (NOT ASL Citizen — MSR license/PII), served from `models/refs/` | resolved — model/dataset side records |

## 11. Out of Scope (PRD §5)

Multi-language, sentence/conversation recognition, teacher/admin portal, rostering, SSO, default server-side inference, default raw-video upload, pretrained models, research-grade bias analysis, production deployment.
