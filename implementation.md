# ASL Learning — Implementation Status

Living progress tracker across both workstreams. Update as work lands. Task definitions: `task.md`.

**Last updated:** 2026-05-21

Legend: ✅ done · 🔄 in progress · ⬜ not started · ⛔ blocked

---

## Design & Planning (shared)
- ✅ PRD reviewed (`ASL learning.pdf`, 15 requirements)
- ✅ Architecture audit of initial vision draft
- ✅ Design spec (`docs/superpowers/specs/2026-05-21-asl-learning-pilot-design.md`)
- ✅ System architecture (`architecture.md`)
- ✅ Vision model plan (`vision-model-plan.md`)
- ✅ Task assignment (`task.md`)
- ✅ Cross-stream interface freeze — Alignments #1 (clip = 16 frames @112) & #2 (reference clips re-recorded) **RESOLVED 2026-05-21** (see `task.md §0`)

## Stream M — Model & Dataset
- ✅ Toolchain: PyTorch 2.7 + MPS (Apple M4); cv2 4.12; ONNX export path chosen
- ✅ ASL Citizen downloaded + resumed (~46 GB) and extracted (`model/data/`, gitignored)
- ✅ M1 — Data audit (`artifacts/audit/audit_candidates.csv`, `audit_all_glosses.csv`)
- ✅ M1 — **75-sign vocab frozen (proposed)** (`artifacts/manifest/proposed_vocab.json`; ≥25 videos & ≥4 signers each)
- ✅ Workstream scaffold (`src/asl/`: `audit.py`, `beginner_vocab.py`, `freeze_vocab.py`)
- ✅ M2 — Preprocessing + tensor cache (`artifacts/cache/clips.npz`, 2362 clips, 16×112×112); subset videos extracted; manifest (`artifacts/manifest/manifest.json`) + norm.json built.
- ✅ M2.5 — Signer re-split favoring training (~23 train / ~3 val / ~5 test per class, signer-disjoint) — official split was test-heavy for classification.
- ✅ M-export — **ONNX round-trip smoke test PASSED** (torch==onnx @2.7e-7; quant 0.11 MB). Browser path de-risked. (Stub-only model; real one from M3.)
- 🔄 M3 — Train + signer-held-out val: baseline running locally (M4/MPS); Colab path ready (`notebooks/train_colab.ipynb` + `python -m asl.package_for_colab`).
- ⬜ M4 — Webcam eval set + WLASL Tier-2 cross-dataset eval.
- ⬜ M5 — Threshold calibration incl. negative/unknown reject → `meta.json`.
- ⬜ M6 — Hint metadata + reference clips (re-record → `models/refs/`).
- ⬜ M7 — ONNX export + quantize + publish to `models/` + validation report.

## Stream A — Application & System
- ✅ A1 — Scaffolded `app/` (Vite React-TS + ONNX Runtime Web + Supabase client + **vitest**); `tsc -b`, build, and tests all green. (Note: test config in separate `vitest.config.ts` to avoid vite 8 / vitest nested-vite plugin type clash.)
- ✅ A2 — Schema applied to **live Supabase project**; **75 signs seeded & verified** (contiguous 0–74, ANGRY…YES). Migration `app/supabase/migrations/0001_init.sql` (signs/sessions/attempts/sign_mastery + RLS + mastery trigger); seed transform `buildSignSeed` **TDD 5/5**; runner `scripts/seed-signs.ts`. (Auth wiring is A3.)
- ✅ A3 — Supabase email auth (`SessionProvider` + `onAuthStateChange`), `ProtectedRoute`, login page (sign in/up), dashboard shell (progress summary stats + 75-sign status grid). `summarizeProgress` **TDD 6/6**. `tsc -b`, build, vitest (11/11), and dev-server boot all green. **Pending manual browser check: full login → dashboard click-through with a real account.**
- 🔄 UI/testing iteration (2026-05-22): **anonymous auth** auto sign-in (login screen deferred; requires Supabase "Anonymous sign-ins" enabled); dashboard simplified to a **progress bar** (mastered/learning/to-go, `progressBarSegments` TDD) — 75-grid removed; **dev-only model switcher** (`ModelProvider`/`ModelSwitcher`, `lib/models.ts` TDD) to compare own vs a pretrained baseline. ⚠️ **The baseline/switcher MUST be stripped from the graded pilot recognition path + documented (Req 7).** Recognizer impls land in A6. tsc/build/vitest (28/28) green.
- ✅ A4 — `useCamera` hook (getUserMedia, release-on-unmount); `describeCameraError` maps denied/unavailable/in_use/unsupported (**TDD**); live brightness check (`computeMeanLuminance` + `assessBrightness`, **TDD**) via frame sampling; framing-guide overlay; `/practice` camera-check screen wired from dashboard. tsc/build/vitest (21/21) green. **Pending manual browser check: real camera grant, denied/in-use paths, live lighting feedback. Visual polish deferred to the post-pipeline UI pass.**
- ✅ A5 — Capture pipeline: `recordClip` (~3s, center-crop→112, uniform resample to 16). **`sampleFrameIndices` is byte-parity with the model's `sample_indices`** — guarded by a golden-vector fixture (`app/src/lib/__fixtures__/frameSampling.json`, generated from the model fn incl. numpy banker's rounding), **TDD 25/25**. `framesToTensor` NFCHW normalize (TDD). Countdown→record flow in `PracticePage` produces the 16-frame clip (inference is A6). Model registry now own-v1 / own-v2 / baseline. 58 tests green. **Pending manual browser check: real on-camera capture.**
- ✅ A6 — `Recognizer` interface; `StubRecognizer` (deterministic dev fallback, per-model salt) + `OnnxRecognizer` (ORT Web, **dynamically imported** so the 26MB wasm loads only when a real model runs); `createRecognizer` factory keyed by the selected model (stub until `MODEL_URLS`/`models/` populated). `softmax` + `topK` **TDD**. PracticePage now runs capture → `framesToTensor` → recognize → softmax → **top-3**, with labels pulled from the DB (`signs` by `model_class_index`). 68 tests green; build OK. **To finish at M7: add the ONNX URL to `MODEL_URLS`, replace `PLACEHOLDER_NORM` with meta.json mean/std, confirm input name/shape. Pass/fail + hints = A7.**
- ✅ A7 — `decidePassFail` (conservative: argmax==prompt AND P(prompt)≥threshold AND margin≥class_margin; classifies failure as wrong_sign / low_confidence / ambiguous) + `buildHint` (rule-based, prefers movement>handshape>…, graceful fallback without metadata) — both **TDD**. PracticePage now prompts a target sign → record → recognize → **PASS/FAIL + targeted hint** + "Next sign". 78 tests green; build OK. **Thresholds = placeholder `DEFAULT_THRESHOLDS` until per-class meta.json values (M5/M7); hint metadata null until meta.json ships.**
- ✅ A8 — Practice loop per word: prompted target → countdown → record → recognize → PASS/FAIL + hint; **2-retry resolution** (`wordOutcome` TDD: pass→advance, fail→retry, final fail→reveal reference placeholder), Skip, retry/continue actions. Floating-overlay visual polish deferred to the UI pass.
- ✅ A9 — Session orchestrator: creates a `sessions` row, builds the **mastery-prioritized deck** (`buildDeck` TDD — least-recently-practiced first, review slots), iterates the deck, writes `attempts` (`buildAttemptRow` TDD) which **fires the mastery trigger**, ends the session (`ended_at`) + shows a summary. 88 tests green; build OK. **Needs browser + camera + anon auth enabled to verify progress persistence + mastery rollup end-to-end.**
- ✅ A10 — Dashboard recent-practice history (last 8 attempts: sign label + pass/fail + relative time via `timeAgo`, **TDD**), atop the existing progress bar.
- ✅ A11 — Privacy documentation `docs/privacy.md` (Req 15). Offline resilience: `pendingAttempts` localStorage queue (**TDD**) — failed attempt writes stash locally and flush on next session init; the practice loop never blocks on the network.

## Open Decisions / Risks (carry-over)
- ✅ Alignment #1 — clip shape FROZEN at **16 frames @112** + uniform resample.
- ✅ Alignment #2 — reference clips **re-recorded** (model/dataset side → `models/refs/`).
- ✅ Frontend framework — **React + Vite** confirmed.
- License scope: **non-commercial / research-educational** pilot (ASL Citizen).
- ⚠️ **Req 7 landmine:** the in-app model switcher includes a pretrained baseline for dev/comparison only. It MUST be removed from the submission recognition path and documented as eval-only. Track to submission.
- ⬜ Full UI design pass — deferred by user until the pipeline is finished (feedback placement, practice-screen polish, etc.).
- Vision risks (data sufficiency, domain shift, hand localization) tracked in `vision-model-plan.md §11`.

## Integration Milestones
1. ✅ Interface frozen (Alignments resolved 2026-05-21)
2. ✅ App runs end-to-end against a **stub** model (capture → resample → tensor → recognize → pass/fail → hint → attempts → mastery rollup → session summary)
3. ⬜ Real ONNX (M7) swapped in → end-to-end pass/fail working
4. ⬜ Pilot-ready: progress persists, privacy doc done, validation report attached
