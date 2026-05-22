# Constellation v2 — Plan 5: Combined Export + Validation Report

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Wire the trained detector + landmark + recognizer into the **combined ONNX graph** (decision (b) from Plan 1), export + quantize within the <20 MB budget, verify it round-trips in ONNX Runtime Web, publish the versioned artifact + `meta.json` to the `models/` registry, and write the honest validation report (incl. the ADR-0001 live-tester check).

**Architecture:** Plan 5 of 5. Replaces Plan 1's dummy modules with the real trained weights inside `CombinedConstellation` (16 frames → detect → topk → roi_align → landmark → geometry → recognizer → 75 logits). Reuses the Plan-1 export spike machinery and the carry-forward notes. Validation follows **ADR 0001** (revised).

**Tech Stack:** PyTorch, onnx, onnxruntime, onnxruntime-web (node), pytest. Outputs to `models/` (shared registry the app pins).

---

## Conventions
- Combined-graph contract: input `frames (16,3,128,128)` (the app resizes its 112² capture to the detector's 128² inside the graph, OR the graph accepts 112² and the detector is trained at 112 — **decide in Task 1 to match the app's frozen 112² capture**; document it). Output: `logits (1,75)`. App contract otherwise unchanged.
- Quantization: int8 dynamic (or fp16 if accuracy drops); **total exported size < 20 MB** (spec §3.7), verified by an assertion.
- Carry-forwards from Plan 1: use the **legacy `dynamo=False`** torch.onnx exporter (onnxscript not installed); **verify RoiAlign `sampling_ratio`** behavior on the real graph.

## File Structure
```
model-v2/src/aslv2/export/
  __init__.py
  build_combined.py   # load trained weights into CombinedConstellation (real)
  export_onnx.py      # export + quantize + size assert + Python ORT round-trip
configs/export.yaml
models/asl-v0.2.onnx           # published artifact (gitignored; registry copy)
models/asl-v0.2.meta.json      # from Plan 4, finalized here
models/asl-v0.2.bundle.json    # content hashes of all components (versioning)
docs/validation/constellation-v2-report.md
docs/validation/live-tester-protocol.md   # PREDECLARED before any live clips
tests/test_build_combined.py, test_export_onnx.py
```

---

## Task 1: Build the real combined module

**Files:** Create `export/__init__.py` (empty), `export/build_combined.py`; Test `tests/test_build_combined.py`.

- [ ] **Step 1: Decision (document):** Confirm the graph's input resolution. The app's frozen capture is **112²** (`MODEL_WORKSTREAM.md §B`). Either (i) train/run the detector at 112² (retrain Plan 2 at 112 — preferred for an unchanged contract), or (ii) add an internal resize 112→128 as the graph's first op. Pick one, document in `configs/export.yaml` and the report. (Resize op keeps Plan 2/3 work; verify it exports.)
- [ ] **Step 2: Failing test** — `build_combined(detector_ckpt, landmark_ckpt, recog_ckpt)` returns a module whose forward on `frames` yields `(1,75)`, and whose three sub-modules each pass `assert_within_budget`. (Use the real checkpoints from Plans 2–4; if running in CI without them, mark `skipif` on missing files.)
- [ ] **Step 3: FAIL** → **Step 4: Implement** `build_combined`: load weights into the real Detector/Landmark/Recognizer, assemble the same graph shape proven in the Plan-1 spike (`src/aslv2/export_spike/combined.py`), but with real geometry (use the tensorized geometry that matches `aslv2.geometry`; the numpy version remains the offline source of truth). Reuse the static `topk` selection (2 hands + 1 head) and `roi_align` crop. **Step 5: PASS** → **Step 6: Commit** `feat(export): real combined Constellation module`.

---

## Task 2: Export + quantize + size budget + Python round-trip

**Files:** Create `export/export_onnx.py`, `configs/export.yaml`; Test `tests/test_export_onnx.py`.

- [ ] **Step 1: Failing test** — `export_combined_real(out_path, ckpts)` produces an ONNX file; loading it in onnxruntime and running `frames` matches the torch reference within atol=1e-2 (looser than the spike's 1e-3 because quantization). Assert the quantized file size **< 20 MB**.
- [ ] **Step 2: FAIL** → **Step 3: Implement** `export_onnx.py`: `torch.onnx.export(..., opset=17, dynamo=False)`; then `onnxruntime.quantization` int8 dynamic; assert `os.path.getsize(out) < 20*1024*1024`; Python ORT round-trip vs torch. Verify the **RoiAlign sampling_ratio** carry-forward produces correct crops (compare crop tensors torch vs ORT). **Step 4: PASS** → **Step 5: Commit** `feat(export): combined ONNX export + quantize + size/round-trip gates`.
- [ ] **Step 6 (GATE — accuracy after quantization):** Run the quantized graph on the signer-held-out **test** clips; top-1 must stay within **~2 points** of the un-exported recognizer (Plan 4 Task 8). If quantization tanks accuracy, fall back to fp16. Record before/after.

---

## Task 3: ONNX Runtime Web round-trip (the deployment gate)

**Files:** reuse `scripts/ort_web_probe.mjs` (Plan 1).

- [ ] **Step 1 (RUN):** `cd model-v2 && node scripts/ort_web_probe.mjs models/asl-v0.2.onnx; echo "exit=$?"` — confirm the **real, quantized** graph loads and runs in ORT-Web (WASM) → `logits.dims [1,75]`, exit 0. (Plan 1 proved the op set on dummy weights; this confirms the real quantized file.)
- [ ] **Step 2 (GATE):** If a quantized op is unsupported in ORT-Web WASM, try fp16 / per-op exclusion; if still unsupported, fall back to export option (a) separate files + coordinate the §B change with the app agent (spec §6). Record the outcome in `export_decision.md`.
- [ ] **Step 3 (RUN — latency):** Time 1 inference of 16 frames in ORT-Web; **< 1 s** target (spec). If over, apply the keyframe-detection mitigation (spec §7). Record.

---

## Task 4: Publish to the models/ registry + versioning bundle

- [ ] **Step 1:** Copy/confirm `models/asl-v0.2.onnx` (artifact) + `models/asl-v0.2.meta.json` (from Plan 4). Verify `meta.json` matches `MODEL_WORKSTREAM.md §B` shape (labels/input/thresholds/hints/temperature).
- [ ] **Step 2:** Write `models/asl-v0.2.bundle.json` = content hashes (sha256) of: detector ckpt, landmark ckpt, recognizer ckpt, the 3 dataset/cache manifests, preprocessing+training configs, the onnx, and meta.json (spec §8, mirrors `vision-model-plan.md §9`).
- [ ] **Step 3 (app handoff):** Note in `task.md`/`implementation.md` that the app should pin `asl-v0.2` (set `MODEL_URLS` → `models/asl-v0.2.onnx`, load `meta.json` for norm + thresholds, per the app's M7 step). The contract is unchanged (frames-in/logits-out), so no app code change beyond pinning the new version + stripping the dev pretrained-baseline switcher from the graded path (existing Req-7 landmine). **Commit** bundle + doc updates.

---

## Task 5: Live-tester protocol (PREDECLARE before any live clip) — ADR 0001

**Files:** Create `docs/validation/live-tester-protocol.md`.

- [ ] **Step 1:** Predeclare, BEFORE recording any live clip: the exact target signs to test, attempt counts per sign, lighting/distance/framing conditions, and the pass/fail summary format. (ADR 0001 §4 — guards against p-hacking.) **Commit this file before running any live test.**
- [ ] **Step 2 (RUN, optional/when convenient):** Record live attempts through the actual app capture path under those conditions; collect the predeclared summary. This is a **separate sanity check**, never tuned-against, never the headline.

---

## Task 6: Validation report (the deliverable)

**Files:** Create `docs/validation/constellation-v2-report.md`.

- [ ] **Step 1:** Write the report covering (spec §8, ADR 0001):
  - **Headline:** v1 (16.4% val / 18% test) → **v2** signer-held-out test + **WLASL cross-dataset** accuracy; report the gap honestly.
  - **Per-stage metrics on the labeled ASL audit slice:** detector detection-rate (hand/head), landmark PCK, head detection-rate/stability.
  - **Per-class accuracy + confusion matrix** (Plan 4 Task 8); false-pass/false-fail per class after thresholds.
  - **Req-7 reinterpretation note** + from-scratch evidence (training curves, init, dependency list = frameworks only) + dataset provenance/licenses for ALL THREE models (100DOH/EgoHands, FreiHAND/COCO-WholeBody, ASL Citizen).
  - **Size-budget revision** (<20 MB, per-stage table) with the actual quantized sizes.
  - **Live-tester sanity check** results under the predeclared protocol — clearly labelled as NOT representative-learner validation; live loop still **declared unvalidated for real users**.
  - **Known limitations & failure cases** (domain shift, confusable signs, error propagation across the 3 stages).
- [ ] **Step 2 (RUN):** Generate the WLASL cross-dataset number: run the exported model on a WLASL slice intersecting the 75-sign vocab (internal-only eval, per `MODEL_WORKSTREAM.md §D.6`). Record.
- [ ] **Step 3: Commit** the report. This + the bundle = the v0.2 deliverable.

---

## Self-Review
- Spec coverage: §6 combined-graph export realized + ORT-Web confirmed on the real quantized file (Tasks 2–3) ✓; §3.7 <20 MB enforced (Task 2) ✓; §8 versioning bundle + validation report (Tasks 4/6) ✓; ADR-0001 headline=datasets, live-tester predeclared+separate (Tasks 5/6) ✓; Req-7 provenance for all 3 models (Task 6) ✓.
- TDD: build_combined + export have failing-first tests; export/ORT-Web/latency/quantization-accuracy are run-and-gate with numeric thresholds and explicit (a)-fallback.
- Integration: consumes Plans 2–4 checkpoints + meta.json; publishes to `models/` for the app (contract unchanged). Closes the v2 build.
- Open dependency: live-tester protocol must be committed BEFORE live clips are used as evidence (ADR 0001).
