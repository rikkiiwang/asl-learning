# Privacy — Camera & Data Handling

ASL Learning pilot. Covers how the app handles camera input and learner data (PRD Req 5, 13, 15).

## Summary

**Camera frames never leave the browser.** All recognition runs locally via ONNX Runtime Web. The only data sent off-device is **outcome metadata** (which sign was attempted, pass/fail, confidence, timestamps) — never images or video.

## What the camera is used for

- The app requests camera access only to capture short (~3 s) signing attempts during practice.
- Frames are drawn to an in-page `<canvas>`, downscaled to 112×112, and fed directly into the local model. They live only in browser memory for the duration of an attempt and are discarded after inference.
- If camera access is denied, unavailable, or unsupported, the app explains why and does not proceed to practice.

## What leaves the device

Only rows written to the app's database (Supabase / Postgres):

| Stored | Example |
|---|---|
| `sessions` | session start/end timestamps |
| `attempts` | sign id, attempt #, pass/fail, model confidence, top-2 margin, timestamp |
| `sign_mastery` | per-sign rollup (counts, mastery status) |

**No frames, images, video, or derived landmarks are ever uploaded.** This holds *by construction* — there is no code path that sends pixel data anywhere.

## Accounts

The pilot uses **anonymous authentication**: a session is created without personal details so progress can persist across visits. No email, name, or profile is required. Row-level security ensures each learner can read only their own `sessions`, `attempts`, and `sign_mastery`.

## Operational logging

As with any hosted backend, Supabase records standard request metadata (timestamps, IP address, endpoint). This is infrastructure logging and contains **no camera content**.

## Data retention & future collection

- Default behavior collects **no raw video**. There is no opt-in video upload in the pilot.
- If a future version proposes collecting clips (e.g., to expand the dataset), it will require **explicit, informed learner consent** and be documented separately from this default behavior.

## Note for evaluators

The app includes a **dev-only model switcher** with a pretrained "baseline" option used solely to test the pipeline and benchmark against the from-scratch model. It is **not** part of the graded recognition path and is excluded from the pilot submission build (PRD Req 7).
