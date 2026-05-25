# ASL Learning — practice app

React + Vite + TypeScript single-page app for the ASL Learning pilot. It captures a short
webcam clip, runs the from-scratch ASL recognizer in-browser (ONNX Runtime Web), and gives
pass/fail feedback with hints. Backend is Supabase (anonymous auth + progress storage).

**Live:** https://willowy-cactus-5db06d.netlify.app/ · Model details: `../model/`.

## Stack

- **React + Vite + TypeScript**, React Router, Vitest.
- **ONNX Runtime Web** — runs the recognizer client-side (wasm fetched from a version-matched CDN).
- **Supabase** — anonymous auth, `signs`/`sessions`/`attempts`/`sign_mastery` tables (RLS + a mastery trigger).

## How the recognition pipeline works

1. `useCamera` opens the webcam; the preview re-attaches via a ref callback so it survives `<video>` remounts.
2. `recordClip` captures ~3 s at a working resolution, computes a classical **motion-ROI crop**
   (`lib/roiCrop.ts`, parity-tested against the model's Python), crops to the signing region,
   resamples to **16 frames @ 112×112** (`lib/frameSampling.ts`, byte-parity with the model).
3. `lib/clipTensor.ts` builds the NFCHW tensor, normalized with the mean/std from the model's
   `meta.json`.
4. `inference/recognizer.ts` (`OnnxRecognizer`) runs the model → logits → softmax.
5. `lib/decision.ts` passes the attempt if the prompted sign is in the model's **top-3**
   (`passTopN`; confidence/margin gating retained but off for the current model). On the final
   failed attempt the result links to an external ASL reference for the word.

## Local development

```bash
cp .env.example .env.local        # fill in the Supabase keys (see below)
npm install
npm run seed                      # seed the `signs` table from ../model/artifacts/manifest/manifest.json
npm run dev                       # http://localhost:5173
```

Enable **Anonymous sign-ins** in the Supabase project (Auth → Providers), or the app shows
"couldn't start a session". Full DB setup: `SUPABASE_SETUP.md`.

The recognizer loads `public/models/asl-v1.onnx` (gitignored — exported from the model side;
re-publish on a fresh clone). `meta.json` (labels, normalization) is committed.

### Environment variables (`.env.local`)
| Var | Use |
|---|---|
| `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY` | browser client (anon key is RLS-gated, safe to ship) |
| `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` | **seed script only**, server-side — never in the bundle |
| `MANIFEST_PATH` | optional; defaults to `../model/artifacts/manifest/manifest.json` |
| `VITE_DEV_TOOLS` | `true` exposes the dev/eval model switcher; **forced off in prod** via `.env.production.local` (Req 7) |

## Scripts

| Command | Does |
|---|---|
| `npm run dev` | dev server |
| `npm run build` | typecheck + production build → `dist/` |
| `npm run test:run` | run the Vitest suite |
| `npm run seed` | seed the Supabase `signs` table |

## Build & deploy

```bash
npm run build        # -> dist/ (includes public/models/, SPA _redirects; dev-tools off)
```
`dist/` is a static bundle — deploy to any static host (Netlify Drop, Vercel, Cloudflare
Pages). SPA routing fallback is configured (`public/_redirects` + `vercel.json`). The app
needs **HTTPS** for the webcam (all these hosts provide it) and the deployed domain added to
the Supabase Auth URL configuration.
