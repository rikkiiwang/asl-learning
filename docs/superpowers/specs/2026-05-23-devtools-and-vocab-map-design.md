# Dev-tools Gating + Vocabulary Map — Design

**Date:** 2026-05-23
**Status:** Proposed
**Scope:** Stream A (learner app). Frontend-only. No schema/model changes.
**Branch:** `app-motivation-pack` (worktree `~/Desktop/asl-motivation-pack`).

Two independent features built together.

---

## Feature A — Dev-tools gating (PRD Req 7)

**Problem:** The pretrained `baseline` model is a selectable option in `lib/models.ts`, and
`<ModelSwitcher>` renders unconditionally on the dashboard. Req 7 forbids pretrained models in the
graded recognition path, so the graded build must provably exclude both the baseline option and the
switcher UI.

**Approach:** A single explicit flag, **off by default**.

- `app/src/lib/devTools.ts` — `export function devToolsEnabled(): boolean { return import.meta.env.VITE_DEV_TOOLS === 'true'; }`
- `lib/models.ts` — add pure `selectableModels(includeBaseline: boolean): ModelOption[]` returning
  `MODELS` filtered to `kind === 'own'` unless `includeBaseline`. `getModelById` gains a guard: when
  the requested id isn't in the selectable set, fall back to `DEFAULT_MODEL_ID` (already `own-v1`).
- `models/ModelProvider.tsx` — build its `models` list from `selectableModels(devToolsEnabled())`, and
  coerce a stale stored id (e.g. `baseline`) to default when it isn't selectable, so a leftover
  `localStorage` value can't reactivate the pretrained path.
- `pages/DashboardPage.tsx` — render `<ModelSwitcher />` only when `devToolsEnabled()`.
- `app/.env.example` — document `VITE_DEV_TOOLS=` (blank/false). `.env.local` (gitignored) sets it
  `true` for local dev. The graded build simply omits it.

**Net:** With the flag off, the switcher UI is gone, `baseline` is unselectable, and `createRecognizer`
only ever sees `own-*` models — the pretrained path is unreachable. With it on, dev/eval behaves as
today.

**Tests** (`lib/models.test.ts`, new): `selectableModels(false)` excludes the baseline / includes only
`own` kinds; `selectableModels(true)` includes the baseline; `getModelById('baseline')` falls back to
default. `devToolsEnabled` is exercised via `vi.stubEnv('VITE_DEV_TOOLS', ...)`.

---

## Feature B — Vocabulary map (`/map`)

**Goal:** A status grid of the whole vocabulary so learners see the shape of their progress and can
jump straight into practicing any sign.

### Routing & navigation
- New protected route `/map` → `VocabMapPage` (added to `App.tsx`).
- Dashboard: a `btn-ghost` link "🗺 Vocabulary map" under the progress card → `/map`. Map page has a
  "← Back" link to `/`.

### `VocabMapPage`
- Loads `signs` + `sign_mastery` (same two queries the dashboard already uses) and computes the
  `summarizeProgress` rollup for the header (% + legend), reusing existing `lib/progress.ts`.
- Renders a header (title, `% mastered` pill, legend) and a `VocabGrid`.

### Components
- `components/VocabGrid.tsx` — responsive CSS grid of `SignTile`s.
- `components/SignTile.tsx` — a tile showing the sign label, background tinted by status via a shared
  map: mastered → `--seg-mastered`, learning → `--seg-learning`, not_started → `--seg-togo`. `onClick`
  selects the sign.
- `components/SignDetailModal.tsx` — centered card overlay (click backdrop to dismiss) showing the
  label, status, `total_passes`/`total_attempts`, `last_practiced_at` via `timeAgo`, and a
  `▶ Practice this` button → `navigate('/practice?sign=' + sign.id)`.
- `lib/vocabStatus.ts` — pure `statusStyle(status): { color: string; label: string }` mapping the three
  statuses to CSS-var color + display label. Single source for tile + modal coloring.

### Targeted practice (the "Practice this" deep-link)
- `lib/deck.ts` — add pure `targetedDeck(signs: Sign[], signId: string): Sign[]` returning `[match]`
  (or `[]` if the id is unknown).
- `pages/PracticePage.tsx` — read `useSearchParams()`. When `sign` is present and resolves, the deck is
  `targetedDeck(signs, signParam)` (a one-word session); otherwise the existing `buildDeck(...)` path is
  unchanged. The deck counter then reads "1 / 1".

**Tests:** `lib/deck.test.ts` (extend) — `targetedDeck` returns the single matching sign; returns `[]`
for an unknown id. `lib/vocabStatus.test.ts` (new) — each status maps to the expected color var + label.
Grid/tile/modal are presentational (covered by build).

---

## Constraints honored
- **Req 7** — Feature A is the whole point: provable exclusion of the pretrained path from the graded
  build.
- **Req 13/5** — Feature B reads only existing outcome metadata; nothing new leaves the device.
- **No schema/model changes**; additive components + one new route + one query-param branch in practice.

## Error handling
- Map load failure → the page shows the same graceful error treatment as the dashboard; tiles simply
  don't render.
- `targetedDeck` with an unknown/blank id → empty deck → PracticePage falls through to its normal
  "no words" / finished handling (or we guard: if targeted deck is empty, fall back to `buildDeck`).
- `devToolsEnabled()`/`selectableModels` are total functions.

## Rollout
Additive. Verify: `tsc -b` + `vite build` clean; full vitest suite green (existing 113 + new model/deck/
vocabStatus tests); `/map` renders the grid, tapping a tile opens the modal, "Practice this" starts a
one-sign session; with `VITE_DEV_TOOLS` unset the dashboard shows no model switcher and `baseline` is
unselectable.
