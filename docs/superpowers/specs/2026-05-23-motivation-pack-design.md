# Motivation Pack — Design

**Date:** 2026-05-23
**Status:** Proposed (awaiting review)
**Scope:** Stream A (the learner app). Frontend-only. No schema changes, no model changes.

## Goal

Make the app feel more like Alpha School's "show up every day and close your ring" loop, using
only data we already record. Four mechanics, all derived from existing tables:

1. **🔥 Streak pill** — consecutive days the learner has practiced.
2. **Daily-goal ring** — an Apple-Watch-style ring that closes as you hit today's practice target
   (layout **A**: the ring is its own "Today's goal" card at the top of the dashboard).
3. **XP tickups** — `+10 XP` on each passing recent-practice row; XP banked shown on the
   end-of-session screen.
4. **End-of-session celebration** — confetti + the session's XP, streak, and whether the ring closed.

Non-goals (possible fast-follows, explicitly out of scope here): achievement badges, a mastery
"constellation" grid, live framing/quality meter, adaptive difficulty, leaderboards/cohorts,
a persisted XP currency table.

## Constraints honored

- **PRD Req 13/5** — no new data leaves the device; we only read outcome metadata already in Supabase.
- **PRD Req 7** — untouched; this is UI on top of the recognition path, not part of it.
- **No new schema** — every number is computed from `sessions`, `attempts`, `sign_mastery`.
- **Minimal deps** — confetti is a ~40-line self-contained canvas effect, no npm package.

## Data sources (existing tables)

| Mechanic        | Source                                                                 |
|-----------------|------------------------------------------------------------------------|
| Streak          | `sessions.started_at` for the user (grouped by local calendar day)     |
| Daily-goal ring | distinct `attempts.sign_id` with `created_at >=` local midnight (today)|
| Recent-row XP   | the `result` already loaded for each recent attempt                    |
| Session XP      | passes counted this session (already tracked in `PracticePage` `stats`)|

### A note on ring semantics (decision to confirm)

The mockup copy said "master 2 more to close the ring." Mastery in this app requires **2 first-try
passes across 2 distinct sessions** (DB trigger), so you literally *cannot* master N new signs in a
single day from scratch — a "master 5 today" ring would be unreachable and dishonest.

**Decision:** the ring counts **distinct signs practiced today** toward a daily target (default **5**).
Copy becomes "Practice 2 more to close the ring." This is honest, habit-forming, and mirrors the
streak (both reward showing up). `DAILY_GOAL` is a single exported constant, trivial to tune.

## New pure modules (unit-tested, no I/O)

- **`lib/streak.ts`** — `computeStreak(sessionStartedAt: string[], now: Date): number`.
  Bucket timestamps into local calendar days; count the consecutive run ending **today**, or ending
  **yesterday** if nothing today yet (streak is still alive until the day ends). Empty → 0.
- **`lib/dailyGoal.ts`** — `DAILY_GOAL = 5`; `goalProgress(signsPracticedToday: number, target = DAILY_GOAL)`
  → `{ done, target, remaining, complete }` (clamps `done` to `target`).
- **`lib/xp.ts`** — `XP_PER_PASS = 10`; `attemptXp(result: 'pass' | 'fail')`;
  `sessionXp(passes: number)`. Single source of truth for the point value.

## Components

- **`components/StreakPill.tsx`** — gold `pill pill-gold` "🔥 {n} day". Renders nothing when `n < 1`.
  Placed inline beside the dashboard `page-title`.
- **`components/GoalRing.tsx`** — SVG progress ring (two circles + the existing `--accent` gradient,
  center text `done/target`), props `{ done, target, size }`. Pure presentational; the
  port of the mockup's `ring()` helper.
- **`components/DailyGoalCard.tsx`** (layout A) — a `.card` at the top of the dashboard: `GoalRing` on
  the left, "Today's goal" + "Practice N more to close the ring" (or "Goal complete! 🎉" when done) in
  the middle, a `pill pill-soft` reward chip (`+{sessionXp(target)} XP`) on the right.
- **`RecentHistory` row update** — append a right-aligned `+10 XP` (pass) / faint `+0` (fail) span,
  using `attemptXp`.
- **`components/SessionComplete.tsx`** — replaces the inline `finished`-phase markup in `PracticePage`.
  Shows the existing "passed X of Y", **XP earned this session** (`sessionXp(stats.passed)`), the
  updated streak, a "ring closed" line if today's goal was met, and fires `Confetti` once on mount.
- **`components/Confetti.tsx`** — self-contained: a fixed full-screen `<canvas>`, ~80 particles with
  the palette's purples/gold/pink, gravity + fade, `requestAnimationFrame`, auto-stops after ~1.8s and
  unmounts. Respects `prefers-reduced-motion` (renders nothing). No dependency.

## Data flow

**Dashboard** (`DashboardPage`): extend the existing `Promise.all` with two reads —
`sessions.select('started_at')` and `attempts.select('sign_id').gte('created_at', localMidnightISO())`
(client computes local-midnight UTC ISO). Feed `computeStreak` and `goalProgress`. The streak value is
lifted so `StreakPill` and the celebration agree.

**Practice** (`PracticePage`): `stats` already carries `{ completed, passed }`. On `finish`, also
capture the practiced-today count and recompute streak (one `sessions` read, or pass the count the
session contributed) so `SessionComplete` can show the post-session streak and ring state. Keep the
existing fire-and-forget attempt-write loop untouched.

## Error handling

- Streak/goal reads fail → pill and ring simply don't render (dashboard already degrades gracefully on
  a load error); the rest of the page is unaffected.
- `computeStreak`/`goalProgress` are total functions (empty input → 0), so no throw paths.
- Confetti is best-effort decoration; if canvas is unavailable it no-ops.

## Testing

- `lib/streak.test.ts` — today-only; today+yesterday consecutive; gap breaks streak; streak alive when
  last practice was yesterday but not today; multiple sessions same day count once; timezone-boundary
  case (two sessions either side of local midnight).
- `lib/dailyGoal.test.ts` — under/at/over target clamps; `complete` flips at target.
- `lib/xp.test.ts` — pass=10, fail=0, `sessionXp` multiplies.
- Components are thin/presentational; covered by existing build + a render smoke check. No change to the
  99 existing tests is expected (additive only).

## Rollout

Additive — no migration, no flag. Verify: `tsc -b` clean, `vite build` clean, full vitest suite green
(existing 99 + new lib tests), dashboard renders streak pill + goal ring, session-complete shows
confetti + XP.
