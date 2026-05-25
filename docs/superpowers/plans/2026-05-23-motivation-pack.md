# Motivation Pack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a streak pill, a daily-goal ring (layout A), per-row + session XP, and an end-of-session confetti celebration to the ASL learner app, using only data already in Supabase.

**Architecture:** Three pure, unit-tested helper modules (`streak`, `dailyGoal`, `xp`) compute every number from existing `sessions`/`attempts`/`sign_mastery` rows. Thin presentational components render them. `DashboardPage` gains two small reads (sessions + today's attempts); `PracticePage`'s finish screen is extracted into a `SessionComplete` component with confetti. Additive only — no schema, no model, no flag.

**Tech Stack:** Vite + React 19 + TypeScript, vitest, Supabase JS client. Confetti is a self-contained `<canvas>` effect (no new dependency).

---

## File Structure

- `app/src/lib/streak.ts` (+ test) — consecutive-day streak from session timestamps.
- `app/src/lib/dailyGoal.ts` (+ test) — today's-target ring math + `DAILY_GOAL`.
- `app/src/lib/xp.ts` (+ test) — `XP_PER_PASS`, `attemptXp`, `sessionXp`.
- `app/src/lib/time.ts` — add `localMidnightISO()` helper (today's local midnight as UTC ISO).
- `app/src/components/StreakPill.tsx` — gold pill, hidden when streak < 1.
- `app/src/components/GoalRing.tsx` — SVG progress ring.
- `app/src/components/DailyGoalCard.tsx` — layout-A "Today's goal" card (ring + copy + reward chip).
- `app/src/components/Confetti.tsx` — full-screen canvas burst, reduced-motion aware.
- `app/src/components/SessionComplete.tsx` — finish-phase screen (XP + streak + ring + confetti).
- `app/src/pages/DashboardPage.tsx` — fetch sessions + today's attempts; render pill + goal card; XP on recent rows.
- `app/src/pages/PracticePage.tsx` — capture today's-practiced count; render `SessionComplete`.

**Test commands:** `cd app && npx vitest run <file>` for a single file; `cd app && npm run test:run` for the full suite; `cd app && npm run build` for typecheck + bundle. Existing suite is 99 tests and must stay green.

**Commit note:** The repo spans the whole Desktop and the user manages git. Each "Commit" step below stages only the files named in that task. If the user prefers, commits can be batched at the end — confirm before the first commit.

---

### Task 1: XP helper

**Files:**
- Create: `app/src/lib/xp.ts`
- Test: `app/src/lib/xp.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
// app/src/lib/xp.test.ts
import { describe, it, expect } from 'vitest';
import { XP_PER_PASS, attemptXp, sessionXp } from './xp';

describe('xp', () => {
  it('awards XP_PER_PASS for a pass and 0 for a fail', () => {
    expect(attemptXp('pass')).toBe(XP_PER_PASS);
    expect(attemptXp('fail')).toBe(0);
  });
  it('sessionXp scales with passes', () => {
    expect(sessionXp(0)).toBe(0);
    expect(sessionXp(3)).toBe(3 * XP_PER_PASS);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd app && npx vitest run src/lib/xp.test.ts`
Expected: FAIL — cannot resolve `./xp`.

- [ ] **Step 3: Write minimal implementation**

```ts
// app/src/lib/xp.ts
/** Points model for the motivation pack. Single source of truth for XP values. */
export const XP_PER_PASS = 10;

/** XP earned by one attempt outcome. */
export function attemptXp(result: 'pass' | 'fail'): number {
  return result === 'pass' ? XP_PER_PASS : 0;
}

/** XP banked from a session given how many words passed. */
export function sessionXp(passes: number): number {
  return passes * XP_PER_PASS;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd app && npx vitest run src/lib/xp.test.ts`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add app/src/lib/xp.ts app/src/lib/xp.test.ts
git commit -m "feat(app): XP helper for motivation pack"
```

---

### Task 2: Daily-goal helper

**Files:**
- Create: `app/src/lib/dailyGoal.ts`
- Test: `app/src/lib/dailyGoal.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
// app/src/lib/dailyGoal.test.ts
import { describe, it, expect } from 'vitest';
import { DAILY_GOAL, goalProgress } from './dailyGoal';

describe('goalProgress', () => {
  it('uses DAILY_GOAL as the default target', () => {
    expect(goalProgress(0).target).toBe(DAILY_GOAL);
  });
  it('reports remaining and incomplete below target', () => {
    expect(goalProgress(3, 5)).toEqual({ done: 3, target: 5, remaining: 2, complete: false });
  });
  it('clamps done at target and marks complete', () => {
    expect(goalProgress(7, 5)).toEqual({ done: 5, target: 5, remaining: 0, complete: true });
  });
  it('is complete exactly at target', () => {
    expect(goalProgress(5, 5).complete).toBe(true);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd app && npx vitest run src/lib/dailyGoal.test.ts`
Expected: FAIL — cannot resolve `./dailyGoal`.

- [ ] **Step 3: Write minimal implementation**

```ts
// app/src/lib/dailyGoal.ts
/** How many distinct signs a learner aims to practice each day. Tune freely. */
export const DAILY_GOAL = 5;

export interface GoalProgress {
  done: number;     // clamped to [0, target]
  target: number;
  remaining: number;
  complete: boolean;
}

/** Ring math: how today's practice count maps onto the daily target. */
export function goalProgress(practicedToday: number, target: number = DAILY_GOAL): GoalProgress {
  const done = Math.max(0, Math.min(practicedToday, target));
  return { done, target, remaining: target - done, complete: done >= target };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd app && npx vitest run src/lib/dailyGoal.test.ts`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add app/src/lib/dailyGoal.ts app/src/lib/dailyGoal.test.ts
git commit -m "feat(app): daily-goal ring math"
```

---

### Task 3: Streak helper

**Files:**
- Create: `app/src/lib/streak.ts`
- Test: `app/src/lib/streak.test.ts`

Streak = number of consecutive local calendar days, ending today (or ending yesterday if nothing
practiced yet today — the streak is alive until the day ends), on which the learner started ≥1 session.

- [ ] **Step 1: Write the failing test**

```ts
// app/src/lib/streak.test.ts
import { describe, it, expect } from 'vitest';
import { computeStreak } from './streak';

// Build a local-time ISO string for a given Y-M-D + hour (no Z → parsed as local).
const at = (y: number, m: number, d: number, h = 12) =>
  new Date(y, m - 1, d, h).toISOString();
const NOW = new Date(2026, 4, 23, 15); // 2026-05-23 15:00 local

describe('computeStreak', () => {
  it('returns 0 with no sessions', () => {
    expect(computeStreak([], NOW)).toBe(0);
  });
  it('counts today only as 1', () => {
    expect(computeStreak([at(2026, 5, 23)], NOW)).toBe(1);
  });
  it('counts consecutive days ending today', () => {
    expect(computeStreak([at(2026, 5, 21), at(2026, 5, 22), at(2026, 5, 23)], NOW)).toBe(3);
  });
  it('collapses multiple sessions on the same day', () => {
    expect(computeStreak([at(2026, 5, 23, 9), at(2026, 5, 23, 20)], NOW)).toBe(1);
  });
  it('stays alive when last practice was yesterday but not yet today', () => {
    expect(computeStreak([at(2026, 5, 21), at(2026, 5, 22)], NOW)).toBe(2);
  });
  it('breaks on a gap', () => {
    expect(computeStreak([at(2026, 5, 19), at(2026, 5, 20), at(2026, 5, 23)], NOW)).toBe(1);
  });
  it('returns 0 when the most recent session is older than yesterday', () => {
    expect(computeStreak([at(2026, 5, 20), at(2026, 5, 21)], NOW)).toBe(0);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd app && npx vitest run src/lib/streak.test.ts`
Expected: FAIL — cannot resolve `./streak`.

- [ ] **Step 3: Write minimal implementation**

```ts
// app/src/lib/streak.ts
/** Local-day key (YYYY-MM-DD in the viewer's timezone) for a timestamp. */
function dayKey(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
}

/** Date offset from `base` by `delta` whole days, at local noon (DST-safe). */
function shiftDay(base: Date, delta: number): Date {
  return new Date(base.getFullYear(), base.getMonth(), base.getDate() + delta, 12);
}

/**
 * Consecutive local days (ending today, or yesterday if nothing yet today) on which
 * the learner started at least one session. Empty input → 0.
 */
export function computeStreak(sessionStartedAt: string[], now: Date = new Date()): number {
  if (sessionStartedAt.length === 0) return 0;
  const days = new Set(sessionStartedAt.map((s) => dayKey(new Date(s))));

  // Anchor on today if practiced today, else yesterday; otherwise the streak is dead.
  let cursor: Date;
  if (days.has(dayKey(now))) cursor = now;
  else if (days.has(dayKey(shiftDay(now, -1)))) cursor = shiftDay(now, -1);
  else return 0;

  let streak = 0;
  while (days.has(dayKey(cursor))) {
    streak += 1;
    cursor = shiftDay(cursor, -1);
  }
  return streak;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd app && npx vitest run src/lib/streak.test.ts`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add app/src/lib/streak.ts app/src/lib/streak.test.ts
git commit -m "feat(app): consecutive-day streak helper"
```

---

### Task 4: `localMidnightISO` time helper

**Files:**
- Modify: `app/src/lib/time.ts`
- Test: `app/src/lib/time.test.ts` (create if absent; otherwise append)

- [ ] **Step 1: Write the failing test**

Create `app/src/lib/time.test.ts` (or append the `describe` block if the file exists):

```ts
// app/src/lib/time.test.ts
import { describe, it, expect } from 'vitest';
import { localMidnightISO } from './time';

describe('localMidnightISO', () => {
  it('returns the UTC ISO of local midnight for the given day', () => {
    const now = new Date(2026, 4, 23, 15, 30); // 2026-05-23 15:30 local
    const iso = localMidnightISO(now);
    const back = new Date(iso);
    expect(back.getFullYear()).toBe(2026);
    expect(back.getMonth()).toBe(4);
    expect(back.getDate()).toBe(23);
    expect(back.getHours()).toBe(0);
    expect(back.getMinutes()).toBe(0);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd app && npx vitest run src/lib/time.test.ts`
Expected: FAIL — `localMidnightISO` is not exported.

- [ ] **Step 3: Write minimal implementation**

Append to `app/src/lib/time.ts`:

```ts
/** UTC ISO string for local midnight of `now`'s calendar day (for "since today" queries). */
export function localMidnightISO(now: Date = new Date()): string {
  return new Date(now.getFullYear(), now.getMonth(), now.getDate(), 0, 0, 0, 0).toISOString();
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd app && npx vitest run src/lib/time.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/src/lib/time.ts app/src/lib/time.test.ts
git commit -m "feat(app): localMidnightISO helper"
```

---

### Task 5: `StreakPill` and `GoalRing` components

**Files:**
- Create: `app/src/components/StreakPill.tsx`
- Create: `app/src/components/GoalRing.tsx`

No new tests (presentational; covered by the build typecheck and dashboard render). Mirrors the
mockup's `ring()` helper using the existing `--accent`/`--accent-2` gradient.

- [ ] **Step 1: Write `StreakPill`**

```tsx
// app/src/components/StreakPill.tsx
/** Gold streak chip. Renders nothing until the learner has a 1+ day streak. */
export function StreakPill({ days }: { days: number }) {
  if (days < 1) return null;
  return <span className="pill pill-gold">🔥 {days} day{days === 1 ? '' : 's'}</span>;
}
```

- [ ] **Step 2: Write `GoalRing`**

```tsx
// app/src/components/GoalRing.tsx
/** Apple-Watch-style progress ring. `done/target` shown in the center; closes as done→target. */
export function GoalRing({ done, target, size = 84 }: { done: number; target: number; size?: number }) {
  const stroke = 9;
  const r = size / 2 - stroke / 2 - 1;
  const c = 2 * Math.PI * r;
  const frac = target > 0 ? Math.min(done / target, 1) : 0;
  const gid = 'goalRingGrad';
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden>
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--seg-togo)" strokeWidth={stroke} />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        stroke={`url(#${gid})`}
        strokeWidth={stroke}
        strokeLinecap="round"
        strokeDasharray={c}
        strokeDashoffset={c * (1 - frac)}
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
      />
      <text x="50%" y="50%" textAnchor="middle" dominantBaseline="central" fontSize={size * 0.26} fontWeight={800} fill="var(--text)">
        {done}/{target}
      </text>
      <defs>
        <linearGradient id={gid} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="var(--accent)" />
          <stop offset="1" stopColor="var(--accent-2)" />
        </linearGradient>
      </defs>
    </svg>
  );
}
```

- [ ] **Step 3: Verify typecheck**

Run: `cd app && npm run build`
Expected: build succeeds (components compile; not yet referenced is fine).

- [ ] **Step 4: Commit**

```bash
git add app/src/components/StreakPill.tsx app/src/components/GoalRing.tsx
git commit -m "feat(app): StreakPill and GoalRing components"
```

---

### Task 6: `DailyGoalCard` component (layout A)

**Files:**
- Create: `app/src/components/DailyGoalCard.tsx`

- [ ] **Step 1: Write `DailyGoalCard`**

```tsx
// app/src/components/DailyGoalCard.tsx
import { GoalRing } from './GoalRing';
import { goalProgress } from '../lib/dailyGoal';
import { sessionXp } from '../lib/xp';

/** Layout A: today's-goal ring as its own card with copy + a reward chip. */
export function DailyGoalCard({ practicedToday }: { practicedToday: number }) {
  const g = goalProgress(practicedToday);
  return (
    <section className="card">
      <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
        <GoalRing done={g.done} target={g.target} />
        <div>
          <div style={{ fontWeight: 800 }}>Today’s goal</div>
          <div className="muted" style={{ fontSize: 13 }}>
            {g.complete
              ? 'Goal complete! 🎉'
              : `Practice ${g.remaining} more sign${g.remaining === 1 ? '' : 's'} to close the ring`}
          </div>
        </div>
        <span className="pill pill-soft" style={{ marginLeft: 'auto' }}>
          +{sessionXp(g.target)} XP
        </span>
      </div>
    </section>
  );
}
```

- [ ] **Step 2: Verify typecheck**

Run: `cd app && npm run build`
Expected: build succeeds.

- [ ] **Step 3: Commit**

```bash
git add app/src/components/DailyGoalCard.tsx
git commit -m "feat(app): DailyGoalCard (layout A)"
```

---

### Task 7: Wire streak pill, goal card, and recent-row XP into the dashboard

**Files:**
- Modify: `app/src/pages/DashboardPage.tsx`

- [ ] **Step 1: Add imports**

At the top of `DashboardPage.tsx`, add to the existing import block:

```tsx
import { localMidnightISO } from '../lib/time';
import { computeStreak } from '../lib/streak';
import { attemptXp } from '../lib/xp';
import { StreakPill } from '../components/StreakPill';
import { DailyGoalCard } from '../components/DailyGoalCard';
```

- [ ] **Step 2: Add state for streak + today's practiced count**

After the existing `const [loading, setLoading] = useState(true);` line, add:

```tsx
  const [streak, setStreak] = useState(0);
  const [practicedToday, setPracticedToday] = useState(0);
```

- [ ] **Step 3: Extend the data load**

Replace the existing `Promise.all([...])` array in the `useEffect` with the version below (adds a
sessions read and a today's-attempts read):

```tsx
      const [signsRes, masteryRes, attemptsRes, sessionsRes, todayRes] = await Promise.all([
        supabase.from('signs').select('id,model_class_index,label,gloss,category').order('model_class_index'),
        supabase
          .from('sign_mastery')
          .select('sign_id,mastery_status,total_attempts,total_passes,first_try_pass_session_count,last_practiced_at,last_result'),
        supabase.from('attempts').select('sign_id,result,created_at').order('created_at', { ascending: false }).limit(8),
        supabase.from('sessions').select('started_at'),
        supabase.from('attempts').select('sign_id').gte('created_at', localMidnightISO()),
      ]);
```

Then, just before `setLoading(false);` at the end of the async block, add:

```tsx
      setStreak(computeStreak(((sessionsRes.data ?? []) as { started_at: string }[]).map((s) => s.started_at)));
      const todayIds = new Set(((todayRes.data ?? []) as { sign_id: string }[]).map((a) => a.sign_id));
      setPracticedToday(todayIds.size);
```

(Streak/goal reads are best-effort; if they error, `data` is null and the values default to 0, leaving the rest of the page intact.)

- [ ] **Step 4: Render the pill and the goal card**

In the header block, change:

```tsx
        <div className="page-header">
          <h1 className="page-title">ASL Practice</h1>
        </div>
```

to:

```tsx
        <div className="page-header">
          <h1 className="page-title">ASL Practice</h1>
          <StreakPill days={streak} />
        </div>
```

Then inside the `{summary && ( <> ... </> )}` fragment, add `<DailyGoalCard>` as the FIRST child,
above `<ProgressCard ... />`:

```tsx
          <DailyGoalCard practicedToday={practicedToday} />
          <ProgressCard summary={summary} />
```

- [ ] **Step 5: Add XP to recent rows**

In `RecentHistory`, inside the `.recent-row`, after the `<span className="when">` line, add an XP span:

```tsx
            <span className="when">{timeAgo(it.at)}</span>
            <span style={{ fontWeight: 700, color: it.result === 'pass' ? 'var(--accent)' : 'var(--text-faint)' }}>
              {attemptXp(it.result) > 0 ? `+${attemptXp(it.result)} XP` : '+0'}
            </span>
```

- [ ] **Step 6: Verify typecheck + full suite**

Run: `cd app && npm run build && npm run test:run`
Expected: build succeeds; all existing tests + the new lib tests pass (102 total).

- [ ] **Step 7: Commit**

```bash
git add app/src/pages/DashboardPage.tsx
git commit -m "feat(app): streak pill, daily-goal card, recent-row XP on dashboard"
```

---

### Task 8: `Confetti` component

**Files:**
- Create: `app/src/components/Confetti.tsx`

Self-contained canvas burst. No dependency. Renders nothing under `prefers-reduced-motion`.

- [ ] **Step 1: Write `Confetti`**

```tsx
// app/src/components/Confetti.tsx
import { useEffect, useRef } from 'react';

const COLORS = ['#7c4dff', '#b14dff', '#ffd54a', '#fb7185', '#22c55e'];

/** One-shot full-screen confetti burst (~1.8s) that unmounts itself. Respects reduced-motion. */
export function Confetti() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return;
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    const W = (canvas.width = window.innerWidth);
    const H = (canvas.height = window.innerHeight);
    const parts = Array.from({ length: 90 }, () => ({
      x: W / 2 + (Math.random() - 0.5) * 120,
      y: H / 3,
      vx: (Math.random() - 0.5) * 9,
      vy: Math.random() * -11 - 4,
      size: 5 + Math.random() * 6,
      color: COLORS[(Math.random() * COLORS.length) | 0],
      rot: Math.random() * Math.PI,
      vr: (Math.random() - 0.5) * 0.3,
    }));

    const start = performance.now();
    let raf = 0;
    const tick = (t: number) => {
      const elapsed = t - start;
      ctx.clearRect(0, 0, W, H);
      for (const p of parts) {
        p.vy += 0.3; // gravity
        p.x += p.vx;
        p.y += p.vy;
        p.rot += p.vr;
        ctx.save();
        ctx.globalAlpha = Math.max(0, 1 - elapsed / 1800);
        ctx.translate(p.x, p.y);
        ctx.rotate(p.rot);
        ctx.fillStyle = p.color;
        ctx.fillRect(-p.size / 2, -p.size / 2, p.size, p.size);
        ctx.restore();
      }
      if (elapsed < 1800) raf = requestAnimationFrame(tick);
      else ctx.clearRect(0, 0, W, H);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden
      style={{ position: 'fixed', inset: 0, width: '100%', height: '100%', pointerEvents: 'none', zIndex: 50 }}
    />
  );
}
```

- [ ] **Step 2: Verify typecheck**

Run: `cd app && npm run build`
Expected: build succeeds.

- [ ] **Step 3: Commit**

```bash
git add app/src/components/Confetti.tsx
git commit -m "feat(app): self-contained confetti effect"
```

---

### Task 9: `SessionComplete` screen + wire into PracticePage

**Files:**
- Create: `app/src/components/SessionComplete.tsx`
- Modify: `app/src/pages/PracticePage.tsx`

- [ ] **Step 1: Write `SessionComplete`**

```tsx
// app/src/components/SessionComplete.tsx
import { Link } from 'react-router-dom';
import { Confetti } from './Confetti';
import { sessionXp } from '../lib/xp';

/** Celebration screen shown when a practice deck is finished. */
export function SessionComplete({
  passed,
  completed,
  streak,
  goalClosed,
}: {
  passed: number;
  completed: number;
  streak: number;
  goalClosed: boolean;
}) {
  return (
    <main className="page" style={{ textAlign: 'center' }}>
      <Confetti />
      <h1 className="page-title" style={{ fontSize: 28 }}>
        Session complete 🎉
      </h1>
      <p className="spacer-top" style={{ fontSize: 18 }}>
        Passed <strong>{passed}</strong> of <strong>{completed}</strong> words.
      </p>
      <p className="spacer-top" style={{ fontSize: 18 }}>
        <span className="pill pill-soft">+{sessionXp(passed)} XP</span>
      </p>
      {streak >= 1 && (
        <p className="spacer-top">
          <span className="pill pill-gold">🔥 {streak} day{streak === 1 ? '' : 's'} streak</span>
        </p>
      )}
      {goalClosed && <p className="spacer-top muted">You closed today’s goal ring! 🟣</p>}
      <div className="spacer-top">
        <Link to="/" className="btn btn-primary">
          ← Back to dashboard
        </Link>
      </div>
    </main>
  );
}
```

- [ ] **Step 2: Add imports + finish-time state to PracticePage**

In `app/src/pages/PracticePage.tsx`, add to the import block:

```tsx
import { localMidnightISO } from '../lib/time';
import { computeStreak } from '../lib/streak';
import { goalProgress } from '../lib/dailyGoal';
import { SessionComplete } from '../components/SessionComplete';
```

After the existing `const [stats, setStats] = useState({ completed: 0, passed: 0 });` line, add:

```tsx
  const [finishInfo, setFinishInfo] = useState({ streak: 0, goalClosed: false });
```

- [ ] **Step 3: Compute streak + goal in `finish`**

Replace the existing `finish` function body with:

```tsx
  async function finish() {
    if (sessionId && userId) {
      await supabase.from('sessions').update({ ended_at: new Date().toISOString() }).eq('id', sessionId);
      const [sessionsRes, todayRes] = await Promise.all([
        supabase.from('sessions').select('started_at'),
        supabase.from('attempts').select('sign_id').gte('created_at', localMidnightISO()),
      ]);
      const streak = computeStreak(((sessionsRes.data ?? []) as { started_at: string }[]).map((s) => s.started_at));
      const practiced = new Set(((todayRes.data ?? []) as { sign_id: string }[]).map((a) => a.sign_id)).size;
      setFinishInfo({ streak, goalClosed: goalProgress(practiced).complete });
    }
    setPhase('finished');
  }
```

- [ ] **Step 4: Render `SessionComplete` for the finished phase**

Replace the entire `if (phase === 'finished') return ( ... );` block with:

```tsx
  if (phase === 'finished')
    return (
      <SessionComplete
        passed={stats.passed}
        completed={stats.completed}
        streak={finishInfo.streak}
        goalClosed={finishInfo.goalClosed}
      />
    );
```

- [ ] **Step 5: Verify typecheck + full suite**

Run: `cd app && npm run build && npm run test:run`
Expected: build succeeds; all tests pass (102 total — additive, none changed).

- [ ] **Step 6: Commit**

```bash
git add app/src/components/SessionComplete.tsx app/src/pages/PracticePage.tsx
git commit -m "feat(app): end-of-session celebration with confetti, XP, streak"
```

---

### Task 10: Final verification

- [ ] **Step 1: Full build + test**

Run: `cd app && npm run build && npm run test:run`
Expected: build clean; 102 tests pass.

- [ ] **Step 2: Manual smoke (dev server)**

Run: `cd app && npm run dev`, open the printed localhost URL.
Check: dashboard shows the 🔥 streak pill (when ≥1 day) and the "Today's goal" ring card at the top;
recent rows show `+10 XP` / `+0`; finishing a practice deck shows confetti + session XP + streak.
(Requires Anonymous sign-ins enabled in Supabase to load data; without it the page still renders.)

- [ ] **Step 3: Lint**

Run: `cd app && npm run lint`
Expected: no new errors.
