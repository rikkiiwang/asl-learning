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
