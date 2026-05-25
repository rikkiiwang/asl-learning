/** Compact relative time ("just now", "5m ago", "2h ago", "3d ago"); a date past a week. */
export function timeAgo(input: string | Date, now: Date = new Date()): string {
  const then = typeof input === 'string' ? new Date(input) : input;
  const sec = Math.floor((now.getTime() - then.getTime()) / 1000);
  if (sec < 60) return 'just now';
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}h ago`;
  const day = Math.floor(hr / 24);
  if (day < 7) return `${day}d ago`;
  return then.toLocaleDateString();
}

/** UTC ISO string for local midnight of `now`'s calendar day (for "since today" queries). */
export function localMidnightISO(now: Date = new Date()): string {
  return new Date(now.getFullYear(), now.getMonth(), now.getDate(), 0, 0, 0, 0).toISOString();
}
