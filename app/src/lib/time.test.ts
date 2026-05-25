import { describe, it, expect } from 'vitest';
import { timeAgo, localMidnightISO } from './time';

const now = new Date('2026-05-22T12:00:00Z');
const ago = (ms: number) => new Date(now.getTime() - ms);

describe('timeAgo', () => {
  it('says "just now" under a minute', () => {
    expect(timeAgo(ago(30 * 1000), now)).toBe('just now');
  });
  it('reports minutes', () => {
    expect(timeAgo(ago(5 * 60 * 1000), now)).toBe('5m ago');
  });
  it('reports hours', () => {
    expect(timeAgo(ago(2 * 60 * 60 * 1000), now)).toBe('2h ago');
  });
  it('reports days', () => {
    expect(timeAgo(ago(3 * 24 * 60 * 60 * 1000), now)).toBe('3d ago');
  });
  it('falls back to a date past a week', () => {
    expect(timeAgo(ago(30 * 24 * 60 * 60 * 1000), now)).not.toContain('ago');
  });
  it('accepts ISO strings', () => {
    expect(timeAgo(ago(60 * 1000).toISOString(), now)).toBe('1m ago');
  });
});

describe('localMidnightISO', () => {
  it('returns the UTC ISO of local midnight for the given day', () => {
    const day = new Date(2026, 4, 23, 15, 30); // 2026-05-23 15:30 local
    const back = new Date(localMidnightISO(day));
    expect(back.getFullYear()).toBe(2026);
    expect(back.getMonth()).toBe(4);
    expect(back.getDate()).toBe(23);
    expect(back.getHours()).toBe(0);
    expect(back.getMinutes()).toBe(0);
  });
});
