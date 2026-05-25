import { describe, it, expect } from 'vitest';
import { computeStreak } from './streak';

// Build a local-time ISO string for a given Y-M-D + hour (no Z → parsed as local).
const at = (y: number, m: number, d: number, h = 12) => new Date(y, m - 1, d, h).toISOString();
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
