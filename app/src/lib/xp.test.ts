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
