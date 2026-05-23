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
