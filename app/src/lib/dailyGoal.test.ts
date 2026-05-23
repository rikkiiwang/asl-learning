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
