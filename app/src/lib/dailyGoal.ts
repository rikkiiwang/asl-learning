/** How many distinct signs a learner aims to practice each day. Tune freely. */
export const DAILY_GOAL = 5;

export interface GoalProgress {
  done: number; // clamped to [0, target]
  target: number;
  remaining: number;
  complete: boolean;
}

/** Ring math: how today's practice count maps onto the daily target. */
export function goalProgress(practicedToday: number, target: number = DAILY_GOAL): GoalProgress {
  const done = Math.max(0, Math.min(practicedToday, target));
  return { done, target, remaining: target - done, complete: done >= target };
}
