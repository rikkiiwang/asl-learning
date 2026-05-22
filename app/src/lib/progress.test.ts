import { describe, it, expect } from 'vitest';
import { summarizeProgress, progressBarSegments } from './progress';
import type { Sign, SignMastery, ProgressSummary } from './types';

function sign(id: string, idx: number): Sign {
  return { id, model_class_index: idx, label: `S${idx}`, gloss: `S${idx}`, category: null };
}
function mastery(sign_id: string, status: SignMastery['mastery_status']): SignMastery {
  return {
    sign_id,
    mastery_status: status,
    total_attempts: 1,
    total_passes: status === 'not_started' ? 0 : 1,
    first_try_pass_session_count: status === 'mastered' ? 2 : 0,
    last_practiced_at: null,
    last_result: null,
  };
}

const signs: Sign[] = [sign('a', 0), sign('b', 1), sign('c', 2), sign('d', 3)];

describe('summarizeProgress', () => {
  it('counts signs with no mastery row as not_started', () => {
    const s = summarizeProgress(signs, []);
    expect(s).toEqual({ total: 4, mastered: 0, learning: 0, notStarted: 4, percentComplete: 0 });
  });

  it('counts mastered / learning / not_started from mastery rows', () => {
    const m = [mastery('a', 'mastered'), mastery('b', 'mastered'), mastery('c', 'learning')];
    const s = summarizeProgress(signs, m);
    expect(s.mastered).toBe(2);
    expect(s.learning).toBe(1);
    expect(s.notStarted).toBe(1); // 'd' has no row
  });

  it('percentComplete is mastered/total rounded to a whole percent', () => {
    const m = [mastery('a', 'mastered')];
    // 1/4 = 25
    expect(summarizeProgress(signs, m).percentComplete).toBe(25);
  });

  it('rounds percentComplete (1 of 3 mastered -> 33)', () => {
    const three = [sign('a', 0), sign('b', 1), sign('c', 2)];
    expect(summarizeProgress(three, [mastery('a', 'mastered')]).percentComplete).toBe(33);
  });

  it('handles an empty catalog without dividing by zero', () => {
    expect(summarizeProgress([], [])).toEqual({
      total: 0,
      mastered: 0,
      learning: 0,
      notStarted: 0,
      percentComplete: 0,
    });
  });

  it('ignores mastery rows for signs not in the catalog', () => {
    const s = summarizeProgress(signs, [mastery('zzz', 'mastered')]);
    expect(s.mastered).toBe(0);
    expect(s.notStarted).toBe(4);
  });
});

describe('progressBarSegments', () => {
  const summary = (over: Partial<ProgressSummary>): ProgressSummary => ({
    total: 0, mastered: 0, learning: 0, notStarted: 0, percentComplete: 0, ...over,
  });

  it('is all zero for an empty catalog', () => {
    expect(progressBarSegments(summary({}))).toEqual({ masteredPct: 0, learningPct: 0, notStartedPct: 0 });
  });

  it('splits proportionally and sums to 100', () => {
    const seg = progressBarSegments(summary({ total: 4, mastered: 2, learning: 1, notStarted: 1 }));
    expect(seg).toEqual({ masteredPct: 50, learningPct: 25, notStartedPct: 25 });
    expect(seg.masteredPct + seg.learningPct + seg.notStartedPct).toBe(100);
  });

  it('absorbs rounding into not-started so segments still sum to 100', () => {
    const seg = progressBarSegments(summary({ total: 3, mastered: 1, learning: 0, notStarted: 2 }));
    expect(seg.masteredPct).toBe(33);
    expect(seg.masteredPct + seg.learningPct + seg.notStartedPct).toBe(100);
  });
});
