import { describe, it, expect } from 'vitest';
import { wordOutcome, buildAttemptRow } from './session';
import type { Decision } from './decision';

const passDecision: Decision = { pass: true, predictedIndex: 3, promptedProb: 0.8, topProb: 0.8, margin: 0.5, failReason: null };
const failDecision: Decision = { pass: false, predictedIndex: 7, promptedProb: 0.2, topProb: 0.6, margin: 0.3, failReason: 'wrong_sign' };

describe('wordOutcome', () => {
  it('advances on a pass', () => {
    expect(wordOutcome(1, true)).toBe('advance');
  });
  it('retries on a fail while attempts remain', () => {
    expect(wordOutcome(1, false)).toBe('retry');
    expect(wordOutcome(2, false)).toBe('retry');
  });
  it('reveals (then advances) on the final failed attempt', () => {
    expect(wordOutcome(3, false)).toBe('reveal');
  });
});

describe('buildAttemptRow', () => {
  const base = { userId: 'u1', sessionId: 'sess1', signId: 'sign3', predictedSignId: 'sign7' };

  it('marks attempt 1 as first try and maps a pass', () => {
    const row = buildAttemptRow({ ...base, attemptNumber: 1, decision: passDecision, predictedSignId: 'sign3' });
    expect(row.is_first_try).toBe(true);
    expect(row.result).toBe('pass');
    expect(row.attempt_number).toBe(1);
    expect(row.confidence).toBeCloseTo(0.8, 6);
    expect(row.margin).toBeCloseTo(0.5, 6);
  });

  it('marks later attempts as not first try and maps a fail', () => {
    const row = buildAttemptRow({ ...base, attemptNumber: 2, decision: failDecision });
    expect(row.is_first_try).toBe(false);
    expect(row.result).toBe('fail');
    expect(row.predicted_sign_id).toBe('sign7');
    expect(row.sign_id).toBe('sign3');
    expect(row.user_id).toBe('u1');
    expect(row.session_id).toBe('sess1');
  });
});
