import type { Decision } from './decision';

export type WordOutcome = 'advance' | 'retry' | 'reveal';

/**
 * Resolve a word after an attempt (spec §3): pass → advance; fail with attempts
 * remaining → retry; final failed attempt → reveal the reference, then advance.
 */
export function wordOutcome(attemptNumber: number, pass: boolean, maxAttempts = 3): WordOutcome {
  if (pass) return 'advance';
  return attemptNumber < maxAttempts ? 'retry' : 'reveal';
}

export interface AttemptRow {
  user_id: string;
  session_id: string;
  sign_id: string;
  attempt_number: number;
  is_first_try: boolean;
  result: 'pass' | 'fail';
  predicted_sign_id: string | null;
  confidence: number;
  margin: number;
}

/** Assemble an `attempts` insert payload from a recognition decision. */
export function buildAttemptRow(p: {
  userId: string;
  sessionId: string;
  signId: string;
  attemptNumber: number;
  decision: Decision;
  predictedSignId: string | null;
}): AttemptRow {
  return {
    user_id: p.userId,
    session_id: p.sessionId,
    sign_id: p.signId,
    attempt_number: p.attemptNumber,
    is_first_try: p.attemptNumber === 1,
    result: p.decision.pass ? 'pass' : 'fail',
    predicted_sign_id: p.predictedSignId,
    confidence: p.decision.promptedProb,
    margin: p.decision.margin,
  };
}
