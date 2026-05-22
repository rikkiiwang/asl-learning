export type FailReason = 'wrong_sign' | 'low_confidence' | 'ambiguous';

export interface Thresholds {
  threshold: number; // min P(prompted) to pass
  margin: number; // min (P1 - P2) to pass
}

export interface Decision {
  pass: boolean;
  predictedIndex: number;
  promptedProb: number;
  topProb: number;
  margin: number; // P1 - P2
  failReason: FailReason | null;
}

// Placeholder until per-class thresholds arrive in meta.json (M5/M7).
// Conservative: bias against false passes (PRD Req 9).
export const DEFAULT_THRESHOLDS: Thresholds = { threshold: 0.6, margin: 0.15 };

/**
 * Conservative pass/fail (spec §4): pass iff the prompted class is the argmax,
 * its probability clears `threshold`, AND it leads the runner-up by `margin`.
 * Otherwise classify why it failed, to drive a targeted hint.
 */
export function decidePassFail(
  probs: ArrayLike<number>,
  promptedIndex: number,
  thr: Thresholds = DEFAULT_THRESHOLDS,
): Decision {
  let topIdx = 0;
  let top1 = -Infinity;
  let top2 = -Infinity;
  for (let i = 0; i < probs.length; i++) {
    const p = probs[i];
    if (p > top1) {
      top2 = top1;
      top1 = p;
      topIdx = i;
    } else if (p > top2) {
      top2 = p;
    }
  }

  const promptedProb = probs[promptedIndex] ?? 0;
  const margin = top1 - top2;

  let failReason: FailReason | null = null;
  if (topIdx !== promptedIndex) failReason = 'wrong_sign';
  else if (promptedProb < thr.threshold) failReason = 'low_confidence';
  else if (margin < thr.margin) failReason = 'ambiguous';

  return { pass: failReason === null, predictedIndex: topIdx, promptedProb, topProb: top1, margin, failReason };
}
