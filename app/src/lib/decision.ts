export type FailReason = 'wrong_sign' | 'low_confidence' | 'ambiguous';

export interface Thresholds {
  threshold: number; // min P(prompted) to pass
  margin: number; // min (P1 - P2) to pass
  passTopN?: number; // prompted must rank within the top-N predictions (default 1)
}

export interface Decision {
  pass: boolean;
  predictedIndex: number;
  promptedProb: number;
  topProb: number;
  margin: number; // P1 - P2
  failReason: FailReason | null;
}

// Pass policy: the prompted sign must rank within the model's TOP-3 predictions.
// Rationale: on held-out signers the model puts the right sign in its top-3 ~86%
// of the time (vs ~73% top-1) — a fairer bar for a learning app than exact top-1.
// Confidence/margin gating stays disabled (0/0): the label-smoothed 75-way model
// sits below a 0.6 bar even when right. Re-tighten via calibrated thresholds in
// meta.json once live confidence is trustworthy.
export const DEFAULT_THRESHOLDS: Thresholds = { threshold: 0, margin: 0, passTopN: 3 };

/**
 * Pass/fail: pass iff the prompted class ranks within the top `passTopN`
 * predictions (default 1), its probability clears `threshold`, AND it leads the
 * runner-up by `margin`. Otherwise classify why it failed, to drive a targeted hint.
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

  // Rank of the prompted class = how many classes score strictly higher (0 = top-1).
  let rank = 0;
  for (let i = 0; i < probs.length; i++) if (probs[i] > promptedProb) rank++;
  const topN = thr.passTopN ?? 1;

  let failReason: FailReason | null = null;
  if (rank >= topN) failReason = 'wrong_sign';
  else if (promptedProb < thr.threshold) failReason = 'low_confidence';
  else if (margin < thr.margin) failReason = 'ambiguous';

  return { pass: failReason === null, predictedIndex: topIdx, promptedProb, topProb: top1, margin, failReason };
}
