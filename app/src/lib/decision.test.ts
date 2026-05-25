import { describe, it, expect } from 'vitest';
import { decidePassFail } from './decision';

describe('decidePassFail', () => {
  const thr = { threshold: 0.6, margin: 0.15 };

  it('passes when prompted is argmax, confident, and clear of the runner-up', () => {
    const d = decidePassFail([0.8, 0.1, 0.1], 0, thr);
    expect(d.pass).toBe(true);
    expect(d.failReason).toBeNull();
    expect(d.predictedIndex).toBe(0);
    expect(d.promptedProb).toBeCloseTo(0.8, 6);
    expect(d.margin).toBeCloseTo(0.7, 6);
  });

  it('fails as wrong_sign when a different class wins', () => {
    const d = decidePassFail([0.2, 0.7, 0.1], 0, thr);
    expect(d.pass).toBe(false);
    expect(d.failReason).toBe('wrong_sign');
    expect(d.predictedIndex).toBe(1);
  });

  it('fails as low_confidence when prompted wins but below threshold', () => {
    const d = decidePassFail([0.5, 0.45, 0.05], 0, thr);
    expect(d.pass).toBe(false);
    expect(d.failReason).toBe('low_confidence');
    expect(d.predictedIndex).toBe(0);
  });

  it('fails as ambiguous when prompted wins, is confident, but the margin is thin', () => {
    const d = decidePassFail([0.55, 0.45, 0.0], 0, { threshold: 0.5, margin: 0.15 });
    expect(d.pass).toBe(false);
    expect(d.failReason).toBe('ambiguous');
  });

  it('reports the runner-up margin', () => {
    const d = decidePassFail([0.6, 0.3, 0.1], 0, thr);
    expect(d.margin).toBeCloseTo(0.3, 6);
  });

  // Default policy (DEFAULT_THRESHOLDS: passTopN=3, gating off): pass iff the
  // prompted sign is within the model's top-3, regardless of confidence/margin.
  it('default top-3 policy: passes when the prompted sign is the top-1', () => {
    const d = decidePassFail([0.3, 0.28, 0.1], 0);
    expect(d.pass).toBe(true);
    expect(d.failReason).toBeNull();
  });

  it('default top-3 policy: passes when the prompted sign is 3rd (within top-3)', () => {
    // prompted = index 2 (prob 0.20) ranks 3rd -> pass; predicted/top-1 is index 0.
    const d = decidePassFail([0.4, 0.35, 0.2, 0.05], 2);
    expect(d.pass).toBe(true);
    expect(d.predictedIndex).toBe(0);
    expect(d.failReason).toBeNull();
  });

  it('default top-3 policy: fails when the prompted sign is outside the top-3', () => {
    // prompted = index 3 (prob 0.10) ranks 4th -> fail.
    const d = decidePassFail([0.4, 0.3, 0.18, 0.1], 3);
    expect(d.pass).toBe(false);
    expect(d.failReason).toBe('wrong_sign');
    expect(d.predictedIndex).toBe(0);
  });
});
