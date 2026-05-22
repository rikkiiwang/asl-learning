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
});
