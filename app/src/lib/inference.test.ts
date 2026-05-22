import { describe, it, expect } from 'vitest';
import { softmax, topK } from './inference';

describe('softmax', () => {
  it('is uniform for equal logits', () => {
    expect(softmax([0, 0])).toEqual([0.5, 0.5]);
  });

  it('sums to 1', () => {
    const p = softmax([1, 2, 3]);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
  });

  it('is monotonic in the logits', () => {
    const p = softmax([1, 2, 3]);
    expect(p[2]).toBeGreaterThan(p[1]);
    expect(p[1]).toBeGreaterThan(p[0]);
  });

  it('is numerically stable for large logits (no overflow to NaN)', () => {
    const p = softmax([1000, 1000]);
    expect(p[0]).toBeCloseTo(0.5, 6);
    expect(Number.isNaN(p[0])).toBe(false);
  });
});

describe('topK', () => {
  it('returns the highest-probability entries, sorted descending', () => {
    expect(topK([0.1, 0.7, 0.2], 2)).toEqual([
      { index: 1, prob: 0.7 },
      { index: 2, prob: 0.2 },
    ]);
  });

  it('returns all entries (sorted) when k exceeds length', () => {
    const r = topK([0.2, 0.8], 5);
    expect(r.map((x) => x.index)).toEqual([1, 0]);
  });
});
