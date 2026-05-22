import { describe, it, expect } from 'vitest';
import { buildHint } from './hints';

describe('buildHint', () => {
  it('wrong_sign: names the confused sign and the prompted target', () => {
    const h = buildHint({ failReason: 'wrong_sign', promptedLabel: 'MILK', competitorLabel: 'COW' });
    expect(h).toContain('COW');
    expect(h).toContain('MILK');
  });

  it('uses a teachable cue from the prompted sign metadata when available', () => {
    const h = buildHint({
      failReason: 'wrong_sign',
      promptedLabel: 'MILK',
      competitorLabel: 'COW',
      promptedHints: { movement: 'squeeze the fist open and closed' },
    });
    expect(h).toContain('squeeze the fist open and closed');
  });

  it('prefers movement over handshape when both are present', () => {
    const h = buildHint({
      failReason: 'ambiguous',
      promptedLabel: 'MILK',
      promptedHints: { handshape: 'C-hand', movement: 'repeated squeeze' },
    });
    expect(h).toContain('repeated squeeze');
    expect(h).not.toContain('C-hand');
  });

  it('low_confidence: gives a clarity/framing nudge mentioning the sign', () => {
    const h = buildHint({ failReason: 'low_confidence', promptedLabel: 'BOOK' });
    expect(h).toContain('BOOK');
    expect(h.length).toBeGreaterThan(0);
  });

  it('returns a non-empty hint even with no metadata or competitor', () => {
    for (const failReason of ['wrong_sign', 'low_confidence', 'ambiguous'] as const) {
      expect(buildHint({ failReason, promptedLabel: 'CAT' }).length).toBeGreaterThan(0);
    }
  });
});
