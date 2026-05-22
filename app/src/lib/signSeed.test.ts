import { describe, it, expect } from 'vitest';
import { buildSignSeed } from './signSeed';
import type { Manifest } from './types';

const manifest: Manifest = {
  version: 'test',
  num_classes: 3,
  labels: ['EAT', 'DOG', 'RED'],
  signs: [
    { label: 'RED', gloss: 'RED', label_idx: 2, category: 'colors', counts: { train: 20 } },
    { label: 'EAT', gloss: 'EAT1', label_idx: 0, category: 'food_drink' },
    { label: 'DOG', gloss: 'DOG1', label_idx: 1, category: 'animals' },
  ],
};

describe('buildSignSeed', () => {
  it('maps label_idx to model_class_index and preserves label/gloss/category', () => {
    const rows = buildSignSeed(manifest);
    const eat = rows.find((r) => r.label === 'EAT')!;
    expect(eat).toEqual({ model_class_index: 0, label: 'EAT', gloss: 'EAT1', category: 'food_drink' });
  });

  it('returns rows sorted by model_class_index', () => {
    const rows = buildSignSeed(manifest);
    expect(rows.map((r) => r.model_class_index)).toEqual([0, 1, 2]);
    expect(rows.map((r) => r.label)).toEqual(['EAT', 'DOG', 'RED']);
  });

  it('returns one row per sign', () => {
    expect(buildSignSeed(manifest)).toHaveLength(3);
  });

  it('throws on duplicate class indices', () => {
    const bad: Manifest = {
      ...manifest,
      signs: [
        { label: 'A', gloss: 'A', label_idx: 0, category: null },
        { label: 'B', gloss: 'B', label_idx: 0, category: null },
      ],
    };
    expect(() => buildSignSeed(bad)).toThrow(/duplicate/i);
  });

  it('throws when indices are not contiguous from 0 (would break softmax mapping)', () => {
    const bad: Manifest = {
      ...manifest,
      signs: [
        { label: 'A', gloss: 'A', label_idx: 0, category: null },
        { label: 'B', gloss: 'B', label_idx: 2, category: null },
      ],
    };
    expect(() => buildSignSeed(bad)).toThrow(/contiguous/i);
  });
});
