import { describe, it, expect } from 'vitest';
import { signingSavvyUrl, groupByCategory, filterSigns } from './signReference';
import type { Sign } from './types';

const mk = (label: string, category: string | null, gloss = label): Sign => ({
  id: label.toLowerCase(),
  model_class_index: 0,
  label,
  gloss,
  category,
});

describe('signingSavvyUrl', () => {
  it('lowercases and URL-encodes the label', () => {
    expect(signingSavvyUrl('APPLE')).toBe('https://www.signingsavvy.com/search/apple');
    expect(signingSavvyUrl('Ice Cream')).toBe('https://www.signingsavvy.com/search/ice%20cream');
  });
});

describe('groupByCategory', () => {
  it('groups by category, sorts categories alphabetically, uncategorized last', () => {
    const groups = groupByCategory([
      mk('APPLE', 'Food'),
      mk('BLUE', 'Colors'),
      mk('BANANA', 'Food'),
      mk('XYZ', null),
    ]);
    expect(groups.map((g) => g.category)).toEqual(['Colors', 'Food', 'Other']);
    expect(groups[1].signs.map((s) => s.label)).toEqual(['APPLE', 'BANANA']); // input order kept
    expect(groups[2].signs[0].label).toBe('XYZ'); // null -> "Other", last
  });
});

describe('filterSigns', () => {
  const signs = [mk('CAT', 'Animals'), mk('DOG', 'Animals'), mk('HAPPY', 'Feelings', 'glad')];
  it('returns all on empty query', () => {
    expect(filterSigns(signs, '  ')).toHaveLength(3);
  });
  it('matches label case-insensitively', () => {
    expect(filterSigns(signs, 'ca').map((s) => s.label)).toEqual(['CAT']);
  });
  it('matches gloss too', () => {
    expect(filterSigns(signs, 'glad').map((s) => s.label)).toEqual(['HAPPY']);
  });
});
