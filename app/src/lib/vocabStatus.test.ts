import { describe, it, expect } from 'vitest';
import { statusStyle } from './vocabStatus';

describe('statusStyle', () => {
  it('maps mastered', () => {
    expect(statusStyle('mastered')).toEqual({ color: 'var(--seg-mastered)', label: 'Mastered' });
  });
  it('maps learning', () => {
    expect(statusStyle('learning')).toEqual({ color: 'var(--seg-learning)', label: 'Learning' });
  });
  it('maps not_started', () => {
    expect(statusStyle('not_started')).toEqual({ color: 'var(--seg-togo)', label: 'Not started' });
  });
});
