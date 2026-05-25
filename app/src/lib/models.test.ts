import { describe, it, expect } from 'vitest';
import { MODELS, DEFAULT_MODEL_ID, getModelById, selectableModels } from './models';

describe('model registry', () => {
  it('includes the from-scratch own model and a baseline comparison model', () => {
    expect(MODELS.some((m) => m.kind === 'own')).toBe(true);
    expect(MODELS.some((m) => m.kind === 'baseline')).toBe(true);
  });

  it('offers two own model versions (v1 and v2)', () => {
    expect(MODELS.filter((m) => m.kind === 'own')).toHaveLength(2);
  });

  it('defaults to the own (pilot) model', () => {
    expect(getModelById(DEFAULT_MODEL_ID).kind).toBe('own');
  });

  it('looks a model up by id', () => {
    const baseline = MODELS.find((m) => m.kind === 'baseline')!;
    expect(getModelById(baseline.id).id).toBe(baseline.id);
  });

  it('falls back to the default for unknown or missing ids', () => {
    expect(getModelById('nope').id).toBe(DEFAULT_MODEL_ID);
    expect(getModelById(null).id).toBe(DEFAULT_MODEL_ID);
  });
});

describe('selectableModels (Req 7 gating)', () => {
  it('excludes the pretrained baseline when dev tools are off', () => {
    const ids = selectableModels(false).map((m) => m.id);
    expect(ids).not.toContain('baseline');
    expect(selectableModels(false).every((m) => m.kind === 'own')).toBe(true);
  });
  it('includes the baseline when dev tools are on', () => {
    expect(selectableModels(true).map((m) => m.id)).toContain('baseline');
  });
  it('refuses a stale baseline id when baseline is not selectable', () => {
    expect(getModelById('baseline', false).id).toBe(DEFAULT_MODEL_ID);
  });
});
