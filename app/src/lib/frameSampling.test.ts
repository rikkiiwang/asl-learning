import { describe, it, expect } from 'vitest';
import { sampleFrameIndices } from './frameSampling';
import fixture from './__fixtures__/frameSampling.json';

// Golden vectors generated from the model's own sample_indices (see fixture.source).
// This is the parity contract for the train↔inference invariant: the app's TS
// resampler must reproduce the model's Python output EXACTLY (incl. numpy's
// round-half-to-even), or live accuracy silently degrades.
describe('sampleFrameIndices — parity with the model (golden vectors)', () => {
  for (const c of fixture.cases) {
    it(`matches model output for n_frames=${c.n_frames}, k=${c.k}`, () => {
      expect(sampleFrameIndices(c.n_frames, c.k)).toEqual(c.indices);
    });
  }

  it('defaults k to 16', () => {
    const c = fixture.cases.find((x) => x.n_frames === 24)!;
    expect(sampleFrameIndices(24)).toEqual(c.indices);
  });

  it('always returns exactly k indices', () => {
    for (const n of [0, 1, 5, 16, 50]) {
      expect(sampleFrameIndices(n, 16)).toHaveLength(16);
    }
  });
});
