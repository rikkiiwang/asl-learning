import { describe, it, expect } from 'vitest';
import { StubRecognizer, createRecognizer } from './recognizer';
import { MODELS } from '../lib/models';

describe('StubRecognizer', () => {
  it('returns logits of length numClasses', async () => {
    const logits = await new StubRecognizer(75).recognize(new Float32Array(200));
    expect(logits.length).toBe(75);
  });

  it('is deterministic for the same input', async () => {
    const r = new StubRecognizer(75, 3);
    const input = new Float32Array([1, 2, 3, 4, 5]);
    expect(Array.from(await r.recognize(input))).toEqual(Array.from(await r.recognize(input)));
  });

  it('different model salts can produce different top classes', async () => {
    // Not guaranteed for every input, but the salt must influence the output.
    const input = new Float32Array(Array.from({ length: 50 }, (_, i) => Math.sin(i)));
    const a = await new StubRecognizer(75, 1).recognize(input);
    const b = await new StubRecognizer(75, 999).recognize(input);
    expect(Array.from(a)).not.toEqual(Array.from(b));
  });
});

describe('createRecognizer', () => {
  it('falls back to a stub (correct class count) while no model URLs are configured', async () => {
    const logits = await createRecognizer(MODELS[0], 75).recognize(new Float32Array(10));
    expect(logits.length).toBe(75);
  });
});
