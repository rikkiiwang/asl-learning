import { describe, it, expect } from 'vitest';
import { StubRecognizer, OnnxRecognizer, createRecognizer } from './recognizer';
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
  it('uses the ONNX recognizer for a model with a configured artifact URL (own-v1)', () => {
    const ownV1 = MODELS.find((m) => m.id === 'own-v1')!;
    expect(createRecognizer(ownV1, 75)).toBeInstanceOf(OnnxRecognizer);
  });

  it('falls back to a stub (correct class count) for a model with no configured URL', async () => {
    const noUrl = MODELS.find((m) => m.id === 'own-v2')!; // no artifact published yet
    const logits = await createRecognizer(noUrl, 75).recognize(new Float32Array(10));
    expect(logits.length).toBe(75);
  });
});
