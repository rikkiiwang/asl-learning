// Uniform frame-index resampling — MUST stay byte-for-byte identical to the
// model's Python `sample_indices` (model/src/asl/preprocess.py). The model uses
// numpy `linspace(...).round()`, which rounds halves to even — JS `Math.round`
// rounds halves up, so we replicate banker's rounding here. Guarded by the
// golden-vector parity test (frameSampling.test.ts).

function roundHalfToEven(x: number): number {
  const floor = Math.floor(x);
  const frac = x - floor;
  if (frac < 0.5) return floor;
  if (frac > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

/** Pick `k` frame indices spanning a clip of `nFrames`, matching the model. */
export function sampleFrameIndices(nFrames: number, k = 16): number[] {
  if (nFrames <= 0) return new Array(k).fill(0);

  if (nFrames >= k) {
    // numpy linspace(0, nFrames-1, k): y[i] = i*step, with the last point pinned to the endpoint.
    const step = (nFrames - 1) / (k - 1);
    const out: number[] = [];
    for (let i = 0; i < k; i++) {
      const x = i === k - 1 ? nFrames - 1 : i * step;
      out.push(roundHalfToEven(x));
    }
    return out;
  }

  // Fewer frames than needed: take all, then repeat the last to pad.
  const out = Array.from({ length: nFrames }, (_, i) => i);
  while (out.length < k) out.push(nFrames - 1);
  return out;
}
