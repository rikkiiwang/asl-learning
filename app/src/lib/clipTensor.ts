// Build the model input tensor from sampled clip frames.
// Layout NFCHW with N=1 implicit: length F*C*H*W, ordered frame → channel → row → col.
// Normalization mirrors training: (pixel/255 - mean) / std, per channel. The mean/std
// come from the model's meta.json (norm.json) — injected so this stays pure/testable.

export function framesToTensor(
  frames: Uint8ClampedArray[],
  size: number,
  mean: readonly number[],
  std: readonly number[],
): Float32Array {
  const F = frames.length;
  const C = 3;
  const HW = size * size;
  const out = new Float32Array(F * C * HW);

  for (let f = 0; f < F; f++) {
    const px = frames[f];
    for (let p = 0; p < HW; p++) {
      for (let c = 0; c < C; c++) {
        const v = px[p * 4 + c] / 255;
        out[(f * C + c) * HW + p] = (v - mean[c]) / std[c];
      }
    }
  }
  return out;
}
