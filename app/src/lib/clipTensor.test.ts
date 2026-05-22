import { describe, it, expect } from 'vitest';
import { framesToTensor } from './clipTensor';

// Build a single-pixel (size=1) RGBA frame from [r,g,b].
function frame(r: number, g: number, b: number): Uint8ClampedArray {
  return new Uint8ClampedArray([r, g, b, 255]);
}

describe('framesToTensor', () => {
  it('scales pixels to 0–1 and normalizes by per-channel mean/std', () => {
    const t = framesToTensor([frame(255, 0, 128)], 1, [0, 0, 0], [1, 1, 1]);
    expect(t.length).toBe(3); // F=1, C=3, HW=1
    expect(t[0]).toBeCloseTo(1, 5);
    expect(t[1]).toBeCloseTo(0, 5);
    expect(t[2]).toBeCloseTo(128 / 255, 5);
  });

  it('applies mean/std: (v/255 - mean)/std', () => {
    const t = framesToTensor([frame(128, 128, 128)], 1, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]);
    expect(t[0]).toBeCloseTo((128 / 255 - 0.5) / 0.5, 5);
  });

  it('lays out as F,C,H,W (channels-major within a frame, frames outermost)', () => {
    const t = framesToTensor([frame(255, 0, 0), frame(0, 0, 255)], 1, [0, 0, 0], [1, 1, 1]);
    expect(Array.from(t).map((x) => Math.round(x * 255))).toEqual([255, 0, 0, 0, 0, 255]);
  });

  it('produces a tensor of length F*C*H*W', () => {
    const blank = new Uint8ClampedArray(4 * 4 * 4); // 2x2 frame = 4 px * 4 channels
    const t = framesToTensor([blank, blank], 2, [0, 0, 0], [1, 1, 1]);
    expect(t.length).toBe(2 * 3 * 2 * 2);
  });
});
