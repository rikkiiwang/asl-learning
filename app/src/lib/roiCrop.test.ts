import { describe, it, expect } from 'vitest';
import { motionRoiBox, percentile, type RoiFrame } from './roiCrop';

const W = 192;

function blank(value: number): Uint8ClampedArray {
  const d = new Uint8ClampedArray(W * W * 4);
  for (let p = 0; p < W * W; p++) {
    d[p * 4] = value;
    d[p * 4 + 1] = value;
    d[p * 4 + 2] = value;
    d[p * 4 + 3] = 255;
  }
  return d;
}

/** Gray bg with a bright square whose top-left is (sx, sy). */
function frameWithSquare(sx: number, sy: number, sz = 40, bg = 100, fg = 230): RoiFrame {
  const d = blank(bg);
  for (let y = sy; y < sy + sz; y++) {
    for (let x = sx; x < sx + sz; x++) {
      const i = (y * W + x) * 4;
      d[i] = fg;
      d[i + 1] = fg;
      d[i + 2] = fg;
    }
  }
  return { data: d };
}

describe('motionRoiBox', () => {
  it('matches the Python motion_roi_box on an identical synthetic clip (parity)', () => {
    // Same input the Python reference produced (43, 35, 106) for — see model
    // preprocess.motion_roi_box. Tolerance covers the JS-vs-cv2 blur approximation.
    const frames = Array.from({ length: 8 }, (_, t) => frameWithSquare(40 + 8 * t, 76));
    const box = motionRoiBox(frames, W);
    expect(box).not.toBeNull();
    expect(Math.abs(box!.y0 - 43)).toBeLessThanOrEqual(3); // Python: (43, 35, 106)
    expect(Math.abs(box!.x0 - 35)).toBeLessThanOrEqual(3);
    expect(Math.abs(box!.side - 106)).toBeLessThanOrEqual(3);
  });

  it('returns null with fewer than 3 frames', () => {
    expect(motionRoiBox([frameWithSquare(40, 76), frameWithSquare(48, 76)], W)).toBeNull();
  });

  it('returns null when there is no motion (identical frames)', () => {
    const still = frameWithSquare(80, 80);
    expect(motionRoiBox([still, still, still, still], W)).toBeNull();
  });

  it('centers the box on the moving region', () => {
    // Square moving around (120,120) -> box center should sit in that quadrant.
    const frames = Array.from({ length: 8 }, (_, t) => frameWithSquare(110 + 4 * t, 110));
    const box = motionRoiBox(frames, W)!;
    expect(box).not.toBeNull();
    const cx = box.x0 + box.side / 2;
    const cy = box.y0 + box.side / 2;
    expect(cx).toBeGreaterThan(W / 2 - 30);
    expect(cy).toBeGreaterThan(W / 2 - 30);
  });

  it('clamps the side to the [0.55*W, W] range', () => {
    const frames = Array.from({ length: 8 }, (_, t) => frameWithSquare(40 + 8 * t, 76));
    const box = motionRoiBox(frames, W)!;
    expect(box.side).toBeGreaterThanOrEqual(Math.round(0.55 * W));
    expect(box.side).toBeLessThanOrEqual(W);
    expect(box.x0).toBeGreaterThanOrEqual(0);
    expect(box.x0 + box.side).toBeLessThanOrEqual(W);
  });
});

describe('percentile', () => {
  it('linearly interpolates like numpy', () => {
    expect(percentile([0, 10], 50)).toBeCloseTo(5, 6);
    expect(percentile([0, 1, 2, 3, 4], 5)).toBeCloseTo(0.2, 6);
    expect(percentile([0, 1, 2, 3, 4], 95)).toBeCloseTo(3.8, 6);
  });
});
