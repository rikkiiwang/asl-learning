import { describe, it, expect } from 'vitest';
import { computeMeanLuminance, assessBrightness, describeCameraError } from './camera';

function rgba(pixels: [number, number, number][]): Uint8ClampedArray {
  const out = new Uint8ClampedArray(pixels.length * 4);
  pixels.forEach(([r, g, b], i) => {
    out[i * 4] = r;
    out[i * 4 + 1] = g;
    out[i * 4 + 2] = b;
    out[i * 4 + 3] = 255; // alpha ignored
  });
  return out;
}

describe('computeMeanLuminance', () => {
  it('is 0 for all-black pixels', () => {
    expect(computeMeanLuminance(rgba([[0, 0, 0], [0, 0, 0]]))).toBe(0);
  });

  it('is ~255 for all-white pixels (alpha ignored)', () => {
    expect(Math.round(computeMeanLuminance(rgba([[255, 255, 255]])))).toBe(255);
  });

  it('averages luminance across pixels', () => {
    // one black, one white -> ~127.5
    const m = computeMeanLuminance(rgba([[0, 0, 0], [255, 255, 255]]));
    expect(m).toBeGreaterThan(126);
    expect(m).toBeLessThan(129);
  });
});

describe('assessBrightness', () => {
  it('flags too dark below the low threshold', () => {
    expect(assessBrightness(20)).toBe('too_dark');
  });
  it('flags too bright above the high threshold', () => {
    expect(assessBrightness(240)).toBe('too_bright');
  });
  it('is ok in the comfortable middle', () => {
    expect(assessBrightness(128)).toBe('ok');
  });
});

describe('describeCameraError', () => {
  it('maps a permission denial', () => {
    const e = describeCameraError({ name: 'NotAllowedError' });
    expect(e.kind).toBe('denied');
  });
  it('maps no-camera-found to unavailable', () => {
    expect(describeCameraError({ name: 'NotFoundError' }).kind).toBe('unavailable');
  });
  it('maps a camera already in use', () => {
    expect(describeCameraError({ name: 'NotReadableError' }).kind).toBe('in_use');
  });
  it('treats unknown errors as a generic error with a message', () => {
    const e = describeCameraError({ name: 'WeirdError' });
    expect(e.kind).toBe('error');
    expect(e.message.length).toBeGreaterThan(0);
  });
});
