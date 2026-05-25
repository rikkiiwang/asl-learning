// Browser preprocessing for the v2 pipeline: temporal windowing + the canvas
// crops/normalisation each stage expects. The numerically load-bearing math
// (decode, geometry) lives in detect.ts/geometry.ts and is parity-tested; these
// helpers are standard canvas ops mirroring cache.py's frame handling.
import type { Box } from './boxes';

export interface Norm {
  mean: [number, number, number];
  std: [number, number, number];
}

/** v1's uniform temporal resample (preprocess.py:sample_indices). */
export function sampleIndices(n: number, k = 16): number[] {
  if (n <= 0) return new Array(k).fill(0);
  if (n >= k) {
    const out: number[] = [];
    for (let i = 0; i < k; i++) out.push(Math.round((i * (n - 1)) / (k - 1)));
    return out;
  }
  const out = Array.from({ length: n }, (_, i) => i);
  while (out.length < k) out.push(n - 1);
  return out;
}

function toGray64(img: ImageData, cv: HTMLCanvasElement): Float32Array {
  const ctx = cv.getContext('2d', { willReadFrequently: true })!;
  // draw the frame scaled to 64×64, then read luma.
  const tmp = document.createElement('canvas');
  tmp.width = img.width;
  tmp.height = img.height;
  tmp.getContext('2d')!.putImageData(img, 0, 0);
  ctx.clearRect(0, 0, 64, 64);
  ctx.drawImage(tmp, 0, 0, 64, 64);
  const d = ctx.getImageData(0, 0, 64, 64).data;
  const g = new Float32Array(64 * 64);
  for (let i = 0; i < 64 * 64; i++) {
    // OpenCV BGR2GRAY weights (cache.py uses cv2): 0.299R+0.587G+0.114B.
    g[i] = 0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2];
  }
  return g;
}

/**
 * Motion-based signing window (cache.py:active_window). Frame-diff on 64² gray,
 * 5-tap box smoothing (zero-padded, to match np.convolve mode='same'), threshold
 * at frac×max. Returns inclusive [lo, hi].
 */
export function activeWindow(frames: ImageData[], frac = 0.3): [number, number] {
  const n = frames.length;
  if (n < 6) return [0, n - 1];
  const cv = document.createElement('canvas');
  cv.width = 64;
  cv.height = 64;
  const grays = frames.map((f) => toGray64(f, cv));
  const motion = new Float32Array(n); // motion[0] = 0
  for (let i = 1; i < n; i++) {
    let s = 0;
    const a = grays[i];
    const b = grays[i - 1];
    for (let p = 0; p < a.length; p++) s += Math.abs(a[p] - b[p]);
    motion[i] = s / a.length;
  }
  const sm = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let d = -2; d <= 2; d++) {
      const j = i + d;
      if (j >= 0 && j < n) s += motion[j];
    }
    sm[i] = s / 5;
  }
  let mx = 0;
  for (let i = 0; i < n; i++) mx = Math.max(mx, sm[i]);
  const thr = frac * mx;
  const active: number[] = [];
  for (let i = 0; i < n; i++) if (sm[i] > thr) active.push(i);
  if (active.length < 2) return [0, n - 1];
  return [active[0], active[active.length - 1]];
}

/** Resize an ImageData to img×img and return normalised CHW float32 (detector). */
export function detectorInput(src: HTMLCanvasElement, img: number, norm: Norm): Float32Array {
  const c = document.createElement('canvas');
  c.width = img;
  c.height = img;
  const ctx = c.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(src, 0, 0, img, img);
  const d = ctx.getImageData(0, 0, img, img).data;
  const out = new Float32Array(3 * img * img);
  const plane = img * img;
  for (let p = 0; p < plane; p++) {
    for (let ch = 0; ch < 3; ch++) {
      const v = d[p * 4 + ch] / 255;
      out[ch * plane + p] = (v - norm.mean[ch]) / norm.std[ch];
    }
  }
  return out;
}

/**
 * Crop a 2×-framed square window around a hand box, resize to 64×64, normalise.
 * Returns [3,64,64] float32. Mirrors cache.py:landmark_kps framing (window =
 * max(w,h)*frameScale, centred on the box). Edge handling: canvas clamps where
 * Python warpAffine reflects — a small border-only difference.
 */
export function handCrop64Input(
  src: HTMLCanvasElement,
  box: Box,
  frameScale: number,
  norm: Norm,
): { input: Float32Array; window: Box } {
  const [x1, y1, x2, y2] = box;
  const cx = (x1 + x2) / 2;
  const cy = (y1 + y2) / 2;
  const win = Math.max(x2 - x1, y2 - y1) * frameScale;
  const wx = cx - win / 2;
  const wy = cy - win / 2;

  // Two-step crop so the FULL window maps to 64×64 even when it extends past the
  // frame edge: place the source on a window-sized canvas (out-of-frame = blank,
  // where Python warpAffine reflects), then downscale. A single drawImage with an
  // out-of-bounds source rect would instead clip+rescale and distort the mapping.
  const W = Math.max(1, Math.round(win));
  const winCv = document.createElement('canvas');
  winCv.width = W;
  winCv.height = W;
  winCv.getContext('2d')!.drawImage(src, -wx, -wy);

  const c = document.createElement('canvas');
  c.width = 64;
  c.height = 64;
  const ctx = c.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(winCv, 0, 0, W, W, 0, 0, 64, 64);
  const d = ctx.getImageData(0, 0, 64, 64).data;
  const out = new Float32Array(3 * 64 * 64);
  const plane = 64 * 64;
  for (let p = 0; p < plane; p++) {
    for (let ch = 0; ch < 3; ch++) {
      const v = d[p * 4 + ch] / 255;
      out[ch * plane + p] = (v - norm.mean[ch]) / norm.std[ch];
    }
  }
  return { input: out, window: [wx, wy, wx + win, wy + win] };
}
