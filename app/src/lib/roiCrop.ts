// Classical motion-based ROI crop, ported from the model's Python
// (model/src/asl/preprocess.py :: motion_roi_box). It finds the signing-motion
// region across the captured frames and returns a square crop box, so the live
// app frames the signer the SAME way the model was trained (ROI-cropped clips).
// Pure pixel statistics — no learned model, matching the from-scratch constraint.

export interface RoiBox {
  y0: number;
  x0: number;
  side: number;
}

export interface RoiFrame {
  data: Uint8ClampedArray; // RGBA, length w*w*4
}

/** RGBA -> grayscale luma. Matches cv2 BGR2GRAY weights (0.299R+0.587G+0.114B). */
function toGray(data: Uint8ClampedArray, n: number): Float32Array {
  const g = new Float32Array(n);
  for (let p = 0; p < n; p++) {
    const i = p * 4;
    g[p] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }
  return g;
}

/** Separable Gaussian blur of a w×h float image (approximates cv2.GaussianBlur). */
export function gaussianBlur(src: Float32Array, w: number, h: number, sigma: number): Float32Array {
  if (sigma <= 0) return src;
  const radius = Math.max(1, Math.ceil(sigma * 3));
  const ker = new Float32Array(radius * 2 + 1);
  let sum = 0;
  for (let i = -radius; i <= radius; i++) {
    const v = Math.exp(-(i * i) / (2 * sigma * sigma));
    ker[i + radius] = v;
    sum += v;
  }
  for (let i = 0; i < ker.length; i++) ker[i] /= sum;

  const tmp = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let k = -radius; k <= radius; k++) {
        let xx = x + k;
        if (xx < 0) xx = 0;
        else if (xx >= w) xx = w - 1;
        acc += src[y * w + xx] * ker[k + radius];
      }
      tmp[y * w + x] = acc;
    }
  }
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let k = -radius; k <= radius; k++) {
        let yy = y + k;
        if (yy < 0) yy = 0;
        else if (yy >= h) yy = h - 1;
        acc += tmp[yy * w + x] * ker[k + radius];
      }
      out[y * w + x] = acc;
    }
  }
  return out;
}

/** numpy-style linear-interpolation percentile over an ascending-sorted array. */
export function percentile(sortedAsc: number[], p: number): number {
  const n = sortedAsc.length;
  if (n === 0) return 0;
  if (n === 1) return sortedAsc[0];
  const idx = (p / 100) * (n - 1);
  const lo = Math.floor(idx);
  if (lo + 1 >= n) return sortedAsc[n - 1];
  return sortedAsc[lo] + (idx - lo) * (sortedAsc[lo + 1] - sortedAsc[lo]);
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/**
 * Motion-ROI box over square w×w RGBA frames. Returns a square box (y0,x0,side),
 * or null to fall back to a center crop. Faithful port of motion_roi_box:
 * mean inter-frame abs-diff -> blur -> threshold at 0.25*max -> robust [5,95]
 * percentile bbox -> square + margin, clamped to [0.55*w, w].
 */
export function motionRoiBox(frames: RoiFrame[], w: number, margin = 0.2): RoiBox | null {
  if (frames.length < 3) return null;
  const n = w * w;
  const grays = frames.map((f) => toGray(f.data, n));

  const motion = new Float32Array(n);
  for (let t = 1; t < grays.length; t++) {
    const a = grays[t];
    const b = grays[t - 1];
    for (let p = 0; p < n; p++) motion[p] += Math.abs(a[p] - b[p]);
  }
  const denom = grays.length - 1;
  for (let p = 0; p < n; p++) motion[p] /= denom;

  const blurred = gaussianBlur(motion, w, w, w / 80);
  let mx = 0;
  for (let p = 0; p < n; p++) if (blurred[p] > mx) mx = blurred[p];
  if (mx < 2.0) return null; // almost no motion -> center-crop fallback

  const thr = 0.25 * mx;
  const xs: number[] = [];
  const ys: number[] = [];
  for (let y = 0; y < w; y++) {
    for (let x = 0; x < w; x++) {
      if (blurred[y * w + x] > thr) {
        xs.push(x);
        ys.push(y);
      }
    }
  }
  if (xs.length < 25) return null;

  xs.sort((p, q) => p - q);
  ys.sort((p, q) => p - q);
  const x1 = percentile(xs, 5);
  const x2 = percentile(xs, 95);
  const y1 = percentile(ys, 5);
  const y2 = percentile(ys, 95);
  const bw = x2 - x1;
  const bh = y2 - y1;
  const cx = (x1 + x2) / 2;
  const cy = (y1 + y2) / 2;
  let side = Math.max(bw, bh) * (1 + margin);
  side = clamp(side, 0.55 * w, w);
  const x0 = Math.round(clamp(cx - side / 2, 0, w - side));
  const y0 = Math.round(clamp(cy - side / 2, 0, w - side));
  return { y0, x0, side: Math.round(side) };
}
