// Detector anchor grid + decode/NMS — port of model-v2 detect/anchors.py,
// detect/encode.py:decode, and detect/infer.py:detect_frame post-processing.
import { type Box, nms } from './boxes';

const BASE_IMG = 128;
const BASE_SCALES = [32, 64, 96];

function scalesFor(img: number): number[] {
  const f = img / BASE_IMG;
  return BASE_SCALES.map((s) => s * f);
}

/** Square anchor grid; same row-major (cell, scale) ordering as make_anchors. */
export function anchorsFor(img: number, stride = 16): Box[] {
  const n = Math.floor(img / stride);
  const cs: number[] = [];
  for (let i = 0; i < n; i++) cs.push((i + 0.5) * stride);
  const scales = scalesFor(img);
  const out: Box[] = [];
  // np.meshgrid(cs, cs).reshape(-1): row i = cy, col j = cx, row-major.
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const cx = cs[j];
      const cy = cs[i];
      for (const s of scales) out.push([cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2]);
    }
  }
  return out;
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
const clip = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

export interface FrameDetections {
  hands: Box[]; // <=2, highest score first
  head: Box | null; // top-1
}

export interface DecodeOpts {
  img: number; // detector input resolution (e.g. 192)
  origW: number;
  origH: number;
  scoreThr?: number;
  iouThr?: number;
}

/**
 * Decode raw detector outputs into hand/head boxes in original-frame pixels.
 * `cls`: length n*2 logits (row-major [anchor, class]); `box`: length n*4 deltas.
 * Mirrors detect_frame exactly: clip deltas, decode, filter, scale, NMS per class,
 * top-2 hands + top-1 head.
 */
export function decodeDetections(
  cls: ArrayLike<number>,
  box: ArrayLike<number>,
  opts: DecodeOpts,
): FrameDetections {
  const { img, origW, origH } = opts;
  const scoreThr = opts.scoreThr ?? 0.3;
  const iouThr = opts.iouThr ?? 0.45;
  const anchors = anchorsFor(img);
  const n = anchors.length;

  // decode each anchor's box + class (with delta clip to [-10,10]).
  const boxes: Box[] = new Array(n);
  const labels: number[] = new Array(n);
  const conf: number[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const [ax0, ay0, ax1, ay1] = anchors[i];
    const acx = (ax0 + ax1) / 2;
    const acy = (ay0 + ay1) / 2;
    const aw = ax1 - ax0;
    const ah = ay1 - ay0;
    const d0 = clip(box[i * 4 + 0], -10, 10);
    const d1 = clip(box[i * 4 + 1], -10, 10);
    const d2 = clip(box[i * 4 + 2], -10, 10);
    const d3 = clip(box[i * 4 + 3], -10, 10);
    const cx = d0 * aw + acx;
    const cy = d1 * ah + acy;
    const w = Math.exp(d2) * aw;
    const h = Math.exp(d3) * ah;
    boxes[i] = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
    const p0 = sigmoid(cls[i * 2 + 0]);
    const p1 = sigmoid(cls[i * 2 + 1]);
    labels[i] = p1 > p0 ? 1 : 0; // argmax (ties -> class 0, matching numpy)
    conf[i] = Math.max(p0, p1);
  }

  const sx = origW / img;
  const sy = origH / img;

  // per-class: threshold, NMS, then scale + clip to frame, drop degenerate.
  const perClass = (c: number): { b: Box; s: number }[] => {
    const idx: number[] = [];
    for (let i = 0; i < n; i++) {
      if (labels[i] !== c || conf[i] < scoreThr) continue;
      const b = boxes[i];
      if (!b.every(Number.isFinite) || b[2] <= b[0] || b[3] <= b[1]) continue;
      idx.push(i);
    }
    const subB = idx.map((i) => boxes[i]);
    const subS = idx.map((i) => conf[i]);
    const keep = nms(subB, subS, iouThr);
    const out: { b: Box; s: number }[] = [];
    for (const k of keep) {
      let [x0, y0, x1, y1] = subB[k];
      x0 = clip(x0 * sx, 0, origW);
      x1 = clip(x1 * sx, 0, origW);
      y0 = clip(y0 * sy, 0, origH);
      y1 = clip(y1 * sy, 0, origH);
      if (x1 > x0 && y1 > y0) out.push({ b: [x0, y0, x1, y1], s: subS[k] });
    }
    return out;
  };

  const hands = perClass(0)
    .sort((p, q) => q.s - p.s)
    .slice(0, 2)
    .map((p) => p.b);
  const headCands = perClass(1);
  let head: Box | null = null;
  if (headCands.length) {
    head = headCands.reduce((a, b) => (b.s > a.s ? b : a)).b;
  }
  return { hands, head };
}
