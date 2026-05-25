// Pose-geometry feature builder — port of model-v2 geometry.py + clip_features.py.
// Produces the (F, GEOM_DIM) tensor the recognizer consumes. Kept numerically
// faithful to the Python source (parity-tested against geometry_golden.json).
import type { Box } from './boxes';

export const GEOM_DIM = 2 * 21 * 2 + 2 * 2 + 2 + 2 + 1; // 93
const HEAD_FALLBACK_SCALE = 0.5;

export interface FrameDet {
  hands: Box[]; // 0..2 boxes
  head: Box | null;
}
export type HandKps = number[][]; // 21 × [x, y]
/** Hands in one frame (aligned with that frame's dets.hands). */
export type FrameKps = HandKps[];
/** Per-clip keypoints: clip → frame → hand → 21 × [x,y]. */
export type ClipKps = FrameKps[];

function median(vals: number[]): number {
  const s = [...vals].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function centroid(b: Box): [number, number] {
  return [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2];
}

/** Median of detected head boxes; fallback to a centered box if none. */
export function resolveHeadAnchor(
  heads: (Box | null)[],
  frameW: number,
  frameH: number,
): { anchor: Box; present: number } {
  const valid = heads.filter((h): h is Box => h !== null);
  if (valid.length > 0) {
    const col = (j: number) => median(valid.map((b) => b[j]));
    return { anchor: [col(0), col(1), col(2), col(3)], present: 1.0 };
  }
  const half = (HEAD_FALLBACK_SCALE * frameH) / 2;
  const cx = frameW / 2;
  const cy = frameH / 2;
  return { anchor: [cx - half, cy - half, cx + half, cy + half], present: 0.0 };
}

interface Track {
  last: [number, number];
  rows: Map<number, Box>;
  xs: number[];
  slot?: number;
}

/**
 * Associate hand boxes across frames into <=2 tracks by nearest-centroid (within
 * `gate` px), then slot each track left(0)/right(1) by clip-median x vs head_cx.
 * Returns per-frame slotted boxes (null if absent) and presence.
 */
export function slotHands(
  framesHands: Box[][],
  headCx: number,
  gate = 80.0,
): { slots: (Box | null)[][]; present: number[][] } {
  const F = framesHands.length;
  const tracks: Track[] = [];

  for (let f = 0; f < F; f++) {
    const boxes = framesHands[f];
    if (boxes.length === 0) continue;
    const cents = boxes.map(centroid);
    const used = new Set<number>();
    const existing = tracks.length; // new tracks (added below) skip this frame
    for (let ti = 0; ti < existing; ti++) {
      const tr = tracks[ti];
      const d = cents.map((c) => Math.hypot(c[0] - tr.last[0], c[1] - tr.last[1]));
      const orderAsc = d.map((_, i) => i).sort((a, b) => d[a] - d[b]);
      const cand = orderAsc.find((i) => !used.has(i) && d[i] <= gate);
      if (cand !== undefined) {
        used.add(cand);
        tr.rows.set(f, boxes[cand]);
        tr.last = cents[cand];
        tr.xs.push(cents[cand][0]);
      }
    }
    for (let i = 0; i < boxes.length; i++) {
      if (!used.has(i)) {
        tracks.push({ last: cents[i], rows: new Map([[f, boxes[i]]]), xs: [cents[i][0]] });
      }
    }
  }

  // keep the two best-supported tracks (stable on ties, matching Python sorted).
  let kept = tracks;
  if (tracks.length > 2) {
    kept = tracks
      .map((t, i) => ({ t, i }))
      .sort((a, b) => b.t.rows.size - a.t.rows.size || a.i - b.i)
      .slice(0, 2)
      .map((x) => x.t);
  }

  const medX = (t: Track) => median(t.xs);
  if (kept.length === 2) {
    const ordered = [...kept].sort((a, b) => medX(a) - medX(b));
    ordered[0].slot = 0;
    ordered[1].slot = 1;
  } else if (kept.length === 1) {
    kept[0].slot = medX(kept[0]) < headCx ? 0 : 1;
  }

  const slots: (Box | null)[][] = Array.from({ length: F }, () => [null, null]);
  const present: number[][] = Array.from({ length: F }, () => [0, 0]);
  for (const tr of kept) {
    if (tr.slot === undefined) continue;
    for (const [f, box] of tr.rows) {
      slots[f][tr.slot] = box;
      present[f][tr.slot] = 1.0;
    }
  }
  return { slots, present };
}

/** Translation/scale-invariant per-frame geometry (one frame). Returns GEOM_DIM. */
export function normalizeGeometry(
  kps: (number[][] | null)[], // [slot0|null, slot1|null], each 21x[x,y]
  present: number[],
  head: Box,
  headPresent: number,
): Float32Array {
  const hcx = (head[0] + head[2]) / 2;
  const hcy = (head[1] + head[3]) / 2;
  let hs = Math.max(head[2] - head[0], head[3] - head[1]);
  if (hs <= 0) hs = 1.0;

  const kpBlock = [
    new Float64Array(42),
    new Float64Array(42),
  ]; // per slot, 21*[x,y]
  const hand2head = [
    [0, 0],
    [0, 0],
  ];
  const handCenters: (number[] | null)[] = [null, null];

  for (let s = 0; s < 2; s++) {
    const k = kps[s];
    const ok = present[s] > 0 && k != null && k.every((p) => Number.isFinite(p[0]) && Number.isFinite(p[1]));
    if (!ok || k == null) continue;
    let mx = 0;
    let my = 0;
    for (let i = 0; i < 21; i++) {
      kpBlock[s][i * 2] = (k[i][0] - hcx) / hs;
      kpBlock[s][i * 2 + 1] = (k[i][1] - hcy) / hs;
      mx += k[i][0];
      my += k[i][1];
    }
    mx /= 21;
    my /= 21;
    handCenters[s] = [mx, my];
    hand2head[s] = [(mx - hcx) / hs, (my - hcy) / hs];
  }

  let hand2hand = [0, 0];
  if (handCenters[0] && handCenters[1]) {
    hand2hand = [
      (handCenters[1]![0] - handCenters[0]![0]) / hs,
      (handCenters[1]![1] - handCenters[0]![1]) / hs,
    ];
  }

  const out = new Float32Array(GEOM_DIM);
  let o = 0;
  for (let s = 0; s < 2; s++) for (let i = 0; i < 42; i++) out[o++] = kpBlock[s][i]; // 84
  out[o++] = hand2head[0][0];
  out[o++] = hand2head[0][1];
  out[o++] = hand2head[1][0];
  out[o++] = hand2head[1][1]; // 4
  out[o++] = hand2hand[0];
  out[o++] = hand2hand[1]; // 2
  out[o++] = present[0];
  out[o++] = present[1]; // 2
  out[o++] = headPresent; // 1
  return out;
}

/**
 * Full clip → (F*GEOM_DIM) geometry, matching build_clip_features (geometry path;
 * appearance crops are unused by the geometry-only RecognizerA).
 */
export function buildClipGeometry(
  dets: FrameDet[],
  kps: ClipKps,
  frameW: number,
  frameH: number,
): Float32Array {
  const F = dets.length;
  const { anchor, present: headPresent } = resolveHeadAnchor(
    dets.map((d) => d.head),
    frameW,
    frameH,
  );
  const headCx = (anchor[0] + anchor[2]) / 2;
  const framesHands = dets.map((d) => d.hands);
  const { slots, present } = slotHands(framesHands, headCx);

  const geom = new Float32Array(F * GEOM_DIM);
  for (let f = 0; f < F; f++) {
    const kpsSlot: (number[][] | null)[] = [null, null];
    const handsF = framesHands[f];
    const kpsF = kps[f];
    for (let s = 0; s < 2; s++) {
      if (present[f][s] <= 0) continue;
      const box = slots[f][s]!;
      if (handsF.length > 0) {
        // nearest box by L1 (matches np.argmin(|hands_f - box|.sum(axis=1)))
        let best = 0;
        let bestD = Infinity;
        for (let i = 0; i < handsF.length; i++) {
          const hb = handsF[i];
          const dd =
            Math.abs(hb[0] - box[0]) + Math.abs(hb[1] - box[1]) +
            Math.abs(hb[2] - box[2]) + Math.abs(hb[3] - box[3]);
          if (dd < bestD) {
            bestD = dd;
            best = i;
          }
        }
        if (best < kpsF.length) kpsSlot[s] = kpsF[best];
      }
    }
    const g = normalizeGeometry(kpsSlot, present[f], anchor, headPresent);
    geom.set(g, f * GEOM_DIM);
  }
  return geom;
}
