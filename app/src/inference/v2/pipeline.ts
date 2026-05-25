// v2 "Constellation" 3-stage browser inference: detector -> landmark -> geometry
// -> recognizer. The numerically load-bearing glue (decode, geometry) is in
// detect.ts/geometry.ts and parity-tested vs Python; this file orchestrates the
// three ONNX sessions and the canvas preprocessing.
import type { InferenceSession } from 'onnxruntime-web';
import { decodeDetections } from './detect';
import { buildClipGeometry, GEOM_DIM, type ClipKps, type FrameDet, type FrameKps } from './geometry';
import {
  activeWindow,
  detectorInput,
  handCrop64Input,
  sampleIndices,
  type Norm,
} from './preprocess';

const V2_DIR = '/models/v2';
const FRAMES = 16;

interface V2Meta {
  detector: { img: number; score_thr: number; iou_thr: number; norm: Norm };
  landmark: { frame_scale: number; norm: Norm };
}

export class V2Recognizer {
  private det: InferenceSession | null = null;
  private lm: InferenceSession | null = null;
  private rec: InferenceSession | null = null;
  private meta: V2Meta | null = null;
  private srcCanvas: HTMLCanvasElement | null = null;

  private async ensure() {
    if (this.rec) return;
    const ort = await import('onnxruntime-web');
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/';
    this.meta = (await (await fetch(`${V2_DIR}/v2_meta.json`)).json()) as V2Meta;
    const opts = { executionProviders: ['wasm'] as const };
    [this.det, this.lm, this.rec] = await Promise.all([
      ort.InferenceSession.create(`${V2_DIR}/detector.onnx`, opts),
      ort.InferenceSession.create(`${V2_DIR}/landmark.onnx`, opts),
      ort.InferenceSession.create(`${V2_DIR}/recognizer.onnx`, opts),
    ]);
  }

  /** Run the full pipeline on raw work×work RGBA frames; returns 75 logits. */
  async recognizeFrames(frames: ImageData[], work: number): Promise<Float32Array> {
    await this.ensure();
    const ort = await import('onnxruntime-web');
    const meta = this.meta!;

    // temporal window + uniform resample to 16 (cache.py:decode_frames)
    const [lo, hi] = activeWindow(frames);
    const span = hi > lo ? frames.slice(lo, hi + 1) : frames;
    const sampled = sampleIndices(span.length, FRAMES).map((i) => span[i]);

    if (!this.srcCanvas) this.srcCanvas = document.createElement('canvas');
    const src = this.srcCanvas;
    src.width = work;
    src.height = work;
    const sctx = src.getContext('2d', { willReadFrequently: true })!;

    const dets: FrameDet[] = [];
    const kps: ClipKps = [];

    for (const frame of sampled) {
      sctx.putImageData(frame, 0, 0);

      // --- stage 1: detector ---
      const detIn = detectorInput(src, meta.detector.img, meta.detector.norm);
      const detOut = await this.det!.run({
        frames: new ort.Tensor('float32', detIn, [1, 3, meta.detector.img, meta.detector.img]),
      });
      const cls = detOut.cls_logits.data as Float32Array;
      const box = detOut.box_pred.data as Float32Array;
      const { hands, head } = decodeDetections(cls, box, {
        img: meta.detector.img,
        origW: work,
        origH: work,
        scoreThr: meta.detector.score_thr,
        iouThr: meta.detector.iou_thr,
      });
      dets.push({ hands, head });

      // --- stage 1.5: landmark (batch the <=2 hand crops) ---
      const frameKps: FrameKps = [];
      if (hands.length > 0) {
        const windows = hands.map((b) =>
          handCrop64Input(src, b, meta.landmark.frame_scale, meta.landmark.norm),
        );
        const batch = new Float32Array(windows.length * 3 * 64 * 64);
        windows.forEach((w, i) => batch.set(w.input, i * 3 * 64 * 64));
        const lmOut = await this.lm!.run({
          crops: new ort.Tensor('float32', batch, [windows.length, 3, 64, 64]),
        });
        const kp = lmOut.kps.data as Float32Array; // [k,21,2] in [0,1]
        windows.forEach((w, i) => {
          const [wx0, wy0, wx1, wy1] = w.window;
          const pts: number[][] = [];
          for (let j = 0; j < 21; j++) {
            const x01 = kp[i * 42 + j * 2];
            const y01 = kp[i * 42 + j * 2 + 1];
            pts.push([wx0 + x01 * (wx1 - wx0), wy0 + y01 * (wy1 - wy0)]);
          }
          frameKps.push(pts);
        });
      }
      kps.push(frameKps);
    }

    // --- stage 2: geometry -> recognizer ---
    const geom = buildClipGeometry(dets, kps, work, work);
    const recOut = await this.rec!.run({
      geom: new ort.Tensor('float32', geom, [1, FRAMES, GEOM_DIM]),
    });
    return recOut.logits.data as Float32Array;
  }
}

let singleton: V2Recognizer | null = null;
export function getV2Recognizer(): V2Recognizer {
  if (!singleton) singleton = new V2Recognizer();
  return singleton;
}
