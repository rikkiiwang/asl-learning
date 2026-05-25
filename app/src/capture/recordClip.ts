import { sampleFrameIndices } from '../lib/frameSampling';
import { motionRoiBox, type RoiFrame } from '../lib/roiCrop';

export interface RecordedClip {
  /** Exactly `frames` (default 16) RGBA frames, uniformly resampled to match the model. */
  frames: Uint8ClampedArray[];
  size: number;
  /** Raw frames captured before resampling (for diagnostics). */
  capturedCount: number;
  /** Whether a motion-ROI crop was applied (vs. center-crop fallback). */
  roiApplied: boolean;
}

/**
 * Record the live video for ~durationMs, then frame it the SAME way the model was
 * trained: capture center-square frames at a `work` resolution, compute a classical
 * motion-ROI box across them (zooms onto the signing region), crop each sampled
 * frame to that box (or center-crop if no motion), and resize to `size`×`size`.
 * Frame indices use the SAME logic as the model (sampleFrameIndices).
 */
export async function recordClip(
  video: HTMLVideoElement,
  opts: { durationMs?: number; fps?: number; size?: number; frames?: number; work?: number; margin?: number } = {},
): Promise<RecordedClip> {
  const { durationMs = 3000, fps = 12, size = 112, frames: k = 16, work = 192, margin = 0.2 } = opts;

  const cap = document.createElement('canvas');
  cap.width = work;
  cap.height = work;
  const capCtx = cap.getContext('2d', { willReadFrequently: true });
  if (!capCtx) throw new Error('Canvas 2D context unavailable');

  // 1) Capture raw center-square frames at the working resolution.
  const raw: ImageData[] = [];
  const interval = 1000 / fps;
  const start = performance.now();
  await new Promise<void>((resolve) => {
    const tick = () => {
      const vw = video.videoWidth;
      const vh = video.videoHeight;
      if (vw && vh) {
        const s = Math.min(vw, vh); // center square crop
        capCtx.drawImage(video, (vw - s) / 2, (vh - s) / 2, s, s, 0, 0, work, work);
        raw.push(capCtx.getImageData(0, 0, work, work));
      }
      if (performance.now() - start < durationMs) setTimeout(tick, interval);
      else resolve();
    };
    tick();
  });

  // 2) Motion-ROI box across the captured frames (null -> center crop).
  const box = motionRoiBox(raw as unknown as RoiFrame[], work, margin);
  const sx = box ? box.x0 : 0;
  const sy = box ? box.y0 : 0;
  const sSide = box ? box.side : work;

  // 3) Sample k frames, crop each to the ROI box, resize to size×size.
  const out = document.createElement('canvas');
  out.width = size;
  out.height = size;
  const outCtx = out.getContext('2d', { willReadFrequently: true });
  const src = document.createElement('canvas');
  src.width = work;
  src.height = work;
  const srcCtx = src.getContext('2d');
  if (!outCtx || !srcCtx) throw new Error('Canvas 2D context unavailable');

  const blank = new Uint8ClampedArray(size * size * 4);
  const frames = sampleFrameIndices(raw.length, k).map((i) => {
    const f = raw[i];
    if (!f) return blank;
    srcCtx.putImageData(f, 0, 0);
    outCtx.drawImage(src, sx, sy, sSide, sSide, 0, 0, size, size);
    return new Uint8ClampedArray(outCtx.getImageData(0, 0, size, size).data);
  });

  return { frames, size, capturedCount: raw.length, roiApplied: box !== null };
}
