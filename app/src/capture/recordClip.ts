import { sampleFrameIndices } from '../lib/frameSampling';

export interface RecordedClip {
  /** Exactly `frames` (default 16) RGBA frames, uniformly resampled to match the model. */
  frames: Uint8ClampedArray[];
  size: number;
  /** Raw frames captured before resampling (for diagnostics). */
  capturedCount: number;
}

/**
 * Record the live video for ~durationMs, center-square-cropping each frame to
 * `size`×`size` (mirroring the model's preprocessing), then uniformly resample
 * the captured frames down to `frames` using the SAME index logic as the model.
 */
export async function recordClip(
  video: HTMLVideoElement,
  opts: { durationMs?: number; fps?: number; size?: number; frames?: number } = {},
): Promise<RecordedClip> {
  const { durationMs = 3000, fps = 12, size = 112, frames: k = 16 } = opts;

  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Canvas 2D context unavailable');

  const raw: Uint8ClampedArray[] = [];
  const interval = 1000 / fps;
  const start = performance.now();

  await new Promise<void>((resolve) => {
    const tick = () => {
      const vw = video.videoWidth;
      const vh = video.videoHeight;
      if (vw && vh) {
        const s = Math.min(vw, vh); // center square crop
        ctx.drawImage(video, (vw - s) / 2, (vh - s) / 2, s, s, 0, 0, size, size);
        raw.push(new Uint8ClampedArray(ctx.getImageData(0, 0, size, size).data));
      }
      if (performance.now() - start < durationMs) setTimeout(tick, interval);
      else resolve();
    };
    tick();
  });

  const blank = new Uint8ClampedArray(size * size * 4);
  const frames = sampleFrameIndices(raw.length, k).map((i) => raw[i] ?? blank);
  return { frames, size, capturedCount: raw.length };
}
