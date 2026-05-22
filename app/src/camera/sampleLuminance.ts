import { computeMeanLuminance } from '../lib/camera';

/**
 * Draw the current video frame to a small offscreen canvas and return its mean
 * luminance (0–255), or null if the frame isn't ready. Downscaled for cheap polling.
 */
export function sampleMeanLuminance(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
): number | null {
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (!w || !h) return null;

  const scale = 64 / Math.max(w, h);
  const cw = Math.max(1, Math.round(w * scale));
  const ch = Math.max(1, Math.round(h * scale));
  canvas.width = cw;
  canvas.height = ch;

  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  ctx.drawImage(video, 0, 0, cw, ch);
  return computeMeanLuminance(ctx.getImageData(0, 0, cw, ch).data);
}
