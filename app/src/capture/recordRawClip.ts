/**
 * Capture raw center-square frames for the v2 pipeline. Unlike recordClip (which
 * motion-ROI-crops and resamples to 16×112 for the v1 contract), v2 needs the
 * FULL frames at a working resolution — its detector finds the hands itself, and
 * its own temporal windowing/resampling runs inside the pipeline. Returns every
 * captured frame as work×work RGBA ImageData.
 */
export async function recordRawClip(
  video: HTMLVideoElement,
  opts: { durationMs?: number; fps?: number; work?: number } = {},
): Promise<{ frames: ImageData[]; work: number }> {
  const { durationMs = 3000, fps = 12, work = 256 } = opts;

  const cap = document.createElement('canvas');
  cap.width = work;
  cap.height = work;
  const ctx = cap.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('Canvas 2D context unavailable');

  const frames: ImageData[] = [];
  const interval = 1000 / fps;
  const start = performance.now();
  await new Promise<void>((resolve) => {
    const tick = () => {
      const vw = video.videoWidth;
      const vh = video.videoHeight;
      if (vw && vh) {
        const s = Math.min(vw, vh); // center square crop
        ctx.drawImage(video, (vw - s) / 2, (vh - s) / 2, s, s, 0, 0, work, work);
        frames.push(ctx.getImageData(0, 0, work, work));
      }
      if (performance.now() - start < durationMs) setTimeout(tick, interval);
      else resolve();
    };
    tick();
  });

  return { frames, work };
}
