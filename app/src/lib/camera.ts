export type BrightnessStatus = 'ok' | 'too_dark' | 'too_bright';
export type CameraErrorKind = 'denied' | 'unavailable' | 'in_use' | 'unsupported' | 'error';

export interface CameraError {
  kind: CameraErrorKind;
  message: string;
}

// Luminance thresholds on a 0–255 scale (tunable; documented as supported conditions).
const DARK_THRESHOLD = 50;
const BRIGHT_THRESHOLD = 215;

/** Mean relative luminance (Rec. 709) of RGBA pixel data; alpha ignored. 0–255. */
export function computeMeanLuminance(rgba: Uint8ClampedArray): number {
  const pixels = Math.floor(rgba.length / 4);
  if (pixels === 0) return 0;
  let sum = 0;
  for (let i = 0; i < pixels; i++) {
    const r = rgba[i * 4];
    const g = rgba[i * 4 + 1];
    const b = rgba[i * 4 + 2];
    sum += 0.2126 * r + 0.7152 * g + 0.0722 * b;
  }
  return sum / pixels;
}

export function assessBrightness(meanLuminance: number): BrightnessStatus {
  if (meanLuminance < DARK_THRESHOLD) return 'too_dark';
  if (meanLuminance > BRIGHT_THRESHOLD) return 'too_bright';
  return 'ok';
}

/** Map a getUserMedia rejection (DOMException-like) to a learner-facing message. */
export function describeCameraError(err: unknown): CameraError {
  const name =
    typeof err === 'object' && err !== null && 'name' in err
      ? String((err as { name: unknown }).name)
      : '';
  switch (name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return {
        kind: 'denied',
        message: 'Camera permission was blocked. Allow camera access in your browser, then try again.',
      };
    case 'NotFoundError':
    case 'DevicesNotFoundError':
      return { kind: 'unavailable', message: 'No camera was found on this device.' };
    case 'NotReadableError':
    case 'TrackStartError':
      return {
        kind: 'in_use',
        message: 'Your camera is in use by another app. Close it and try again.',
      };
    case 'OverconstrainedError':
    case 'ConstraintNotSatisfiedError':
      return { kind: 'unsupported', message: 'This camera does not support the required settings.' };
    default:
      return { kind: 'error', message: 'Could not start the camera. Please try again.' };
  }
}
