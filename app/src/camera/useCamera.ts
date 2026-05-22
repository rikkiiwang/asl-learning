import { useCallback, useEffect, useRef, useState } from 'react';
import { describeCameraError, type CameraError } from '../lib/camera';

export type CameraState = 'idle' | 'requesting' | 'ready' | 'error';

/** Manage a front-facing camera MediaStream and bind it to a <video>. */
export function useCamera() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [state, setState] = useState<CameraState>('idle');
  const [error, setError] = useState<CameraError | null>(null);

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }, []);

  const start = useCallback(async () => {
    setError(null);
    setState('requesting');
    if (!navigator.mediaDevices?.getUserMedia) {
      setError({ kind: 'unsupported', message: 'This browser does not support camera access.' });
      setState('error');
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => undefined);
      }
      setState('ready');
    } catch (e) {
      setError(describeCameraError(e));
      setState('error');
    }
  }, []);

  // Always release the camera when the component unmounts.
  useEffect(() => () => stop(), [stop]);

  return { videoRef, state, error, start, stop };
}
