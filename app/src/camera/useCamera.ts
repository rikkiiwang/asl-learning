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
    // Release any prior stream first (prevents leaks / "camera busy" black frames).
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
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

  // Ref callback for the <video>: whenever the element (re)mounts, immediately
  // re-attach the live stream. This is the robust fix for the element being
  // remounted between attempts — a fresh <video> has no srcObject, so binding it
  // here (rather than once in start) guarantees the preview survives remounts.
  const attachVideo = useCallback((el: HTMLVideoElement | null) => {
    videoRef.current = el;
    const s = streamRef.current;
    if (el && s && s.getVideoTracks()[0]?.readyState === 'live') {
      el.srcObject = s;
      void el.play().catch(() => undefined);
    }
  }, []);

  // Ensure a live preview: re-bind if the stream is still live, else re-acquire it.
  const resume = useCallback(() => {
    const v = videoRef.current;
    const s = streamRef.current;
    const live = !!s && s.getVideoTracks()[0]?.readyState === 'live';
    if (v && s && live) {
      v.srcObject = s;
      void v.play().catch(() => undefined);
    } else {
      void start(); // stream stopped/lost — re-acquire the camera
    }
  }, [start]);

  // Always release the camera when the component unmounts.
  useEffect(() => () => stop(), [stop]);

  return { videoRef, attachVideo, state, error, start, stop, resume };
}
