/**
 * Microphone recording as a hook, with the live analyser exposed so the
 * sonagram strip can draw from it while capture is still running.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { encodeWav, startCapture, type CaptureHandle } from '../lib/audio';

export interface Recorder {
  recording: boolean;
  seconds: number;
  analyser: AnalyserNode | null;
  error: string | null;
  start: () => Promise<void>;
  stop: () => Promise<Blob | null>;
  cancel: () => void;
}

export function useRecorder(maxSeconds = 30): Recorder {
  const handleRef = useRef<CaptureHandle | null>(null);
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);
  const [error, setError] = useState<string | null>(null);

  const cancel = useCallback(() => {
    handleRef.current?.cancel();
    handleRef.current = null;
    setRecording(false);
    setAnalyser(null);
    setSeconds(0);
  }, []);

  // Releasing the microphone on unmount matters: leaving it open keeps the
  // browser's recording indicator lit after the user has navigated away.
  useEffect(() => () => handleRef.current?.cancel(), []);

  const stop = useCallback(async (): Promise<Blob | null> => {
    const handle = handleRef.current;
    if (!handle) return null;
    handleRef.current = null;
    setRecording(false);
    setAnalyser(null);

    const samples = await handle.stop();
    setSeconds(0);
    if (samples.length < 1600) {
      setError('That was too short to analyse. Hold the button and speak for a second or two.');
      return null;
    }
    return encodeWav(samples);
  }, []);

  const start = useCallback(async () => {
    setError(null);
    try {
      const handle = await startCapture();
      handleRef.current = handle;
      setAnalyser(handle.analyser);
      setRecording(true);
      setSeconds(0);
    } catch (cause) {
      const name = (cause as DOMException)?.name;
      setError(
        name === 'NotAllowedError'
          ? 'Microphone permission was denied. Allow it in your browser, or upload a file instead.'
          : name === 'NotFoundError'
            ? 'No microphone found. Upload a file instead.'
            : `Could not open the microphone: ${(cause as Error)?.message ?? cause}`,
      );
    }
  }, []);

  useEffect(() => {
    if (!recording) return;
    const id = window.setInterval(() => {
      setSeconds((s) => {
        const next = s + 0.1;
        if (next >= maxSeconds) {
          // Stopping here rather than recording indefinitely: the encoder only
          // samples a handful of windows, so more audio adds upload size
          // without improving the embedding.
          void stop();
        }
        return next;
      });
    }, 100);
    return () => window.clearInterval(id);
  }, [recording, maxSeconds, stop]);

  return { recording, seconds, analyser, error, start, stop, cancel };
}
