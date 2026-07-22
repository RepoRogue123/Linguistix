/**
 * The sonagram strip: the one element this interface is built around.
 *
 * A single continuous band carries the whole narrative rather than scattering
 * effects across the page. It has four states and they are the same object
 * throughout, so the user watches one thing transform instead of watching
 * panels appear and disappear:
 *
 *   idle       ambient room tone, barely moving
 *   recording  live voice scrolling right to left
 *   analysing  frozen, with a scan line sweeping once across it
 *   resolved   the result, with saliency burning the identifying regions brighter
 *
 * Rendering is column-major into an offscreen buffer and blitted, so a live
 * update costs one new column rather than a full repaint.
 */

import { useEffect, useRef } from 'react';
import { buildLut } from '../lib/colormap';

export type StripState = 'idle' | 'recording' | 'analysing' | 'resolved';

interface Props {
  state: StripState;
  /** Live analyser while recording. */
  analyser?: AnalyserNode | null;
  /** Server spectrogram, shape [n_mels][frames], 0-255. */
  spectrogram?: number[][] | null;
  /** Optional saliency overlay, same shape, 0-255. */
  saliency?: number[][] | null;
  theme: 'scope' | 'print';
  height?: number;
  label?: string;
  className?: string;
}

const HISTORY = 512;

export function SonagramStrip({
  state,
  analyser,
  spectrogram,
  saliency,
  theme,
  height = 200,
  label,
  className,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const bufferRef = useRef<Uint8Array | null>(null);
  // One ImageData reused across frames. Allocating a fresh one per frame cost
  // roughly 3 MB of garbage at 60 fps — about 184 MB/s of GC pressure.
  const imageRef = useRef<ImageData | null>(null);
  const idleTickRef = useRef(0);
  const writeRef = useRef(0);
  const rafRef = useRef<number>(0);
  const scanRef = useRef(0);


  // Reset the rolling buffer when a new recording starts, so the previous
  // take does not scroll through behind the new one.
  useEffect(() => {
    if (state === 'recording') {
      bufferRef.current = null;
      writeRef.current = 0;
    }
    if (state === 'analysing') scanRef.current = 0;
  }, [state]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const context = canvas.getContext('2d', { alpha: false });
    if (!context) return;

    const lut = buildLut(theme);
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    /** Reused framebuffer, resized only when the canvas actually changes. */
    const getImage = (width: number, height: number): ImageData => {
      const existing = imageRef.current;
      if (existing && existing.width === width && existing.height === height) return existing;
      const created = context.createImageData(width, height);
      imageRef.current = created;
      return created;
    };

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    };
    resize();

    const observer = new ResizeObserver(resize);
    observer.observe(canvas);

    /** Paint a [bins][frames] grid, optionally brightened by a saliency mask. */
    const paintGrid = (grid: number[][], mask: number[][] | null, sweep = 1) => {
      const bins = grid.length;
      const frames = grid[0]?.length ?? 0;
      if (!bins || !frames) return;

      const { width, height: h } = canvas;
      const image = getImage(width, h);
      const data = image.data;
      const visible = Math.max(1, Math.floor(frames * sweep));

      for (let x = 0; x < width; x++) {
        const fx = Math.floor((x / width) * frames);
        const revealed = fx < visible;

        for (let y = 0; y < h; y++) {
          // Flip vertically: low frequencies belong at the bottom, the way a
          // spectrogram has always been drawn.
          const by = bins - 1 - Math.floor((y / h) * bins);
          let value = grid[by]?.[fx] ?? 0;

          if (mask) {
            // Saliency does not replace the spectrogram, it emphasises within
            // it, so the user still sees their own voice underneath.
            const weight = (mask[by]?.[fx] ?? 0) / 255;
            value = Math.min(255, value * (0.45 + weight * 1.1));
          }
          if (!revealed) value = Math.floor(value * 0.12);

          const li = Math.max(0, Math.min(255, Math.floor(value))) * 3;
          const di = (y * width + x) * 4;
          data[di] = lut[li];
          data[di + 1] = lut[li + 1];
          data[di + 2] = lut[li + 2];
          data[di + 3] = 255;
        }
      }
      context.putImageData(image, 0, 0);
    };

    /** Append one analyser frame and scroll the history left. */
    const pushLive = () => {
      if (!analyser) return;
      const bins = analyser.frequencyBinCount;

      if (!bufferRef.current || bufferRef.current.length !== bins * HISTORY) {
        bufferRef.current = new Uint8Array(bins * HISTORY);
        writeRef.current = 0;
      }

      const frame = new Uint8Array(bins);
      analyser.getByteFrequencyData(frame);

      const buffer = bufferRef.current;
      const column = writeRef.current % HISTORY;
      for (let b = 0; b < bins; b++) buffer[b * HISTORY + column] = frame[b];
      writeRef.current++;

      const { width, height: h } = canvas;
      const image = getImage(width, h);
      const data = image.data;
      const filled = Math.min(writeRef.current, HISTORY);

      // Only the lower ~62% of FFT bins carry speech at 16 kHz; showing the
      // empty top third would waste most of the strip.
      const usable = Math.floor(bins * 0.62);

      for (let x = 0; x < width; x++) {
        const age = Math.floor((x / width) * filled);
        const col = (writeRef.current - filled + age) % HISTORY;

        for (let y = 0; y < h; y++) {
          const b = usable - 1 - Math.floor((y / h) * usable);
          const value = buffer[b * HISTORY + col] ?? 0;
          const li = value * 3;
          const di = (y * width + x) * 4;
          data[di] = lut[li];
          data[di + 1] = lut[li + 1];
          data[di + 2] = lut[li + 2];
          data[di + 3] = 255;
        }
      }
      context.putImageData(image, 0, 0);
    };

    /** Ambient idle texture. Low amplitude so it reads as a live instrument
     * that is powered on, not as content demanding attention. */
    const paintIdle = (t: number) => {
      const { width, height: h } = canvas;
      const image = getImage(width, h);
      const data = image.data;

      for (let x = 0; x < width; x++) {
        const drift = Math.sin(x * 0.014 + t * 0.0006) * 0.5 + 0.5;
        for (let y = 0; y < h; y++) {
          const fromBottom = 1 - y / h;
          const value = Math.floor(drift * 26 * Math.pow(fromBottom, 2.2) + Math.random() * 5);
          const li = Math.min(255, value) * 3;
          const di = (y * width + x) * 4;
          data[di] = lut[li];
          data[di + 1] = lut[li + 1];
          data[di + 2] = lut[li + 2];
          data[di + 3] = 255;
        }
      }
      context.putImageData(image, 0, 0);
    };

    const draw = (t: number) => {
      if (state === 'recording' && analyser) {
        pushLive();
      } else if (state === 'analysing' && spectrogram) {
        // One sweep, not a loop: it marks a single act of measurement.
        // Paced to the 1.6s scan window in Bench, ~60fps.
        scanRef.current = Math.min(1, scanRef.current + (reduceMotion ? 1 : 0.0104));
        paintGrid(spectrogram, null, scanRef.current);

        if (scanRef.current < 1) {
          const x = canvas.width * scanRef.current;
          context.fillStyle = theme === 'print' ? 'rgba(20,18,15,0.85)' : 'rgba(251,212,106,0.9)';
          context.fillRect(x - 1, 0, 2, canvas.height);
        }
      } else if (state === 'resolved' && spectrogram) {
        paintGrid(spectrogram, saliency ?? null, 1);
      } else if (!reduceMotion) {
        // Throttled to ~10 fps. This is ambient texture at rest, not content;
        // repainting 768,000 pixels 60 times a second for it was the second
        // largest cost in the app.
        if (t - idleTickRef.current > 100) {
          idleTickRef.current = t;
          paintIdle(t);
        }
      } else {
        paintIdle(0);
      }

      rafRef.current = requestAnimationFrame(draw);
    };

    if (state === 'idle' && reduceMotion) {
      paintIdle(0);
      return () => observer.disconnect();
    }

    // Static states need one paint, not a loop.
    if (state === 'resolved' && spectrogram) {
      paintGrid(spectrogram, saliency ?? null, 1);
      if (reduceMotion) {
        return () => observer.disconnect();
      }
    }

    rafRef.current = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(rafRef.current);
      observer.disconnect();
    };
  }, [state, analyser, spectrogram, saliency, theme]);

  return (
    <figure className={`strip strip--${state} ${className ?? ''}`}>
      <canvas
        ref={canvasRef}
        style={{ height: `${height}px` }}
        role="img"
        aria-label={
          label ??
          {
            idle: 'Spectrogram display, idle',
            recording: 'Live spectrogram of microphone input',
            analysing: 'Analysing the recorded audio',
            resolved: 'Spectrogram of the analysed clip',
          }[state]
        }
      />
      <figcaption className="strip__axis" aria-hidden="true">
        <span className="num">7 kHz</span>
        <span className="num">frequency</span>
        <span className="num">0</span>
      </figcaption>
    </figure>
  );
}
