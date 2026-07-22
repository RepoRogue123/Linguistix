/**
 * Live identification: who is speaking, right now.
 *
 * A rolling window over the microphone, re-embedded a couple of times a second,
 * drawing a ribbon of who the system thinks is talking over time. Lightweight
 * diarization, and it makes an enrolment tangible: enrol yourself, come back
 * here, and watch your own name hold steady while you speak.
 *
 * The work is split. The encoder runs locally through ONNX because embedding
 * several times a second over a network would not survive a free CPU tier; the
 * gallery match stays on the server because that is where enrolments live and
 * because a matrix multiply against 50 centroids costs almost nothing.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

import { SonagramStrip } from '../components/SonagramStrip';
import { startCapture, TARGET_SAMPLE_RATE, type CaptureHandle } from '../lib/audio';
import { confidenceTextColor, ramp, rgbCss } from '../lib/colormap';
import { embed, loadEncoder, type EncoderMeta } from '../lib/onnx';

interface Props {
  theme: 'scope' | 'print';
}

interface Frame {
  at: number;
  speaker: string | null;
  score: number;
  matched: boolean;
}

const WINDOW_SECONDS = 2.5;
const STEP_MS = 700;
const RIBBON_LENGTH = 90;

export function Live({ theme }: Props) {
  const [running, setRunning] = useState(false);
  const [frames, setFrames] = useState<Frame[]>([]);
  const [meta, setMeta] = useState<EncoderMeta | null>(null);
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);
  const [status, setStatus] = useState('Loading the encoder into your browser…');
  const [error, setError] = useState<string | null>(null);

  const handleRef = useRef<CaptureHandle | null>(null);
  // A fixed-size ring with a write cursor. The previous version grew an array
  // and re-sliced it on every callback, doing two full 144,000-float copies
  // roughly twelve times a second — and it was O(n^2) during warm-up.
  const ringRef = useRef<Float32Array>(new Float32Array(0));
  const writeRef = useRef(0);
  const filledRef = useRef(0);
  const nodesRef = useRef<{ processor: ScriptProcessorNode; mute: GainNode } | null>(null);
  const timerRef = useRef<number>(0);
  const inflight = useRef(false);


  useEffect(() => {
    loadEncoder()
      .then(({ meta: m }) => {
        setMeta(m);
        setStatus('Ready. Start listening and speak.');
      })
      .catch((cause) => setError(`Could not load the in-browser encoder: ${cause}`));
  }, []);

  const stop = useCallback(() => {
    window.clearInterval(timerRef.current);
    nodesRef.current?.processor.disconnect();
    nodesRef.current?.mute.disconnect();
    nodesRef.current = null;
    handleRef.current?.cancel();
    handleRef.current = null;
    ringRef.current = new Float32Array(0);
    writeRef.current = 0;
    filledRef.current = 0;
    setAnalyser(null);
    setRunning(false);
    setStatus('Stopped.');
  }, []);

  // Always release the microphone on unmount, or the browser keeps showing the
  // recording indicator after the user navigates away.
  useEffect(() => () => stop(), [stop]);

  const tick = useCallback(async () => {
    if (inflight.current) return;
    const need = Math.floor(WINDOW_SECONDS * TARGET_SAMPLE_RATE);
    const ring = ringRef.current;
    if (filledRef.current < need * 0.6) return;

    inflight.current = true;
    try {
      // Unwrap the ring into contiguous order, newest last.
      const window = new Float32Array(Math.min(need, filledRef.current));
      const start = (writeRef.current - window.length + ring.length) % ring.length;
      for (let i = 0; i < window.length; i++) {
        window[i] = ring[(start + i) % ring.length];
      }
      const vector = await embed(window);
      const response = await fetch('/api/match?top_k=3', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ embedding: Array.from(vector) }),
      });
      const payload = await response.json();
      if (!payload.ok) throw new Error(payload.error);

      const id = payload.identification;
      setFrames((existing) =>
        [
          ...existing,
          {
            at: Date.now(),
            speaker: id.matched ? id.speaker : id.closest,
            score: id.score ?? 0,
            matched: Boolean(id.matched),
          },
        ].slice(-RIBBON_LENGTH),
      );
    } catch (cause) {
      setError(String(cause));
    } finally {
      inflight.current = false;
    }
  }, []);

  const start = useCallback(async () => {
    setError(null);
    setFrames([]);
    try {
      const handle = await startCapture();
      handleRef.current = handle;
      setAnalyser(handle.analyser);
      setRunning(true);
      setStatus('Listening.');

      // A second tap on the same stream keeps a rolling buffer for the encoder,
      // independent of the analyser feeding the spectrogram.
      const source = handle.context.createMediaStreamSource(handle.stream);
      const processor = handle.context.createScriptProcessor(4096, 1, 1);
      const keep = Math.floor(WINDOW_SECONDS * handle.context.sampleRate * 1.2);
      ringRef.current = new Float32Array(keep);
      writeRef.current = 0;
      filledRef.current = 0;

      processor.onaudioprocess = (event) => {
        const chunk = event.inputBuffer.getChannelData(0);
        const ring = ringRef.current;
        // Copy the chunk in at the cursor, wrapping once if it straddles the
        // end. O(chunk), zero allocation, no matter how long recording runs.
        for (let i = 0; i < chunk.length; i++) {
          ring[(writeRef.current + i) % ring.length] = chunk[i];
        }
        writeRef.current = (writeRef.current + chunk.length) % ring.length;
        filledRef.current = Math.min(ring.length, filledRef.current + chunk.length);
      };

      const mute = handle.context.createGain();
      mute.gain.value = 0;
      source.connect(processor);
      processor.connect(mute);
      mute.connect(handle.context.destination);
      nodesRef.current = { processor, mute };

      timerRef.current = window.setInterval(() => void tick(), STEP_MS);
    } catch (cause) {
      const name = (cause as DOMException)?.name;
      setError(
        name === 'NotAllowedError'
          ? 'Microphone permission was denied. Allow it in your browser to use live identification.'
          : `Could not open the microphone: ${(cause as Error)?.message ?? cause}`,
      );
    }
  }, [tick]);

  const latest = frames[frames.length - 1];
  const stable = frames.slice(-6);
  const steady =
    stable.length >= 4 && stable.every((f) => f.speaker === stable[stable.length - 1].speaker);

  return (
    <div className="view">
      <header className="view__head">
        <span className="eyebrow">Live</span>
        <h1>Who is speaking now</h1>
        <p>
          A {WINDOW_SECONDS}-second rolling window, re-identified every {STEP_MS} milliseconds. The
          encoder runs in your browser and only the resulting 192 numbers are sent for matching, so
          your audio never leaves this machine.
        </p>
      </header>

      <SonagramStrip
        state={running ? 'recording' : 'idle'}
        analyser={analyser}
        theme={theme}
        height={160}
        label="Live microphone input"
      />

      <div className="transport">
        <button
          type="button"
          className="btn btn--record"
          data-recording={running}
          disabled={!meta}
          onClick={() => (running ? stop() : void start())}
        >
          <span className="btn__dot" aria-hidden="true" />
          {running ? 'Stop listening' : 'Start listening'}
        </button>
      </div>

      {error ? <p className="error" role="alert">{error}</p> : null}
      <p className="note" style={{ marginTop: 'var(--space-4)' }} aria-live="polite">{status}</p>

      <div className="verdict" aria-live="polite">
        <span
          className="verdict__name"
          style={{ color: latest?.matched ? undefined : 'var(--ink-dim)' }}
        >
          {latest ? (latest.matched ? latest.speaker : 'Listening…') : 'Nobody yet'}
        </span>
        {latest ? (
          <span className="verdict__tag" style={{ color: confidenceTextColor((latest.score + 1) / 2, theme) }}>
            {latest.score.toFixed(3)}
          </span>
        ) : null}
        {steady ? <span className="verdict__tag">steady</span> : null}
      </div>

      <section className="panel" style={{ marginTop: 'var(--space-4)' }}>
        <div className="panel__title">
          <span className="eyebrow">Confidence over time</span>
          <span className="mono" style={{ fontSize: 'var(--step--2)' }}>{frames.length} windows</span>
        </div>

        {/* Each column is one decision. Height is similarity, colour is the
            speaker, so a stable identification reads as a flat band of one hue
            and a confused stretch reads as flicker. */}
        <div
          style={{ display: 'flex', alignItems: 'flex-end', gap: '2px', height: '120px' }}
          role="img"
          aria-label={`Ribbon of ${frames.length} identification windows`}
        >
          {frames.map((frame) => {
            const height = Math.max(4, ((frame.score + 1) / 2) * 100);
            const hue = frame.speaker
              ? rgbCss(ramp(0.25 + (Math.abs(hash(frame.speaker)) % 100) / 140))
              : 'var(--rule)';
            return (
              <span
                key={frame.at}
                title={`${frame.speaker ?? 'no match'} · ${frame.score.toFixed(3)}`}
                style={{
                  flex: 1,
                  minWidth: '3px',
                  height: `${height}%`,
                  background: hue,
                  opacity: frame.matched ? 1 : 0.3,
                  borderRadius: '1px',
                }}
              />
            );
          })}
          {frames.length === 0 ? (
            /* A ghost of the ribbon it will become, so the panel reads as
               waiting rather than broken. */
            <div className="ribbon-empty" aria-hidden="true">
              {Array.from({ length: 48 }).map((_, i) => (
                <span key={i} style={{ height: `${18 + Math.abs(Math.sin(i * 0.7)) * 52}%` }} />
              ))}
            </div>
          ) : null}
        </div>

        <p className="note" style={{ marginTop: 'var(--space-4)' }}>
          Faded columns are windows the system declined to identify — usually silence between
          sentences. Colour is the speaker, so a steady identification reads as one flat band and a
          confused stretch reads as flicker.
        </p>
      </section>
    </div>
  );
}

/** Stable colour per speaker name, so the ribbon keeps its hue across windows. */
function hash(text: string): number {
  let h = 0;
  for (let i = 0; i < text.length; i++) h = (h << 5) - h + text.charCodeAt(i);
  return h;
}
