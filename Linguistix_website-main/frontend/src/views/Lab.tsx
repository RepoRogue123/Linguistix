/**
 * Robustness lab: break the model on purpose and watch it happen.
 *
 * Everything runs in the browser through the exported ONNX graph, so dragging a
 * slider re-embeds instantly instead of queueing a request per frame against a
 * free-tier CPU.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

import { SonagramStrip } from '../components/SonagramStrip';
import { useRecorder } from '../hooks/useRecorder';
import { decodeFileTo16k, TARGET_SAMPLE_RATE } from '../lib/audio';
import { confidenceTextColor } from '../lib/colormap';
import { CLEAN, cosine, degrade, embed, loadEncoder, type Degradation, type EncoderMeta } from '../lib/onnx';

interface Props {
  theme: 'scope' | 'print';
}

const PRESETS: { label: string; hint: string; value: Degradation }[] = [
  { label: 'Clean', hint: 'Untouched reference', value: CLEAN },
  { label: 'Noisy room', hint: '10 dB SNR', value: { ...CLEAN, snrDb: 10 } },
  { label: 'Telephone', hint: '3.4 kHz lowpass', value: { ...CLEAN, lowpassHz: 3400 } },
  { label: 'Resampled', hint: 'Round-tripped through 22.05 kHz', value: { ...CLEAN, viaRate: 22050 } },
  { label: 'Two seconds only', hint: 'Short-clip limit', value: { ...CLEAN, seconds: 2 } },
];

export function Lab({ theme }: Props) {
  const recorder = useRecorder(20);
  const [reference, setReference] = useState<Float32Array | null>(null);
  const [referenceEmbedding, setReferenceEmbedding] = useState<Float32Array | null>(null);
  const [options, setOptions] = useState<Degradation>(CLEAN);
  const [similarity, setSimilarity] = useState<number | null>(null);
  const [meta, setMeta] = useState<EncoderMeta | null>(null);
  const [status, setStatus] = useState<string>('Loading the encoder into your browser…');
  const [error, setError] = useState<string | null>(null);
  const pending = useRef(false);

  useEffect(() => {
    let cancelled = false;

    loadEncoder()
      .then(async ({ meta: m }) => {
        if (cancelled) return;
        setMeta(m);
        // Load the bundled clip straight away. Arriving to a panel of disabled
        // sliders makes the page look broken; this way every control works
        // immediately and the user can swap in their own voice whenever.
        try {
          const response = await fetch('/sample.wav');
          if (!response.ok) throw new Error('sample missing');
          const samples = await decodeFileTo16k(await response.blob());
          if (cancelled) return;
          await setClipRef.current(samples);
          setStatus('Loaded a sample clip. Degrade it, or record your own voice to replace it.');
        } catch {
          if (!cancelled) setStatus(`Encoder running locally (${m.size_mb} MB). Record or upload a clip to begin.`);
        }
      })
      .catch((cause) => !cancelled && setError(`Could not load the in-browser encoder: ${cause}`));

    return () => {
      cancelled = true;
    };
  }, []);

  const setClipRef = useRef<(samples: Float32Array) => Promise<void>>(async () => {});

  const setClip = useCallback(async (samples: Float32Array) => {
    setReference(samples);
    setStatus('Embedding the reference…');
    try {
      const vector = await embed(samples);
      setReferenceEmbedding(vector);
      setSimilarity(1);
      setOptions(CLEAN);
      setStatus('Reference set. Now degrade it and watch the similarity fall.');
    } catch (cause) {
      setError(String(cause));
    }
  }, []);

  setClipRef.current = setClip;

  // Re-embed whenever the sliders move. `pending` coalesces drags so a fast
  // slider does not queue dozens of inferences behind each other.
  useEffect(() => {
    if (!reference || !referenceEmbedding || pending.current) return;
    let cancelled = false;
    pending.current = true;

    (async () => {
      try {
        const vector = await embed(degrade(reference, options));
        if (!cancelled) setSimilarity(cosine(referenceEmbedding, vector));
      } catch (cause) {
        if (!cancelled) setError(String(cause));
      } finally {
        pending.current = false;
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [reference, referenceEmbedding, options]);

  const threshold = meta?.cosine_threshold ?? 0.39;
  const stillMatches = similarity !== null && similarity >= threshold;

  const slider = (
    label: string,
    value: number,
    min: number,
    max: number,
    step: number,
    format: (v: number) => string,
    onChange: (v: number) => void,
    disabledAt?: number,
  ) => (
    <div className="field" key={label}>
      <label htmlFor={`slider-${label}`}>
        {label} <span className="mono">{value === disabledAt ? 'off' : format(value)}</span>
      </label>
      <input
        id={`slider-${label}`}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value === Infinity ? max : value}
        disabled={!reference}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </div>
  );

  return (
    <div className="view">
      <header className="view__head">
        <span className="eyebrow">Lab</span>
        <h1>Break it on purpose</h1>
        <p>
          Record a clip, then degrade it and watch how far the embedding moves. Everything here runs in
          your browser through the exported model, so nothing is uploaded and the sliders respond
          immediately.
        </p>
      </header>

      <SonagramStrip
        state={recorder.recording ? 'recording' : 'idle'}
        analyser={recorder.analyser}
        theme={theme}
        height={140}
        label="Live input for the robustness lab"
      />

      <div className="transport">
        <button
          type="button"
          className="btn btn--record"
          data-recording={recorder.recording}
          disabled={!meta}
          onClick={async () => {
            if (recorder.recording) {
              const blob = await recorder.stop();
              if (blob) void setClip(await decodeFileTo16k(blob));
            } else {
              void recorder.start();
            }
          }}
        >
          <span className="btn__dot" aria-hidden="true" />
          {recorder.recording ? `Use this · ${recorder.seconds.toFixed(1)}s` : 'Record reference'}
        </button>

        <label className="btn" style={{ cursor: 'pointer' }}>
          Upload reference
          <input
            type="file"
            accept="audio/*"
            className="visually-hidden"
            disabled={!meta}
            onChange={async (event) => {
              const file = event.target.files?.[0];
              if (file) void setClip(await decodeFileTo16k(file));
              event.target.value = '';
            }}
          />
        </label>
      </div>

      {error ? <p className="error" role="alert">{error}</p> : null}
      <p className="note" style={{ marginTop: 'var(--space-4)' }} aria-live="polite">{status}</p>

      <div className="grid grid--2" style={{ marginTop: 'var(--space-6)' }}>
        <section className="panel">
          <div className="panel__title">
            <span className="eyebrow">Degradations</span>
          </div>

          {slider('Noise', options.snrDb === Infinity ? 40 : options.snrDb, 0, 40, 1,
            (v) => `${v} dB SNR`,
            (v) => setOptions((o) => ({ ...o, snrDb: v >= 40 ? Infinity : v })), 40)}

          {slider('Lowpass', options.lowpassHz === Infinity ? 8000 : options.lowpassHz, 500, 8000, 100,
            (v) => `${(v / 1000).toFixed(1)} kHz`,
            (v) => setOptions((o) => ({ ...o, lowpassHz: v >= 8000 ? Infinity : v })), 8000)}

          {slider('Speed / pitch', options.speed, 0.7, 1.4, 0.01,
            (v) => `${v.toFixed(2)}x`,
            (v) => setOptions((o) => ({ ...o, speed: v })))}

          {slider('Clip length', options.seconds === Infinity ? 10 : options.seconds, 0.5, 10, 0.5,
            (v) => `${v.toFixed(1)} s`,
            (v) => setOptions((o) => ({ ...o, seconds: v >= 10 ? Infinity : v })), 10)}

          <div className="field" style={{ marginTop: 'var(--space-4)' }}>
            <label htmlFor="via-rate">Resample round trip</label>
            <select
              id="via-rate"
              value={options.viaRate ?? ''}
              disabled={!reference}
              onChange={(event) =>
                setOptions((o) => ({ ...o, viaRate: event.target.value ? Number(event.target.value) : null }))
              }
            >
              <option value="">None</option>
              <option value="22050">via 22.05 kHz</option>
              <option value="8000">via 8 kHz</option>
            </select>
          </div>

          <div className="transport">
            {PRESETS.map((preset) => (
              <button
                key={preset.label}
                type="button"
                className="btn btn--ghost btn--sm"
                disabled={!reference}
                title={preset.hint}
                onClick={() => setOptions(preset.value)}
              >
                {preset.label}
              </button>
            ))}
          </div>
        </section>

        <section className="panel">
          <div className="panel__title">
            <span className="eyebrow">Effect</span>
            <span className="mono" style={{ fontSize: 'var(--step--2)' }}>
              threshold {threshold.toFixed(3)}
            </span>
          </div>

          <div className="readout">
            <span
              className="readout__value"
              style={{
                fontSize: 'var(--step-3)',
                color: similarity === null ? 'var(--ink-faint)' : confidenceTextColor((similarity + 1) / 2, theme),
              }}
            >
              {similarity === null ? '—' : similarity.toFixed(3)}
            </span>
            <span className="readout__label">cosine to the clean reference</span>
          </div>

          {similarity !== null ? (
            <p className={stillMatches ? 'note' : 'note note--warn'} style={{ marginTop: 'var(--space-4)' }}>
              {stillMatches
                ? `Still above the acceptance threshold, so this clip would still be identified correctly.`
                : `Below the ${threshold.toFixed(3)} threshold. At this level of degradation the system would ` +
                  `refuse to identify the speaker rather than guess.`}
            </p>
          ) : null}

          {meta ? (
            <div className="grid grid--3" style={{ marginTop: 'var(--space-6)' }}>
              <div className="readout">
                <span className="readout__value">{meta.embedding_dim}</span>
                <span className="readout__label">dimensions</span>
              </div>
              <div className="readout">
                <span className="readout__value">{(TARGET_SAMPLE_RATE / 1000).toFixed(0)}k</span>
                <span className="readout__label">sample rate</span>
              </div>
              <div className="readout">
                <span className="readout__value">{meta.size_mb}MB</span>
                <span className="readout__label">in your browser</span>
              </div>
            </div>
          ) : null}
        </section>
      </div>
    </div>
  );
}
