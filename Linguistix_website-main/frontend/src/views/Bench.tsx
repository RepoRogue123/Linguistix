/**
 * The bench: record or upload a voice, and see who the model thinks it is.
 *
 * The voiceprint terrain is the hero — simultaneously the input meter, the
 * progress indicator, and the result. Everything below it is present on arrival
 * rather than waiting for a result, so the page explains itself to a visitor
 * who has not pressed anything yet.
 */

import { Suspense, lazy, useCallback, useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';

import { CandidateBars } from '../components/CandidateBars';
import { PipelineStrip } from '../components/PipelineStrip';
import { StatTiles } from '../components/StatTiles';
import { Voiceprint } from '../components/Voiceprint';
import { useRecorder } from '../hooks/useRecorder';
import { AnimatedNumber } from '../motion/AnimatedNumber';
import { riseIn, springPunchy, stagger } from '../motion/transitions';
import type { StripState } from '../components/SonagramStrip';
import { ApiError, api, type ExplainResult, type Health, type IdentifyResult, type JuryResult } from '../lib/api';

// Lazy so the orb's WebGL context is only created once there is a result to
// show, and never at all under reduced motion.
const ConfidenceOrb = lazy(() =>
  import('../three/ConfidenceOrb').then((m) => ({ default: m.ConfidenceOrb })),
);

/**
 * How long the scan runs before the verdict is revealed.
 *
 * Deliberately longer than the server needs. Identification comes back in about
 * 70 ms, which is too fast to read as anything having happened — the answer
 * simply appeared. Holding the sweep makes the measurement legible: the surface
 * is scanned, and then it resolves.
 */
const SCAN_MS = 1600;

interface Props {
  theme: 'scope' | 'print';
  health: Health | null;
}

export function Bench({ theme, health }: Props) {
  const recorder = useRecorder(30);
  const [state, setState] = useState<StripState>('idle');
  const [result, setResult] = useState<IdentifyResult | null>(null);
  const [explanation, setExplanation] = useState<ExplainResult | null>(null);
  const [jury, setJury] = useState<JuryResult | null>(null);
  const [clip, setClip] = useState<Blob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  // The verdict is gated on this rather than on `result` existing. The server
  // answers in about 70 ms, so binding the verdict to the response put the
  // answer on screen before the scan had even started — the sweep then played
  // over an already-revealed result, which is backwards.
  const [revealed, setRevealed] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const reduced = useReducedMotion();

  const analyse = useCallback(
    async (blob: Blob) => {
      setBusy(true);
      setError(null);
      setExplanation(null);
      setJury(null);
      setResult(null);
      setRevealed(false);
      setClip(blob);
      setState('analysing');

      const startedAt = performance.now();

      try {
        const identified = await api.identify(blob, 6);
        // The spectrogram is set immediately so the scan has the real clip to
        // sweep across; only the verdict waits.
        setResult(identified);

        // Always let the scan finish. Under reduced motion there is nothing to
        // watch, so the answer arrives as soon as it exists.
        const elapsed = performance.now() - startedAt;
        const wait = reduced ? 0 : Math.max(0, SCAN_MS - elapsed);

        window.setTimeout(() => {
          setState('resolved');
          setRevealed(true);
        }, wait);
      } catch (cause) {
        setError(cause instanceof ApiError ? cause.message : String(cause));
        setState('idle');
      } finally {
        setBusy(false);
      }
    },
    [reduced],
  );

  /** The bundled clip, so a visitor with no microphone can still see it work. */
  const trySample = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const response = await fetch('/sample.wav');
      if (!response.ok) throw new Error('Sample clip is missing from the build.');
      await analyse(await response.blob());
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
      setBusy(false);
    }
  }, [analyse]);

  const onStop = useCallback(async () => {
    const blob = await recorder.stop();
    if (blob) void analyse(blob);
  }, [recorder, analyse]);

  useEffect(() => {
    if (recorder.recording) setState('recording');
  }, [recorder.recording]);

  const explain = useCallback(async () => {
    if (!clip) return;
    setBusy(true);
    try {
      setExplanation(await api.explain(clip));
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : String(cause));
    } finally {
      setBusy(false);
    }
  }, [clip]);

  const convene = useCallback(async () => {
    if (!clip) return;
    setBusy(true);
    try {
      setJury(await api.jury(clip));
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : String(cause));
    } finally {
      setBusy(false);
    }
  }, [clip]);

  // One narrowed value, so everything inside the result block reads from a
  // source TypeScript can prove is present.
  const shown = revealed ? result : null;
  const identification = shown?.identification;
  const scanning = state === 'analysing';
  const encoderReady = health?.components?.encoder?.loaded !== false;

  return (
    <div className="view">
      <motion.header className="view__head" variants={stagger} initial="hidden" animate="visible">
        <motion.span className="eyebrow" variants={riseIn}>Analyse</motion.span>
        <motion.h1 variants={riseIn}>Who is speaking?</motion.h1>
        <motion.p variants={riseIn}>
          Fifty known voices, plus anyone you enrol. The encoder turns a few seconds of speech into a
          192-dimension point and finds its nearest neighbours. Nothing is retrained to add a person.
        </motion.p>
      </motion.header>

      <Voiceprint
        state={state}
        analyser={recorder.analyser}
        spectrogram={result?.spectrogram?.data ?? null}
        saliency={explanation?.saliency ?? null}
        theme={theme}
        height={360}
      />

      <div className="transport">
        <button
          type="button"
          className="btn btn--record"
          data-recording={recorder.recording}
          disabled={busy || !encoderReady}
          onClick={() => (recorder.recording ? void onStop() : void recorder.start())}
        >
          <span className="btn__dot" aria-hidden="true" />
          {recorder.recording ? `Stop · ${recorder.seconds.toFixed(1)}s` : 'Record'}
        </button>

        <button
          type="button"
          className="btn btn--primary"
          disabled={busy || recorder.recording || !encoderReady}
          onClick={() => void trySample()}
        >
          Try a sample
        </button>

        <button
          type="button"
          className="btn"
          disabled={busy || recorder.recording || !encoderReady}
          onClick={() => fileRef.current?.click()}
        >
          Upload audio
        </button>
        <input
          ref={fileRef}
          type="file"
          accept="audio/*,.wav,.flac,.ogg,.mp3,.m4a"
          className="visually-hidden"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) void analyse(file);
            event.target.value = '';
          }}
        />

        {result && !busy ? (
          <>
            <button type="button" className="btn btn--ghost" onClick={() => void explain()}>
              {explanation ? 'Saliency on' : 'Explain this'}
            </button>
            <button type="button" className="btn btn--ghost" onClick={() => void convene()}>
              {jury ? 'Jury convened' : 'Ask every model'}
            </button>
          </>
        ) : null}
      </div>

      {recorder.error ? <p className="error" role="alert">{recorder.error}</p> : null}
      {error ? <p className="error" role="alert">{error}</p> : null}
      {!encoderReady ? (
        <p className="note note--warn">
          The encoder did not load, so identification is unavailable. Run{' '}
          <code>python training/train_encoder.py</code> to build it.
        </p>
      ) : null}

      <AnimatePresence>
        {scanning ? (
          <motion.div
            className="scanline"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            role="status"
          >
            <span className="eyebrow">Scanning</span>
            <span className="scanline__track" aria-hidden="true">
              <motion.span
                className="scanline__fill"
                initial={{ scaleX: 0 }}
                animate={{ scaleX: 1 }}
                transition={{ duration: SCAN_MS / 1000, ease: 'linear' }}
              />
            </span>
            <span className="scanline__label mono">
              {result ? 'matching against the gallery' : 'embedding 5 windows'}
            </span>
          </motion.div>
        ) : null}
      </AnimatePresence>

      <div aria-live="polite">
        <AnimatePresence mode="wait">
          {identification ? (
            <motion.div
              key="result"
              variants={stagger}
              initial="hidden"
              animate="visible"
              // Leaves fast. A lingering fade meant the previous answer sat at
              // full opacity for the first moments of the next scan, where it
              // could be misread as the new result.
              exit={{ opacity: 0, transition: { duration: 0.12 } }}
            >
              <div className={`verdict${identification.matched ? '' : ' verdict--rejected'}`}>
                <motion.span
                  className="verdict__name"
                  initial={{ opacity: 0, y: 18, scale: 0.94 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  transition={springPunchy}
                >
                  {identification.matched
                    ? identification.speaker
                    : identification.reason === 'not_speech'
                      ? "That isn't speech"
                      : 'No confident match'}
                </motion.span>
                {identification.matched && identification.source ? (
                  <motion.span
                    className={`verdict__tag${identification.source === 'enrolled' ? ' is-enrolled' : ''}`}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.18 }}
                  >
                    {identification.source}
                  </motion.span>
                ) : null}
                {!identification.matched && identification.closest ? (
                  <span className="verdict__tag">closest: {identification.closest}</span>
                ) : null}
              </div>

              {!identification.matched ? (
                <motion.p className="note note--warn" variants={riseIn}>
                  {identification.explanation ??
                    `The nearest speaker scored ${identification.score?.toFixed(3)}, below the ${identification.threshold.toFixed(3)} ` +
                      `acceptance threshold. That threshold is the equal-error-rate operating point measured on speakers ` +
                      `held out of training, so rejecting is a calibrated decision rather than a guess.`}
                </motion.p>
              ) : null}

              <div className="grid grid--2" style={{ marginTop: 'var(--space-6)' }}>
                <motion.section className="panel" variants={riseIn}>
                  <div className="panel__title">
                    <span className="eyebrow">Candidates</span>
                    <span className="mono" style={{ fontSize: 'var(--step--2)' }}>
                      threshold {identification.threshold.toFixed(3)}
                    </span>
                  </div>
                  <CandidateBars
                    candidates={identification.candidates}
                    threshold={identification.threshold}
                    theme={theme}
                  />
                </motion.section>

                <motion.section className="panel panel--orb" variants={riseIn}>
                  <div className="panel__title">
                    <span className="eyebrow">Measurement</span>
                  </div>

                  <div className="orb-row">
                    {!reduced && identification.score !== undefined ? (
                      <Suspense fallback={<div style={{ width: 132, height: 132 }} />}>
                        <ConfidenceOrb
                          score={identification.score}
                          threshold={identification.threshold}
                          accepted={identification.matched}
                          theme={theme}
                          size={132}
                        />
                      </Suspense>
                    ) : null}

                    <div className="grid grid--3" style={{ flex: 1 }}>
                      <div className="readout">
                        <AnimatedNumber className="readout__value" value={shown.audio.voiced_seconds} decimals={1} suffix="s" />
                        <span className="readout__label">voiced</span>
                      </div>
                      <div className="readout">
                        <AnimatedNumber className="readout__value" value={identification.margin ?? 0} decimals={3} />
                        <span className="readout__label">margin</span>
                      </div>
                      <div className="readout">
                        <AnimatedNumber className="readout__value" value={shown.audio.crop_consistency * 100} decimals={0} suffix="%" />
                        <span className="readout__label">crop agreement</span>
                      </div>
                      <div className="readout">
                        <AnimatedNumber className="readout__value" value={(shown.audio.speech_likeness ?? 1) * 100} decimals={0} suffix="%" />
                        <span className="readout__label">speech-like</span>
                      </div>
                      <div className="readout">
                        <AnimatedNumber className="readout__value" value={shown.total_ms} decimals={0} suffix="ms" />
                        <span className="readout__label">round trip</span>
                      </div>
                    </div>
                  </div>

                  <p className="note" style={{ marginTop: 'var(--space-4)' }}>
                    Crop agreement is how closely {shown.audio.crops} windows from this clip agreed with
                    each other. A low value means the clip changed partway through — background noise, or
                    more than one person.
                  </p>
                </motion.section>
              </div>

              {explanation ? (
                <motion.section className="panel" style={{ marginTop: 'var(--space-4)' }} variants={riseIn} layout>
                  <div className="panel__title">
                    <span className="eyebrow">What identified this voice</span>
                    <span className="mono" style={{ fontSize: 'var(--step--2)' }}>{explanation.method}</span>
                  </div>
                  <p style={{ marginTop: 0 }}>
                    Strongest response around{' '}
                    <strong className="mono">
                      {explanation.top_bands_hz.slice(0, 3).map((b) => `${b.hz} Hz`).join(', ')}
                    </strong>
                    . Those ridges are now raised and brightened on the terrain above.
                  </p>
                  <p className="note">{explanation.caveat}</p>
                </motion.section>
              ) : null}

              {jury ? (
                <motion.section className="panel" style={{ marginTop: 'var(--space-4)' }} variants={riseIn} layout>
                  <div className="panel__title">
                    <span className="eyebrow">The jury</span>
                    <span className="mono" style={{ fontSize: 'var(--step--2)' }}>
                      {jury.consensus.votes_for}/{jury.consensus.total_voters} agree
                      {jury.consensus.unanimous ? ' · unanimous' : ''}
                    </span>
                  </div>

                  <p style={{ marginTop: 0 }}>
                    {jury.consensus.unanimous
                      ? `All ${jury.consensus.total_voters} models independently answered ${jury.consensus.speaker}. ` +
                        `Agreement across models this different is a stronger signal than any single confidence score.`
                      : `${jury.consensus.distinct_answers} different answers across ${jury.consensus.total_voters} ` +
                        `models, with ${jury.consensus.speaker} leading at ${jury.consensus.agreement}%. ` +
                        `Split verdicts mark the genuinely hard clips.`}
                  </p>

                  <div className="scroll-x">
                    <table className="table">
                      <thead>
                        <tr>
                          <th>Model</th>
                          <th>Kind</th>
                          <th>Answer</th>
                          <th className="num">Confidence</th>
                          <th className="num">ms</th>
                        </tr>
                      </thead>
                      <tbody>
                        {jury.votes.map((vote, i) => (
                          <motion.tr
                            key={vote.key}
                            initial={{ opacity: 0, x: -8 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: i * 0.035 }}
                          >
                            <td>{vote.model}</td>
                            <td className="mono">{vote.family}</td>
                            <td className={vote.speaker === jury.consensus.speaker ? undefined : 'mono'}>
                              {vote.speaker ?? '—'}
                            </td>
                            <td className="num">
                              {vote.confidence !== null ? `${vote.confidence.toFixed(1)}%` : '—'}
                              {!vote.calibrated ? '*' : ''}
                            </td>
                            <td className="num">{vote.ms.toFixed(0)}</td>
                          </motion.tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  <p className="note" style={{ marginTop: 'var(--space-4)' }}>
                    * Not a calibrated probability. SVM reports a decision margin and the encoder reports
                    cosine similarity; both are squashed to a percentage for display only.
                  </p>
                </motion.section>
              ) : null}
            </motion.div>
          ) : (
            /* The idle state is the point of this rework: a visitor who has
               pressed nothing still gets the numbers and the explanation. */
            <motion.div key="idle" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              {/* Hidden mid-scan: flashing the marketing panels back between
                  pressing Record and seeing a verdict reads as a glitch. */}
              {!scanning ? (
                <>
                  <StatTiles health={health} />
                  <PipelineStrip />
                </>
              ) : null}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
