/**
 * Enrolment: teach the system a voice it has never heard.
 *
 * A speaker is a centroid in embedding space, so adding one is a write rather
 * than a training run, and it takes about a second.
 */

import { useCallback, useEffect, useState } from 'react';

import { SonagramStrip } from '../components/SonagramStrip';
import { useRecorder } from '../hooks/useRecorder';
import { ApiError, api, type GalleryState } from '../lib/api';

interface Props {
  theme: 'scope' | 'print';
  onChanged: () => void;
}

const RECOMMENDED_CLIPS = 3;

export function Enroll({ theme, onChanged }: Props) {
  const recorder = useRecorder(20);
  const [name, setName] = useState('');
  const [clips, setClips] = useState<Blob[]>([]);
  const [gallery, setGallery] = useState<GalleryState | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [warning, setWarning] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setGallery(await api.gallery());
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : String(cause));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const addClip = useCallback(async () => {
    const blob = await recorder.stop();
    if (blob) setClips((existing) => [...existing, blob]);
  }, [recorder]);

  const submit = useCallback(async () => {
    if (!name.trim() || clips.length === 0) return;
    setBusy(true);
    setError(null);
    setWarning(null);
    setMessage(null);

    try {
      const response = await api.enroll(name.trim(), clips);
      setMessage(
        `${response.enrolled.name} enrolled from ${response.enrolled.samples} clip${
          response.enrolled.samples === 1 ? '' : 's'
        }. Go to Analyse and record again to test it.`,
      );
      if (response.warning) setWarning(response.warning);
      setClips([]);
      setName('');
      await refresh();
      onChanged();
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : String(cause));
    } finally {
      setBusy(false);
    }
  }, [name, clips, refresh, onChanged]);

  const remove = useCallback(
    async (speaker: string) => {
      try {
        await api.removeSpeaker(speaker);
        await refresh();
        onChanged();
      } catch (cause) {
        setError(cause instanceof ApiError ? cause.message : String(cause));
      }
    },
    [refresh, onChanged],
  );

  return (
    <div className="view">
      <header className="view__head">
        <span className="eyebrow">Enrol</span>
        <h1>Teach it a voice</h1>
        <p>
          Record yourself a few times and the system will recognise you alongside the fifty dataset
          speakers. No retraining happens: your voice becomes a point in the same embedding space, and
          identification is a nearest-neighbour search.
        </p>
      </header>

      <SonagramStrip
        state={recorder.recording ? 'recording' : 'idle'}
        analyser={recorder.analyser}
        theme={theme}
        height={140}
        label="Live input while enrolling"
      />

      <div className="transport">
        <button
          type="button"
          className="btn btn--record"
          data-recording={recorder.recording}
          disabled={busy}
          onClick={() => (recorder.recording ? void addClip() : void recorder.start())}
        >
          <span className="btn__dot" aria-hidden="true" />
          {recorder.recording ? `Save clip · ${recorder.seconds.toFixed(1)}s` : 'Record a clip'}
        </button>
        {clips.length > 0 ? (
          <button type="button" className="btn btn--ghost" onClick={() => setClips([])} disabled={busy}>
            Clear {clips.length} clip{clips.length === 1 ? '' : 's'}
          </button>
        ) : null}
      </div>

      {recorder.error ? <p className="error" role="alert">{recorder.error}</p> : null}

      <div className="grid grid--2" style={{ marginTop: 'var(--space-6)' }}>
        <section className="panel">
          <div className="panel__title">
            <span className="eyebrow">New speaker</span>
            <span className="mono" style={{ fontSize: 'var(--step--2)' }}>
              {clips.length}/{RECOMMENDED_CLIPS} clips
            </span>
          </div>

          <div className="field">
            <label htmlFor="speaker-name">Name</label>
            <input
              id="speaker-name"
              type="text"
              value={name}
              maxLength={64}
              placeholder="e.g. Atharva"
              onChange={(event) => setName(event.target.value)}
            />
          </div>

          <p className="note" style={{ marginTop: 'var(--space-4)' }}>
            {clips.length < RECOMMENDED_CLIPS
              ? `Three short clips work better than one long one. Varying what you say and how you sit ` +
                `gives a centroid that survives you sounding slightly different tomorrow.`
              : `That is enough. More clips keep tightening the centroid, since re-enrolling merges into ` +
                `a running mean rather than overwriting.`}
          </p>

          <button
            type="button"
            className="btn btn--primary"
            style={{ marginTop: 'var(--space-4)' }}
            disabled={busy || !name.trim() || clips.length === 0}
            onClick={() => void submit()}
          >
            {busy ? 'Enrolling…' : `Enrol from ${clips.length} clip${clips.length === 1 ? '' : 's'}`}
          </button>

          <div aria-live="polite">
            {message ? <p className="note" style={{ marginTop: 'var(--space-4)' }}>{message}</p> : null}
            {warning ? <p className="note note--warn" style={{ marginTop: 'var(--space-3)' }}>{warning}</p> : null}
            {error ? <p className="error" style={{ marginTop: 'var(--space-3)' }} role="alert">{error}</p> : null}
          </div>
        </section>

        <section className="panel">
          <div className="panel__title">
            <span className="eyebrow">Enrolled</span>
            <span className="mono" style={{ fontSize: 'var(--step--2)' }}>
              {gallery?.enrolled_speakers ?? 0} of {(gallery?.enrolled_speakers ?? 0) + (gallery?.reference_speakers ?? 0)}
            </span>
          </div>

          {gallery?.enrolled.length ? (
            <div className="scroll-x">
              <table className="table">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th className="num">Clips</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {gallery.enrolled.map((speaker) => (
                    <tr key={speaker.name}>
                      <td>{speaker.name}</td>
                      <td className="num">{speaker.samples}</td>
                      <td>
                        <button
                          type="button"
                          className="btn btn--ghost btn--sm"
                          onClick={() => void remove(speaker.name)}
                        >
                          Remove
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p style={{ color: 'var(--ink-dim)', margin: 0 }}>
              Nobody enrolled yet. Record a few clips and the system will know one more voice than it
              was trained on.
            </p>
          )}

          <p className="note note--warn" style={{ marginTop: 'var(--space-4)' }}>
            {gallery?.storage_note ??
              'Enrolments live on the container filesystem and reset when the server restarts.'}
          </p>
        </section>
      </div>
    </div>
  );
}
