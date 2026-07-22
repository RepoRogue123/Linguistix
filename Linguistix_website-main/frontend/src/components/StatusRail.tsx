/**
 * The status strip, in the spirit of the readouts on real lab hardware.
 *
 * Reports what is actually loaded rather than a decorative "online" light.
 * When a component fails to load the API says so per component, and this shows
 * that instead of a single green dot that would be lying.
 */

import type { Health } from '../lib/api';

interface Props {
  health: Health | null;
  error: string | null;
  onRetry: () => void;
}

function Row({ label, value, tone }: { label: string; value: string; tone?: 'ok' | 'warn' | 'bad' }) {
  return (
    <div className="status__row">
      <span className="status__label">{label}</span>
      <span className={`status__value num${tone ? ` is-${tone}` : ''}`}>{value}</span>
    </div>
  );
}

export function StatusRail({ health, error, onRetry }: Props) {
  if (error) {
    return (
      <section className="status" aria-label="System status">
        <span className="eyebrow">Status</span>
        <p className="status__error">{error}</p>
        <button type="button" className="btn btn--ghost btn--sm" onClick={onRetry}>
          Retry
        </button>
      </section>
    );
  }

  if (!health) {
    return (
      <section className="status" aria-label="System status" aria-busy="true">
        <span className="eyebrow">Status</span>
        <Row label="Connecting" value="…" />
      </section>
    );
  }

  const encoder = health.components?.encoder ?? {};
  const gallery = health.components?.gallery ?? {};
  const eer = typeof encoder.heldout_eer === 'number' ? `${encoder.heldout_eer.toFixed(2)}%` : '—';

  return (
    <section className="status" aria-label="System status">
      <span className="eyebrow">Status</span>

      <Row
        label="Engine"
        value={health.status === 'ok' ? 'ready' : 'degraded'}
        tone={health.status === 'ok' ? 'ok' : 'warn'}
      />
      <Row label="Encoder" value={encoder.loaded ? `${encoder.embedding_dim}-d` : 'absent'} tone={encoder.loaded ? undefined : 'bad'} />
      {/* The number that actually says how good this is: error rate on speakers
          the encoder never saw during training. */}
      <Row label="Held-out EER" value={eer} />
      <Row label="Rate" value={`${(health.sample_rate / 1000).toFixed(0)} kHz`} />
      <Row label="Known" value={String(gallery.reference_speakers ?? 0)} />
      <Row
        label="Enrolled"
        value={String(gallery.enrolled_speakers ?? 0)}
        tone={gallery.enrolled_speakers ? 'ok' : undefined}
      />
      <Row label="Window" value={`${health.crop_seconds}s × ${health.crops_per_request}`} />

      {!encoder.loaded && encoder.error ? <p className="status__error">{encoder.error}</p> : null}

      {gallery.enrolled_speakers > 0 ? (
        <p className="status__note">Enrolments reset when the server restarts.</p>
      ) : null}
    </section>
  );
}
