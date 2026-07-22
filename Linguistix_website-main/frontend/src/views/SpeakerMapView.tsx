/**
 * The embedding space, as a place.
 *
 * Every clip projected from the encoder's own 192-dimension space — the space
 * identification actually runs in, so clusters that look tight really are tight.
 * Rendered as an orbitable point cloud where the third dimension is available;
 * the flat canvas remains for reduced motion and for machines without WebGL.
 */

import { Suspense, lazy, useCallback, useEffect, useRef, useState } from 'react';
import { motion, useReducedMotion } from 'framer-motion';

import { AnimatedNumber } from '../motion/AnimatedNumber';
import { riseIn, stagger } from '../motion/transitions';
import { ApiError, api, type SpeakerMap } from '../lib/api';
import { ramp } from '../lib/colormap';

const EmbeddingGalaxy = lazy(() =>
  import('../three/EmbeddingGalaxy').then((m) => ({ default: m.EmbeddingGalaxy })),
);

interface Props {
  theme: 'scope' | 'print';
}

/** The flat renderer, kept as a genuine alternative rather than a placeholder. */
function FlatMap({ map, theme, showHeldout, onHover }: {
  map: SpeakerMap;
  theme: 'scope' | 'print';
  showHeldout: boolean;
  onHover: (info: { speaker: string; heldout: boolean } | null) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext('2d');
    if (!context) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.floor(rect.width * dpr);
    canvas.height = Math.floor(rect.height * dpr);

    const styles = getComputedStyle(document.documentElement);
    context.fillStyle = styles.getPropertyValue('--bg').trim() || '#070d18';
    context.fillRect(0, 0, canvas.width, canvas.height);

    const pad = 24 * dpr;
    const w = canvas.width - pad * 2;
    const h = canvas.height - pad * 2;
    const radius = Math.max(1.5, 2.2 * dpr);

    context.strokeStyle = styles.getPropertyValue('--grid-line').trim() || 'rgba(255,255,255,0.045)';
    context.lineWidth = 1;
    for (let i = 0; i <= 8; i++) {
      const x = pad + (w * i) / 8;
      const y = pad + (h * i) / 8;
      context.beginPath(); context.moveTo(x, pad); context.lineTo(x, pad + h); context.stroke();
      context.beginPath(); context.moveTo(pad, y); context.lineTo(pad + w, y); context.stroke();
    }

    const total = Math.max(1, map.speakers.length - 1);
    for (const point of map.points) {
      if (point.h && !showHeldout) continue;
      const x = pad + ((point.x + 1) / 2) * w;
      const y = pad + ((1 - point.y) / 2) * h;
      const [r, g, b] = ramp(0.18 + (point.s / total) * 0.78);

      if (point.h) {
        context.strokeStyle = `rgba(${r},${g},${b},0.95)`;
        context.lineWidth = 1.2 * dpr;
        context.beginPath(); context.arc(x, y, radius * 1.35, 0, Math.PI * 2); context.stroke();
      } else {
        context.fillStyle = `rgba(${r},${g},${b},0.72)`;
        context.beginPath(); context.arc(x, y, radius, 0, Math.PI * 2); context.fill();
      }
    }
  }, [map, showHeldout, theme]);

  useEffect(() => {
    draw();
    const observer = new ResizeObserver(draw);
    if (canvasRef.current) observer.observe(canvasRef.current);
    return () => observer.disconnect();
  }, [draw]);

  return (
    <canvas
      ref={canvasRef}
      onMouseMove={(event) => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const rect = canvas.getBoundingClientRect();
        const pad = 24;
        const nx = ((event.clientX - rect.left - pad) / (rect.width - pad * 2)) * 2 - 1;
        const ny = 1 - ((event.clientY - rect.top - pad) / (rect.height - pad * 2)) * 2;
        let best: (typeof map.points)[number] | null = null;
        let bestDistance = 0.05;
        for (const point of map.points) {
          if (point.h && !showHeldout) continue;
          const d = Math.hypot(point.x - nx, point.y - ny);
          if (d < bestDistance) { bestDistance = d; best = point; }
        }
        onHover(best ? { speaker: map.speakers[best.s], heldout: Boolean(best.h) } : null);
      }}
      onMouseLeave={() => onHover(null)}
      style={{ display: 'block', width: '100%', height: 'min(62vh, 560px)', cursor: 'crosshair' }}
      role="img"
      aria-label={`Scatter plot of ${map.points.length} voice clips across ${map.speakers.length} speakers`}
    />
  );
}

export function SpeakerMapView({ theme }: Props) {
  const [map, setMap] = useState<SpeakerMap | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hover, setHover] = useState<{ speaker: string; heldout: boolean } | null>(null);
  const [showHeldout, setShowHeldout] = useState(true);
  const reduced = useReducedMotion();

  useEffect(() => {
    api.map().then(setMap).catch((cause) => {
      setError(cause instanceof ApiError ? cause.message : String(cause));
    });
  }, []);

  // 3D only when the projection actually has a third dimension to show.
  const has3D = Boolean(map?.points?.[0] && 'z' in map.points[0]);
  const use3D = has3D && !reduced;

  if (error) return <div className="view"><p className="error">{error}</p></div>;

  return (
    <div className="view">
      <motion.header className="view__head" variants={stagger} initial="hidden" animate="visible">
        <motion.span className="eyebrow" variants={riseIn}>Map</motion.span>
        <motion.h1 variants={riseIn}>The embedding space</motion.h1>
        <motion.p variants={riseIn}>
          Every clip in the dataset, projected from the 192 dimensions the encoder works in. Each
          cluster is one speaker. The larger, brighter points are the ten speakers it was never
          trained on — that they still hold together is what open-set recognition means.
        </motion.p>
      </motion.header>

      <motion.div className="panel" style={{ padding: 0, overflow: 'hidden' }} variants={riseIn} initial="hidden" animate="visible">
        {map ? (
          use3D ? (
            <Suspense fallback={<div style={{ height: 'min(62vh, 560px)' }} />}>
              <EmbeddingGalaxy
                map={map}
                theme={theme}
                showHeldout={showHeldout}
                onHover={setHover}
                height={Math.min(560, Math.round(window.innerHeight * 0.62))}
              />
            </Suspense>
          ) : (
            <FlatMap map={map} theme={theme} showHeldout={showHeldout} onHover={setHover} />
          )
        ) : (
          <div style={{ height: 'min(62vh, 560px)' }} />
        )}
      </motion.div>

      <div className="transport">
        <button type="button" className="btn btn--ghost btn--sm" onClick={() => setShowHeldout((v) => !v)}>
          {showHeldout ? 'Hide' : 'Show'} held-out speakers
        </button>
        <span className="chip" aria-live="polite">
          {hover ? `${hover.speaker}${hover.heldout ? ' · never trained on' : ''}` : use3D ? 'Drag to orbit' : 'Hover a point'}
        </span>
        <span className="chip">
          <AnimatedNumber value={map?.points.length ?? 0} /> clips
        </span>
        <span className="chip">{map?.projection ?? '—'} · {use3D ? '3D' : '2D'}</span>
      </div>

      {map ? <p className="note" style={{ marginTop: 'var(--space-6)' }}>{map.note}</p> : null}
    </div>
  );
}
