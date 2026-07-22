/**
 * Model arena: every trained model, scored on the same held-out split.
 *
 * The 3D bars are the page: each model's accuracy grows into place on load and
 * the whole field can be orbited. The table underneath is for anyone who wants
 * exact figures rather than the shape of the comparison.
 */

import { Suspense, lazy, useEffect, useMemo, useState } from 'react';
import { motion, useReducedMotion } from 'framer-motion';

import { ApiError, api, type Metrics } from '../lib/api';
import { confidenceTextColor } from '../lib/colormap';
import { AnimatedNumber } from '../motion/AnimatedNumber';
import { riseIn, stagger } from '../motion/transitions';

const LeaderboardBars = lazy(() =>
  import('../three/LeaderboardBars').then((m) => ({ default: m.LeaderboardBars })),
);

type SortKey = 'model' | 'representation' | 'accuracy' | 'macro_f1';

export function Arena({ theme }: { theme: 'scope' | 'print' }) {
  const reduced = useReducedMotion();
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: 'accuracy', desc: true });

  useEffect(() => {
    api.metrics().then(setMetrics).catch((cause) => {
      setError(cause instanceof ApiError ? cause.message : String(cause));
    });
  }, []);

  const rows = useMemo(() => {
    const results = (metrics?.benchmarks?.results ?? []) as any[];
    const supervised = results.filter((r) => r.regime === 'leakfree' && !r.metric);

    return [...supervised].sort((a, b) => {
      const dir = sort.desc ? -1 : 1;
      if (sort.key === 'model') return dir * String(a.model).localeCompare(String(b.model));
      if (sort.key === 'representation') return dir * String(a.representation).localeCompare(String(b.representation));
      if (sort.key === 'macro_f1') return dir * (a.test.macro_f1 - b.test.macro_f1);
      return dir * (a.test.accuracy - b.test.accuracy);
    });
  }, [metrics, sort]);

  const encoderBest = metrics?.encoder_history?.best_eer as number | undefined;

  const bars = useMemo(
    () =>
      rows.map((r: any) => ({
        key: `${r.key}-${r.regime}`,
        model: r.model,
        representation: r.representation,
        accuracy: r.test.accuracy,
      })),
    [rows],
  );

  const toggle = (key: SortKey) =>
    setSort((current) => ({ key, desc: current.key === key ? !current.desc : true }));

  const header = (key: SortKey, label: string, numeric = false) => (
    <th
      className={numeric ? 'num' : undefined}
      aria-sort={sort.key === key ? (sort.desc ? 'descending' : 'ascending') : undefined}
      onClick={() => toggle(key)}
    >
      {label}
      {sort.key === key ? (sort.desc ? ' ↓' : ' ↑') : ''}
    </th>
  );

  if (error) return <div className="view"><p className="error">{error}</p></div>;
  if (!metrics) return <div className="view"><p className="loading eyebrow">Loading metrics…</p></div>;

  return (
    <div className="view">
      <motion.header className="view__head" variants={stagger} initial="hidden" animate="visible">
        <motion.span className="eyebrow" variants={riseIn}>Arena</motion.span>
        <motion.h1 variants={riseIn}>What the models actually score</motion.h1>
        <motion.p variants={riseIn}>
          Every classical model re-run with dimensionality reduction fitted inside the training fold,
          scored on a stratified held-out split of {metrics.dataset?.split_sizes?.test ?? '—'} clips.
        </motion.p>
      </motion.header>

      {!reduced && bars.length ? (
        <motion.section className="panel" style={{ padding: 0, overflow: 'hidden', marginBottom: 'var(--space-4)' }} variants={riseIn} initial="hidden" animate="visible">
          <Suspense fallback={<div style={{ height: 400 }} />}>
            <LeaderboardBars data={bars} theme={theme} height={400} />
          </Suspense>
        </motion.section>
      ) : null}

      <section className="panel">
        <div className="panel__title">
          <span className="eyebrow">Leaderboard</span>
          <span className="mono" style={{ fontSize: 'var(--step--2)' }}>closed set · 50 speakers</span>
        </div>
        <div className="scroll-x">
          <table className="table">
            <thead>
              <tr>
                {header('model', 'Model')}
                {header('representation', 'Reduction')}
                <th className="num">Dims</th>
                {header('accuracy', 'Test accuracy', true)}
                {header('macro_f1', 'Macro F1', true)}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={`${row.key}-${row.regime}`}>
                  <td>{row.model}</td>
                  <td className="mono">{row.representation}</td>
                  <td className="num">{row.dims}</td>
                  <td className="num">
                    <span style={{ color: confidenceTextColor(row.test.accuracy / 100, theme) }}>
                      {row.test.accuracy.toFixed(2)}%
                    </span>
                  </td>
                  <td className="num">{row.test.macro_f1.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="note" style={{ marginTop: 'var(--space-4)' }}>
          Macro F1 is worth reading next to accuracy here. Clips per speaker range from 10 to 120, so a
          model can look respectable on accuracy while failing the speakers with the least data.
        </p>
      </section>

      <section className="panel" style={{ marginTop: 'var(--space-4)' }}>
        <div className="panel__title">
          <span className="eyebrow">The encoder is measured differently</span>
        </div>
        <div className="grid grid--3">
          <div className="readout">
            {encoderBest ? (
              <AnimatedNumber className="readout__value" value={encoderBest} decimals={2} suffix="%" />
            ) : (
              <span className="readout__value">—</span>
            )}
            <span className="readout__label">Held-out EER</span>
          </div>
          <div className="readout">
            <AnimatedNumber className="readout__value" value={metrics.dataset?.heldout_speakers?.length ?? 0} />
            <span className="readout__label">Unseen speakers</span>
          </div>
          <div className="readout">
            <span className="readout__value">∞</span>
            <span className="readout__label">Enrollable speakers</span>
          </div>
        </div>
        <p className="note" style={{ marginTop: 'var(--space-4)' }}>
          The table above is closed-set accuracy: every model must name one of the fifty, and is never
          asked about anyone else. The encoder is scored by equal error rate on speakers withheld from
          training entirely, which is a harder question and the one that matters once anyone can enrol.
          The two numbers are not comparable, so they are not put in the same column.
        </p>
      </section>
    </div>
  );
}
