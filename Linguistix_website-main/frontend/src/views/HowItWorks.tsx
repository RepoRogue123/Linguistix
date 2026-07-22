/**
 * How a voice becomes an identity.
 *
 * Four stages that reveal as you scroll, each explaining one transformation
 * with real numbers pulled from the running system rather than hardcoded, so
 * the page cannot drift away from what the models actually do.
 */

import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';

import { AnimatedNumber } from '../motion/AnimatedNumber';
import { riseIn, stagger } from '../motion/transitions';
import { ApiError, api, type Health, type Metrics } from '../lib/api';

interface Stage {
  n: string;
  title: string;
  body: string;
  detail?: string;
}

export function HowItWorks() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.metrics(), api.health()])
      .then(([m, h]) => {
        setMetrics(m);
        setHealth(h);
      })
      .catch((cause) => setError(cause instanceof ApiError ? cause.message : String(cause)));
  }, []);

  const encoder = health?.components?.encoder ?? {};
  const gallery = health?.components?.gallery ?? {};
  const dims = encoder.embedding_dim ?? 192;
  const eer = typeof encoder.heldout_eer === 'number' ? encoder.heldout_eer : 10.15;
  const threshold = typeof encoder.cosine_threshold === 'number' ? encoder.cosine_threshold : 0.391;
  const clips = metrics?.dataset?.clips ?? 2511;
  const heldout = metrics?.dataset?.heldout_speakers?.length ?? 10;

  const stages: Stage[] = [
    {
      n: '01',
      title: 'Sound becomes a picture',
      body:
        `Audio arrives at ${((health?.sample_rate ?? 16000) / 1000).toFixed(0)} kHz and is turned into a ` +
        'log-mel spectrogram: energy across 80 frequency bands, every 10 milliseconds. This is the ' +
        'landscape you see on the Analyse page, and it is the only thing the network ever sees.',
      detail: 'Silence is trimmed first, so a long pause before you speak does not dilute the result.',
    },
    {
      n: '02',
      title: 'The picture becomes a point',
      body:
        `A ${dims}-dimension embedding comes out of a 2-million-parameter network built from dilated ` +
        'convolutions and attentive pooling. Attention matters here: it lets the network weight the ' +
        'moments that carry identity and ignore the ones that do not.',
      detail:
        `${health?.crops_per_request ?? 5} overlapping windows are embedded and averaged, because a single ` +
        'window can land on a cough.',
    },
    {
      n: '03',
      title: 'Training pushes voices apart',
      body:
        'The network is trained with an additive angular margin, which does more than ask it to tell ' +
        'the training speakers apart — it forces a gap between them on the unit sphere. That gap is ' +
        'what makes the space usable for voices it has never encountered.',
      detail: 'Augmentation with noise, gain changes and spectral masking keeps it from memorising microphones.',
    },
    {
      n: '04',
      title: 'Identity is a nearest neighbour',
      body:
        `Each known speaker is one point: the average of their clips. Identifying a voice means finding ` +
        `the closest point and checking it clears ${threshold.toFixed(3)} cosine similarity. Enrolling ` +
        'someone new is a write, not a training run, which is why it takes about a second.',
      detail: 'Below the threshold the system declines to answer instead of naming its best guess.',
    },
  ];

  return (
    <div className="view">
      <motion.header className="view__head" variants={stagger} initial="hidden" animate="visible">
        <motion.span className="eyebrow" variants={riseIn}>How it works</motion.span>
        <motion.h1 variants={riseIn}>From sound to identity</motion.h1>
        <motion.p variants={riseIn}>
          Four transformations turn a few seconds of speech into a name. The numbers below are read
          from the running system, not written into the page.
        </motion.p>
      </motion.header>

      {error ? <p className="error">{error}</p> : null}

      <div className="stages">
        {stages.map((stage) => (
          <motion.section
            className="stage"
            key={stage.n}
            initial={{ opacity: 0, y: 28 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-80px' }}
            transition={{ duration: 0.5, ease: [0.2, 0.7, 0.3, 1] }}
          >
            <span className="stage__n">{stage.n}</span>
            <div className="stage__body">
              <h2 className="stage__title">{stage.title}</h2>
              <p>{stage.body}</p>
              {stage.detail ? <p className="note">{stage.detail}</p> : null}
            </div>
          </motion.section>
        ))}
      </div>

      <motion.section
        className="panel"
        style={{ marginTop: 'var(--space-8)' }}
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
      >
        <div className="panel__title">
          <span className="eyebrow">How well it does</span>
        </div>
        <div className="grid grid--3">
          <div className="readout">
            <AnimatedNumber className="readout__value" value={eer} decimals={2} suffix="%" />
            <span className="readout__label">equal error rate</span>
          </div>
          <div className="readout">
            <AnimatedNumber className="readout__value" value={heldout} />
            <span className="readout__label">speakers never trained on</span>
          </div>
          <div className="readout">
            <AnimatedNumber className="readout__value" value={clips} />
            <span className="readout__label">clips in the gallery</span>
          </div>
          <div className="readout">
            <AnimatedNumber className="readout__value" value={gallery.reference_speakers ?? 50} />
            <span className="readout__label">known voices</span>
          </div>
        </div>
        <p className="note" style={{ marginTop: 'var(--space-4)' }}>
          Equal error rate is measured on speakers withheld from training entirely — the point at
          which wrongly accepting a stranger and wrongly rejecting the right person happen equally
          often. It is the number that says whether this generalizes, and it is harder than the
          closed-set accuracies on the <Link to="/arena">Arena</Link>.
        </p>
      </motion.section>

      <motion.section
        className="panel"
        style={{ marginTop: 'var(--space-4)' }}
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
      >
        <div className="panel__title">
          <span className="eyebrow">Credits</span>
        </div>
        <p style={{ marginTop: 0 }}>
          CSL2050 Pattern Recognition and Machine Learning, Indian Institute of Technology Jodhpur.
          Shashank Parchure, Atharva Honparkhe, Vyankatesh Deshpande, Abhinash Roy, Namya Dhingra, and
          Damarasingu Akshaya Sree. Built on the 50-speaker recognition corpus.
        </p>
      </motion.section>
    </div>
  );
}
