/**
 * The front door.
 *
 * Full bleed — no rail, no status column. The sequence plays once and settles,
 * and everything below it is composition: the pipeline strip, the stat tiles and
 * the galaxy already exist and are reused unchanged.
 *
 * The sequence is never a gate. Skip is focusable in the first frame, and a
 * click, a scroll, Escape or any key ends it immediately; reduced motion means
 * it never starts. Someone arriving for the tenth time can be in the instrument
 * in one keystroke.
 */

import { Suspense, lazy, useCallback, useEffect, useState } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import { Link } from 'react-router-dom';

import { PipelineStrip } from '../components/PipelineStrip';
import { StatTiles } from '../components/StatTiles';
import { loadSampleEnvelope, useIntro } from '../lib/useIntro';
import { riseIn, stagger } from '../motion/transitions';
import { ApiError, api, type Health, type SpeakerMap } from '../lib/api';

const IntroSequence = lazy(() =>
  import('../three/IntroSequence').then((m) => ({ default: m.IntroSequence })),
);
const EmbeddingGalaxy = lazy(() =>
  import('../three/EmbeddingGalaxy').then((m) => ({ default: m.EmbeddingGalaxy })),
);

interface Props {
  theme: 'scope' | 'print';
}

export function Landing({ theme }: Props) {
  // The clock does not start until the canvas has drawn a frame. Starting it
  // on mount meant the first beats played against a blank screen and the
  // sequence then jumped to catch up.
  const [ready, setReady] = useState(false);
  const intro = useIntro(ready);
  const reduced = useReducedMotion();
  const [envelope, setEnvelope] = useState<Float32Array | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [map, setMap] = useState<SpeakerMap | null>(null);

  // The galaxy sits below the fold. Mounting it on load would hold a third
  // WebGL context open for something off-screen, so it waits to be reached.
  // framer's useInView never fired here even with the panel on screen, so the
  // galaxy rendered its placeholder forever. onViewportEnter drives the same
  // animation on this element and is reliable.
  const [galaxyInView, setGalaxyInView] = useState(false);

  const onReady = useCallback(() => setReady(true), []);

  // The intro draws the real sample clip. If it cannot be decoded the sequence
  // falls back to a generated envelope rather than failing.
  useEffect(() => {
    void loadSampleEnvelope().then(setEnvelope);
  }, []);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
    api.map().then(setMap).catch((cause) => {
      if (!(cause instanceof ApiError)) return;
    });
  }, []);

  const titleShown = intro.beat !== 'start';
  const ctaShown = intro.beat === 'settle' || intro.beat === 'end';

  return (
    <div className="landing">
      <section className="landing__hero">
        {/* The surface stays, but its render loop stops the moment the
            sequence settles. Unmounting it would cost nothing to run and
            everything to look at; 'demand' draws the final frame once and then
            idles at zero frames. */}
        <Suspense fallback={null}>
          <IntroSequence
            elapsedRef={intro.elapsedRef}
            envelope={envelope}
            theme={theme}
            settled={intro.done}
            onReady={onReady}
          />
        </Suspense>

        <div className="landing__copy">
          <AnimatePresence>
            {titleShown ? (
              <motion.h1
                className="landing__title"
                initial={reduced ? false : { opacity: 0, y: 24 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
              >
                Linguistix
              </motion.h1>
            ) : null}
          </AnimatePresence>

          <AnimatePresence>
            {ctaShown ? (
              <motion.div
                className="landing__lede"
                initial={reduced ? false : { opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
              >
                <p>
                  A voice is a landscape. This one recognises fifty of them, and anyone else you
                  care to teach it — in about a second, with nothing retrained.
                </p>
                <div className="landing__actions">
                  <Link to="/analyse" className="btn btn--primary btn--lg" data-hero-cta>
                    Enter the instrument
                  </Link>
                  <Link to="/how-it-works" className="btn btn--ghost">
                    How it works
                  </Link>
                </div>
              </motion.div>
            ) : null}
          </AnimatePresence>
        </div>

        {/* Present and focusable from the first frame — the sequence must never
            be something a visitor has to sit through. */}
        {intro.playing ? (
          <button type="button" className="landing__skip" onClick={intro.skip}>
            Skip intro
          </button>
        ) : null}

        {ctaShown ? (
          <motion.div
            className="landing__scroll"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            aria-hidden="true"
          >
            <span className="eyebrow">Scroll</span>
            <span className="landing__scroll-line" />
          </motion.div>
        ) : null}
      </section>

      <section className="landing__section">
        <motion.header
          className="view__head"
          variants={stagger}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: '-80px' }}
        >
          <motion.span className="eyebrow" variants={riseIn}>What it does</motion.span>
          <motion.h2 variants={riseIn}>Three steps, about a second</motion.h2>
        </motion.header>
        <PipelineStrip />
      </section>

      <section className="landing__section">
        <motion.header
          className="view__head"
          variants={stagger}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: '-80px' }}
        >
          <motion.span className="eyebrow" variants={riseIn}>By the numbers</motion.span>
          <motion.h2 variants={riseIn}>Measured on voices it never saw</motion.h2>
        </motion.header>
        <StatTiles health={health} />
      </section>

      {map && !reduced ? (
        <section className="landing__section">
          <motion.header
            className="view__head"
            variants={stagger}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: '-80px' }}
          >
            <motion.span className="eyebrow" variants={riseIn}>The space</motion.span>
            <motion.h2 variants={riseIn}>Every voice it knows, mapped</motion.h2>
            <motion.p variants={riseIn}>
              {map.points.length} clips in the encoder&rsquo;s own 192 dimensions. Each cluster is one
              speaker; the gaps between them are what identification measures.
            </motion.p>
          </motion.header>

          <motion.div
            className="panel landing__galaxy"
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true, margin: '200px' }}
            onViewportEnter={() => setGalaxyInView(true)}
            transition={{ duration: 0.6 }}
          >
            {galaxyInView ? (
              <Suspense fallback={<div style={{ height: 360 }} />}>
                <EmbeddingGalaxy map={map} theme={theme} showHeldout height={360} />
              </Suspense>
            ) : (
              <div style={{ height: 360 }} />
            )}
          </motion.div>

          <div className="transport">
            <Link to="/map" className="btn btn--ghost btn--sm">
              Open the map
            </Link>
          </div>
        </section>
      ) : null}

      <section className="landing__section landing__closing">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          <h2>Say something.</h2>
          <p>
            Record a few seconds and watch it land among fifty known voices — or enrol yourself and
            become the fifty-first.
          </p>
          <div className="landing__actions">
            <Link to="/analyse" className="btn btn--primary btn--lg">
              Enter the instrument
            </Link>
            <Link to="/enrol" className="btn btn--ghost">
              Enrol a voice
            </Link>
          </div>
        </motion.div>
      </section>
    </div>
  );
}
