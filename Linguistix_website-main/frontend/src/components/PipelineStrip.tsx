/**
 * What happens to a voice, in three steps.
 *
 * Sits on the landing view so the page explains itself before anyone presses
 * anything. Each step draws its own small animated diagram rather than using an
 * icon, because the shapes are the actual objects involved — a waveform, a
 * spectrogram, a point among clusters.
 */

import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';

import { riseIn, stagger } from '../motion/transitions';

const STEPS = [
  {
    n: '01',
    title: 'Capture',
    body: 'A few seconds of speech at 16 kHz. Anything shorter than a second gets refused rather than guessed at.',
  },
  {
    n: '02',
    title: 'Embed',
    body: 'Five overlapping windows through a 2-million-parameter network, averaged into one 192-dimension point.',
  },
  {
    n: '03',
    title: 'Match',
    body: 'Nearest neighbour against every known voice. Below the calibrated threshold it declines to answer.',
  },
];

/** A waveform that draws itself, then a spectrogram, then a settling point. */
function Glyph({ index }: { index: number }) {
  if (index === 0) {
    const d = Array.from({ length: 48 }, (_, i) => {
      const x = (i / 47) * 100;
      const amp = Math.sin(i * 0.55) * Math.sin(i * 0.13) * 16;
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${(20 + amp).toFixed(1)}`;
    }).join(' ');
    return (
      <svg viewBox="0 0 100 40" className="glyph" aria-hidden="true">
        <motion.path
          d={d}
          fill="none"
          stroke="currentColor"
          strokeWidth="1.4"
          initial={{ pathLength: 0 }}
          whileInView={{ pathLength: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 1.1, ease: 'easeInOut' }}
        />
      </svg>
    );
  }

  if (index === 1) {
    return (
      <svg viewBox="0 0 100 40" className="glyph" aria-hidden="true">
        {Array.from({ length: 26 }).map((_, i) =>
          Array.from({ length: 7 }).map((__, j) => (
            <motion.rect
              key={`${i}-${j}`}
              x={i * 3.9}
              y={j * 5.5 + 1}
              width={3}
              height={4.6}
              fill="currentColor"
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 0.12 + Math.abs(Math.sin(i * 0.6 + j)) * 0.75 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: i * 0.016 }}
            />
          )),
        )}
      </svg>
    );
  }

  return (
    <svg viewBox="0 0 100 40" className="glyph" aria-hidden="true">
      {Array.from({ length: 34 }).map((_, i) => {
        const cx = 12 + (i % 6) * 4 + Math.floor(i / 6) * 13;
        const cy = 8 + ((i * 7) % 24);
        return <circle key={i} cx={cx} cy={cy} r={1.5} fill="currentColor" opacity={0.32} />;
      })}
      <motion.circle
        r={3}
        fill="currentColor"
        initial={{ cx: 92, cy: 4, opacity: 0 }}
        whileInView={{ cx: 30, cy: 18, opacity: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1], delay: 0.2 }}
      />
    </svg>
  );
}

export function PipelineStrip() {
  return (
    <motion.section
      className="pipeline"
      variants={stagger}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, margin: '-60px' }}
    >
      {STEPS.map((step, i) => (
        <motion.article className="pipeline__step" key={step.n} variants={riseIn}>
          <span className="pipeline__n eyebrow">{step.n}</span>
          <Glyph index={i} />
          <h3 className="pipeline__title">{step.title}</h3>
          <p className="pipeline__body">{step.body}</p>
        </motion.article>
      ))}

      <motion.div className="pipeline__more" variants={riseIn}>
        <Link to="/how-it-works" className="btn btn--ghost btn--sm">
          The longer version
        </Link>
      </motion.div>
    </motion.section>
  );
}
