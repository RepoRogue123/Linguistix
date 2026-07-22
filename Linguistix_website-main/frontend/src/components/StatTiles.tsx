/**
 * The numbers that describe the system, counted up on arrival.
 *
 * Present before any interaction, so the landing view says what this thing is
 * and how well it does it without waiting for the visitor to record something.
 */

import { motion } from 'framer-motion';

import { AnimatedNumber } from '../motion/AnimatedNumber';
import { riseIn, stagger } from '../motion/transitions';
import type { Health } from '../lib/api';

interface Props {
  health: Health | null;
}

export function StatTiles({ health }: Props) {
  const encoder = health?.components?.encoder ?? {};
  const gallery = health?.components?.gallery ?? {};

  const tiles = [
    {
      value: (gallery.reference_speakers ?? 50) + (gallery.enrolled_speakers ?? 0),
      decimals: 0,
      label: 'voices known',
      hint: gallery.enrolled_speakers ? `${gallery.enrolled_speakers} enrolled by you` : 'from the dataset',
    },
    {
      value: encoder.embedding_dim ?? 192,
      decimals: 0,
      label: 'dimensions',
      hint: 'per voiceprint',
    },
    {
      value: typeof encoder.heldout_eer === 'number' ? encoder.heldout_eer : 10.15,
      decimals: 2,
      suffix: '%',
      label: 'error rate',
      hint: 'on voices never trained on',
    },
    {
      value: 0,
      decimals: 0,
      label: 'retraining',
      hint: 'needed to add a person',
    },
  ];

  return (
    <motion.div className="tiles" variants={stagger} initial="hidden" animate="visible">
      {tiles.map((tile) => (
        <motion.div className="tile" key={tile.label} variants={riseIn}>
          <AnimatedNumber
            className="tile__value"
            value={tile.value}
            decimals={tile.decimals}
            suffix={tile.suffix}
          />
          <span className="tile__label">{tile.label}</span>
          <span className="tile__hint">{tile.hint}</span>
        </motion.div>
      ))}
    </motion.div>
  );
}
