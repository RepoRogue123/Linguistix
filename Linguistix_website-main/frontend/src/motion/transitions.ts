/**
 * Shared motion vocabulary.
 *
 * One file so timing stays consistent across views — a panel entering on the
 * Bench should feel identical to one entering on the Arena, and inconsistency
 * there is what makes an interface feel assembled rather than designed.
 *
 * Every duration here is short. This is an instrument: motion should confirm
 * that something changed, not perform.
 */

import type { Transition, Variants } from 'framer-motion';

/** The house easing. Fast out of the gate, settles without bouncing. */
export const EASE = [0.2, 0.7, 0.3, 1] as const;

export const spring: Transition = { type: 'spring', stiffness: 320, damping: 30, mass: 0.8 };

/** For the verdict only — overshoots enough to feel like an answer landing. */
export const springPunchy: Transition = { type: 'spring', stiffness: 420, damping: 18, mass: 0.9 };

export const fade: Variants = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { duration: 0.25, ease: EASE } },
};

/** Route-level: a short cross-fade with a slight rise. */
export const pageVariants: Variants = {
  hidden: { opacity: 0, y: 12 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.32, ease: EASE, when: 'beforeChildren', staggerChildren: 0.04 } },
  exit: { opacity: 0, y: -8, transition: { duration: 0.18, ease: EASE } },
};

/** Panels and cards entering as a group. */
export const stagger: Variants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.045, delayChildren: 0.02 } },
};

export const riseIn: Variants = {
  hidden: { opacity: 0, y: 16 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.4, ease: EASE } },
};

export const scaleIn: Variants = {
  hidden: { opacity: 0, scale: 0.96 },
  visible: { opacity: 1, scale: 1, transition: spring },
};

/** Candidate bars: width grows from zero, staggered down the list. */
export const barTrack: Variants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.06 } },
};

export const barFill = (width: number): Variants => ({
  hidden: { width: 0 },
  visible: { width: `${width}%`, transition: { duration: 0.62, ease: EASE } },
});

/**
 * Read the user's motion preference once, reactively.
 *
 * Returns true when animation should be suppressed. Every 3D surface and every
 * variant above is expected to consult this rather than assume.
 */
export function prefersReducedMotion(): boolean {
  if (typeof window === 'undefined') return false;
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}
