/**
 * A readout that counts to its value instead of snapping.
 *
 * Every number in this interface is a measurement, and watching one settle
 * reads the way an instrument settles. Uses tabular figures via the `.num`
 * class so the width never changes mid-count and the layout stays still.
 */

import { useEffect, useRef } from 'react';
import { animate, useReducedMotion } from 'framer-motion';

interface Props {
  value: number;
  decimals?: number;
  suffix?: string;
  prefix?: string;
  /** Seconds. Kept short — this is a readout, not a slot machine. */
  duration?: number;
  className?: string;
  style?: React.CSSProperties;
}

export function AnimatedNumber({
  value,
  decimals = 0,
  suffix = '',
  prefix = '',
  duration = 0.9,
  className,
  style,
}: Props) {
  const ref = useRef<HTMLSpanElement>(null);
  const reduced = useReducedMotion();

  const format = (n: number) => `${prefix}${n.toFixed(decimals)}${suffix}`;

  useEffect(() => {
    const node = ref.current;
    if (!node) return;

    // Deliberately not gated on viewport visibility. An earlier version waited
    // for the element to scroll into view and rendered zero until then, which
    // meant anyone who did not scroll saw a readout of 0 — a wrong number, not
    // just an unanimated one. Correctness beats saving a few frames of work.
    if (reduced) {
      node.textContent = format(value);
      return;
    }

    const controls = animate(0, value, {
      duration,
      ease: [0.16, 1, 0.3, 1],
      onUpdate: (v) => {
        node.textContent = format(v);
      },
      onComplete: () => {
        // Land exactly on the value rather than on the last eased sample.
        node.textContent = format(value);
      },
    });
    return () => {
      controls.stop();
      node.textContent = format(value);
    };
  }, [value, reduced, duration, decimals, prefix, suffix]);

  return (
    <span ref={ref} className={`num ${className ?? ''}`} style={style}>
      {/* Rendered server-side and pre-hydration as the final value, so the
          number is never missing for a screen reader or with JS disabled. */}
      {format(value)}
    </span>
  );
}
