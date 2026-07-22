/**
 * The intro clock, and every way out of it.
 *
 * A cinematic entry is only tolerable if it is never a gate. Someone arriving
 * for the tenth time should not have to sit through it, and someone arriving
 * for the first time should not feel trapped. So the sequence can be ended by
 * the Skip control, a click anywhere, a scroll, Escape, or any key — and under
 * `prefers-reduced-motion` it never starts at all.
 *
 * Skipping jumps to the settled state rather than to a half-played frame, so
 * every exit lands in the same place.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

/** Beat boundaries in seconds; see IntroSequence for what each one does. */
export const INTRO_BEATS = {
  line: 0.0,
  waveform: 0.4,
  fold: 1.4,
  title: 2.6,
  settle: 3.6,
  end: 4.5,
} as const;

export interface Intro {
  /**
   * Live seconds elapsed, as a ref.
   *
   * Deliberately not React state. Setting state every animation frame
   * re-reconciled the entire landing tree 60 times a second for 4.5 seconds —
   * concurrently with the heaviest GPU load in the app. The canvas reads this
   * ref inside its own render loop instead, and React only re-renders at the
   * three boundaries that actually gate UI.
   */
  elapsedRef: { current: number };
  /** The last crossed beat, and the only thing that triggers a re-render. */
  beat: 'start' | 'title' | 'settle' | 'end';
  /** True once the sequence has finished or been skipped. */
  done: boolean;
  /** True while it is actually playing — false when skipped or reduced. */
  playing: boolean;
  skip: () => void;
}

export function useIntro(enabled = true): Intro {
  // Read the preference once, synchronously, so the first render is already
  // correct. Deciding this in an effect would flash one frame of animation at
  // exactly the people who asked not to see any.
  const [reduced] = useState(
    () =>
      typeof window !== 'undefined' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches,
  );

  const shouldPlay = enabled && !reduced;
  const elapsedRef = useRef(shouldPlay ? 0 : INTRO_BEATS.end);
  const [beat, setBeat] = useState<Intro['beat']>(shouldPlay ? 'start' : 'end');
  const [done, setDone] = useState(!shouldPlay);
  const frame = useRef(0);
  const startedAt = useRef(0);

  const skip = useCallback(() => {
    cancelAnimationFrame(frame.current);
    elapsedRef.current = INTRO_BEATS.end;
    setBeat('end');
    setDone(true);
  }, []);

  useEffect(() => {
    if (!shouldPlay || done) return;

    startedAt.current = performance.now();
    const tick = (now: number) => {
      const t = (now - startedAt.current) / 1000;
      elapsedRef.current = Math.min(t, INTRO_BEATS.end);

      if (t >= INTRO_BEATS.end) {
        setBeat('end');
        setDone(true);
        return;
      }

      // One setState per crossed boundary — three for the whole sequence,
      // rather than one per frame.
      if (t >= INTRO_BEATS.settle) setBeat((b) => (b === 'settle' || b === 'end' ? b : 'settle'));
      else if (t >= INTRO_BEATS.title) setBeat((b) => (b === 'start' ? 'title' : b));

      frame.current = requestAnimationFrame(tick);
    };

    frame.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame.current);
  }, [shouldPlay, done]);

  // Every escape hatch, bound while the sequence is running and removed the
  // moment it is not.
  useEffect(() => {
    if (done) return;

    const onKey = (event: KeyboardEvent) => {
      // Let Tab through: someone reaching for the Skip button by keyboard
      // should be able to focus it rather than having the sequence vanish
      // out from under them mid-navigation.
      if (event.key === 'Tab') return;
      skip();
    };

    window.addEventListener('pointerdown', skip);
    window.addEventListener('wheel', skip, { passive: true });
    window.addEventListener('touchstart', skip, { passive: true });
    window.addEventListener('keydown', onKey);

    return () => {
      window.removeEventListener('pointerdown', skip);
      window.removeEventListener('wheel', skip);
      window.removeEventListener('touchstart', skip);
      window.removeEventListener('keydown', onKey);
    };
  }, [done, skip]);

  return {
    elapsedRef,
    beat,
    done,
    playing: shouldPlay && !done,
    skip,
  };
}

/**
 * Decode the bundled sample clip into a coarse amplitude envelope.
 *
 * The intro's waveform beat draws a real voice — the same clip the Try-a-sample
 * button uses — rather than a synthesised sine. Returns null on any failure so
 * the sequence can fall back to a generated shape; a missing asset should soften
 * the intro, never break the landing page.
 */
export async function loadSampleEnvelope(buckets = 256): Promise<Float32Array | null> {
  try {
    const response = await fetch('/sample.wav');
    if (!response.ok) return null;

    const context = new AudioContext();
    try {
      const decoded = await context.decodeAudioData(await response.arrayBuffer());
      const channel = decoded.getChannelData(0);
      const envelope = new Float32Array(buckets);
      const per = Math.floor(channel.length / buckets);

      for (let i = 0; i < buckets; i++) {
        let peak = 0;
        for (let j = 0; j < per; j++) {
          const v = Math.abs(channel[i * per + j]);
          if (v > peak) peak = v;
        }
        envelope[i] = peak;
      }

      // Normalize so quiet recordings still fill the frame.
      const max = envelope.reduce((m, v) => Math.max(m, v), 0) || 1;
      for (let i = 0; i < buckets; i++) envelope[i] /= max;
      return envelope;
    } finally {
      void context.close();
    }
  } catch {
    return null;
  }
}
