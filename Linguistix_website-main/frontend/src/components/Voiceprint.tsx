/**
 * Picks the right voiceprint renderer for the situation.
 *
 * 3D terrain by default; the existing 2D strip when the user has asked for
 * reduced motion or when WebGL is unavailable. Both draw the same data from the
 * same ramp, so the fallback is a genuine alternative rather than a degraded
 * placeholder — nothing is missing, it simply does not move in space.
 */

import { Suspense, lazy, useEffect, useState } from 'react';
import { useReducedMotion } from 'framer-motion';

import { SonagramStrip, type StripState } from './SonagramStrip';

const VoiceprintTerrain = lazy(() =>
  import('../three/VoiceprintTerrain').then((m) => ({ default: m.VoiceprintTerrain })),
);

interface Props {
  state: StripState;
  analyser?: AnalyserNode | null;
  spectrogram?: number[][] | null;
  saliency?: number[][] | null;
  theme: 'scope' | 'print';
  height?: number;
  label?: string;
  /** Force the flat renderer, for the smaller slots on Enrol and Lab. */
  flat?: boolean;
}

/** One-time WebGL probe. A failure here must not take the page down. */
function detectWebGL(): boolean {
  try {
    const canvas = document.createElement('canvas');
    return Boolean(
      window.WebGLRenderingContext &&
        (canvas.getContext('webgl2') || canvas.getContext('webgl')),
    );
  } catch {
    return false;
  }
}

export function Voiceprint({ flat, height = 340, ...props }: Props) {
  const reduced = useReducedMotion();
  const [webgl, setWebgl] = useState<boolean | null>(null);

  useEffect(() => setWebgl(detectWebGL()), []);

  const use3D = !flat && !reduced && webgl === true;

  if (!use3D) {
    return <SonagramStrip {...props} height={Math.min(height, 220)} />;
  }

  return (
    <Suspense fallback={<SonagramStrip {...props} height={Math.min(height, 220)} />}>
      <VoiceprintTerrain {...props} height={height} />
    </Suspense>
  );
}
