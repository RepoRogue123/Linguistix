/**
 * A match score you can read before you read the number.
 *
 * A displaced sphere whose surface calms as confidence rises: a clear match
 * settles into something smooth and bright, a borderline one churns and dims.
 * The agitation is the useful part — it distinguishes "0.92, comfortable" from
 * "0.41, barely over the line" faster than two similar-looking percentages do.
 */

import { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';

import { NOISE_GLSL, RAMP_GLSL, rampUniforms } from './shaders/ramp';

interface Props {
  /** Cosine similarity, -1..1. */
  score: number;
  /** Acceptance threshold, so the orb knows how much headroom there is. */
  threshold: number;
  accepted: boolean;
  theme: 'scope' | 'print';
  size?: number;
}

const vertexShader = /* glsl */ `
uniform float uTime;
uniform float uTurbulence;
varying float vDisplace;
varying vec3 vNormalW;

${NOISE_GLSL}

void main() {
  float n = fbm(position * 1.9 + vec3(0.0, 0.0, uTime * 0.35));
  float displace = n * uTurbulence;
  vDisplace = n;
  vNormalW = normalize(normalMatrix * normal);

  vec3 displaced = position + normal * displace;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(displaced, 1.0);
}
`;

const fragmentShader = /* glsl */ `
uniform float uScore;
uniform float uMode;
varying float vDisplace;
varying vec3 vNormalW;

${RAMP_GLSL}

void main() {
  // Rim light so the silhouette stays legible against either background.
  float rim = pow(1.0 - abs(vNormalW.z), 2.0);
  float t = clamp(uScore * 0.75 + vDisplace * 0.35 + rim * 0.25, 0.0, 1.0);
  vec3 color = rampColor(t, uMode);
  gl_FragColor = vec4(toSRGB(color), 1.0);
}
`;

function Orb({ score, threshold, accepted, theme }: Props) {
  const materialRef = useRef<THREE.ShaderMaterial>(null);
  const meshRef = useRef<THREE.Mesh>(null);

  // Normalise the score against its own operating point rather than against
  // the raw -1..1 range: the interesting region is the neighbourhood of the
  // threshold, and a linear map would compress all of it into a sliver.
  const headroom = useMemo(() => {
    const span = Math.max(0.05, 1 - threshold);
    return Math.max(0, Math.min(1, (score - threshold) / span));
  }, [score, threshold]);

  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uTurbulence: { value: 0.3 },
      uScore: { value: 0.5 },
      uMode: { value: theme === 'print' ? 1 : 0 },
      ...rampUniforms(THREE),
    }),
    [theme],
  );

  useFrame((_, delta) => {
    const material = materialRef.current;
    if (!material) return;

    material.uniforms.uTime.value += delta;
    material.uniforms.uMode.value = theme === 'print' ? 1 : 0;

    // Confident matches are calm; rejections churn.
    const target = accepted ? 0.04 + (1 - headroom) * 0.22 : 0.34;
    material.uniforms.uTurbulence.value +=
      (target - material.uniforms.uTurbulence.value) * Math.min(1, delta * 3);

    const scoreTarget = accepted ? 0.45 + headroom * 0.55 : 0.22;
    material.uniforms.uScore.value +=
      (scoreTarget - material.uniforms.uScore.value) * Math.min(1, delta * 3);

    if (meshRef.current) meshRef.current.rotation.y += delta * 0.25;
  });

  return (
    <mesh ref={meshRef}>
      <icosahedronGeometry args={[1.15, 5]} />
      <shaderMaterial
        ref={materialRef}
        uniforms={uniforms}
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
      />
    </mesh>
  );
}

export function ConfidenceOrb({ size = 150, ...props }: Props) {
  return (
    <div style={{ width: size, height: size }} aria-hidden="true">
      <Canvas dpr={[1, 1.5]} camera={{ position: [0, 0, 3.4], fov: 45 }} gl={{ alpha: true, antialias: true }}>
        <Orb {...props} />
      </Canvas>
    </div>
  );
}
