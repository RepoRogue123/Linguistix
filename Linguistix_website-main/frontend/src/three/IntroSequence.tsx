/**
 * The front door: a voice becoming a landscape, in about four seconds.
 *
 *   0.0s  a flat line of light — a scope with no signal
 *   0.4s  the line becomes a waveform, rippling outward from the centre
 *   1.4s  the waveform folds upward into terrain as the camera pulls back
 *   2.6s  the title resolves
 *   3.6s  the terrain settles into its idle drift
 *
 * The waveform is the real bundled sample clip, decoded once — the same audio
 * the Try-a-sample button uses. That is the whole reason this is worth building
 * rather than dropping in a generic hero: the thing unfolding on screen is an
 * actual voice, which is what the product is about.
 *
 * The geometry is one plane whose displacement is blended between a flat line,
 * the waveform, and full terrain by sequence progress, so all five beats are a
 * single continuous surface rather than a cut between scenes.
 */

import { useEffect, useMemo, useRef } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { useReducedMotion } from 'framer-motion';
import * as THREE from 'three';

import { NOISE_GLSL, RAMP_GLSL, rampUniforms } from './shaders/ramp';
import { INTRO_BEATS } from '../lib/useIntro';

interface Props {
  /** Live seconds into the sequence, read per frame without re-rendering React. */
  elapsedRef: { current: number };
  envelope: Float32Array | null;
  theme: 'scope' | 'print';
  /** True once the sequence has finished; the loop stops and the image holds. */
  settled?: boolean;
  /** Fired after the first frame is on screen, so the clock can start then. */
  onReady?: () => void;
}

const COLS = 112;
const ROWS = 64;

const vertexShader = /* glsl */ `
uniform float uTime;
uniform float uLine;       // 1 = flat line only
uniform float uWave;       // 0..1 waveform strength
uniform float uTerrain;    // 0..1 terrain strength
uniform sampler2D uEnvelope;
varying vec2 vUv;
varying float vEnergy;
varying float vBand;

${NOISE_GLSL}

void main() {
  vUv = uv;

  // How close this vertex is to the horizontal centreline. The opening beat is
  // only this band; everything else grows out of it.
  float band = 1.0 - smoothstep(0.0, 0.06, abs(uv.y - 0.5));
  vBand = band;

  // Real amplitude from the decoded clip, mirrored about the centre so the
  // waveform opens outward rather than scrolling past.
  float amp = texture2D(uEnvelope, vec2(uv.x, 0.5)).r;
  float wave = amp * (1.0 - smoothstep(0.0, 0.28, abs(uv.y - 0.5))) * uWave;

  // Terrain: the same ridged fbm the main voiceprint uses, so the landing and
  // the instrument are visibly the same surface.
  float slow = fbm(vec3(uv * 2.4, uTime * 0.07));
  float fast = fbm(vec3(uv * 6.5 + 11.0, uTime * 0.15));
  float relief = (1.0 - abs(slow)) * 0.62 + fast * 0.3;
  relief *= mix(1.0, 0.35, uv.y);
  relief = clamp(relief * 1.15 - 0.12, 0.0, 1.0) * uTerrain;

  // Relief has to win once the fold completes, or the bright waveform band
  // keeps dominating and the surface never reads as a landscape.
  float energy = max(band * uLine * 0.06, max(wave * 0.85, relief * 1.15));
  vEnergy = energy;

  vec3 displaced = position;
  displaced.z += energy * 2.3;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(displaced, 1.0);
}
`;

const fragmentShader = /* glsl */ `
uniform float uMode;
uniform float uLine;
uniform float uSolid;   // 0 during the opening beats, 1 once terrain has formed
varying vec2 vUv;
varying float vEnergy;
varying float vBand;

${RAMP_GLSL}

void main() {
  float slope = clamp(length(vec2(dFdx(vEnergy), dFdy(vEnergy))) * 34.0, 0.0, 1.0);

  vec3 color = rampColor(vEnergy, uMode);
  color *= mix(0.72, 1.35, slope);

  // During the opening beats the surface is transparent except where it is
  // carrying signal, so the first thing on screen is a single filament of
  // light. Once the fold completes it becomes an opaque landscape — without
  // uSolid the discard below eats the whole terrain and leaves only the
  // brightest band, which reads as a stray streak rather than a surface.
  float reveal = max(vEnergy * 3.2, vBand * uLine);
  float alpha = clamp(mix(reveal, 1.0, uSolid), 0.0, 1.0);

  // Dissolve at the plane's edges. Without this the surface ends in a hard
  // rectangular cut against the ground, which reads as a pasted-in slab rather
  // than a landscape continuing past the frame.
  vec2 edge = min(vUv, 1.0 - vUv);
  float fade = smoothstep(0.0, 0.14, edge.x) * smoothstep(0.0, 0.10, edge.y);
  alpha *= fade;

  // The line beat is deliberately near-white: it reads as a signal trace before
  // it has become anything a colour ramp could describe.
  color = mix(color, vec3(1.0 - uMode * 0.85), vBand * uLine * 0.8);

  if (alpha < 0.01) discard;
  gl_FragColor = vec4(toSRGB(color), alpha);
}
`;

/** Smooth 0..1 ramp between two times. */
function phase(t: number, from: number, to: number): number {
  const raw = Math.max(0, Math.min(1, (t - from) / Math.max(0.0001, to - from)));
  return raw * raw * (3 - 2 * raw);
}

function Surface({ elapsedRef, envelope, theme, settled, onReady }: Props) {
  const meshRef = useRef<THREE.Mesh>(null);
  const materialRef = useRef<THREE.ShaderMaterial>(null);
  const reduced = useReducedMotion();
  const { invalidate } = useThree();

  // The envelope becomes a 1-pixel-tall texture the vertex shader samples.
  const texture = useMemo(() => {
    const width = envelope?.length ?? 256;
    const data = new Float32Array(width);
    if (envelope) {
      data.set(envelope);
    } else {
      // Fallback if the clip could not be decoded: a plausible speech-like
      // envelope so the sequence still plays rather than showing a flat bar.
      for (let i = 0; i < width; i++) {
        const t = i / width;
        data[i] = Math.abs(Math.sin(t * 22) * Math.sin(t * 5.5)) * (0.35 + 0.65 * Math.sin(t * Math.PI));
      }
    }
    const tex = new THREE.DataTexture(data, width, 1, THREE.RedFormat, THREE.FloatType);
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.wrapS = THREE.ClampToEdgeWrapping;
    tex.needsUpdate = true;
    return tex;
  }, [envelope]);

  useEffect(() => () => texture.dispose(), [texture]);

  // One final render once the loop switches to on-demand, so the settled image
  // is actually painted rather than left on the last animated frame.
  useEffect(() => {
    if (settled) invalidate();
  }, [settled, invalidate]);

  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uLine: { value: 1 },
      uWave: { value: 0 },
      uTerrain: { value: 0 },
      uSolid: { value: 0 },
      uEnvelope: { value: texture },
      uMode: { value: theme === 'print' ? 1 : 0 },
      ...rampUniforms(THREE),
    }),
    [texture, theme],
  );

  const announced = useRef(false);

  useFrame((state, delta) => {
    const material = materialRef.current;
    const mesh = meshRef.current;
    if (!material || !mesh) return;

    // The shader has compiled and this frame is about to be presented.
    if (!announced.current) {
      announced.current = true;
      onReady?.();
    }

    // Under reduced motion the surface is a still image: the settled terrain is
    // content worth showing, but nothing about it should drift.
    if (!reduced) material.uniforms.uTime.value += delta;
    material.uniforms.uMode.value = theme === 'print' ? 1 : 0;

    const t = elapsedRef.current;
    const wave = phase(t, INTRO_BEATS.waveform, INTRO_BEATS.fold);
    const fold = phase(t, INTRO_BEATS.fold, INTRO_BEATS.title);

    // The line fades exactly as the waveform arrives, so there is never a frame
    // where both are competing for the same pixels.
    material.uniforms.uLine.value = 1 - wave;
    // The waveform fades out almost entirely as the fold completes; leaving a
    // quarter of it behind kept a hard bright ridge across the landscape.
    material.uniforms.uWave.value = wave * (1 - fold * 0.96);
    material.uniforms.uTerrain.value = fold;
    material.uniforms.uSolid.value = fold;

    // Camera and tilt are driven by the same progress, so the pull-back and the
    // fold happen as one movement rather than two animations that agree.
    const settle = phase(t, INTRO_BEATS.title, INTRO_BEATS.end);
    state.camera.position.z = 5.2 + fold * 1.4 + settle * 0.2;
    state.camera.position.y = fold * 0.9 + settle * 0.25;
    state.camera.lookAt(0, 0, 0);

    mesh.rotation.x = -fold * 1.02;
    if (settle > 0.99 && !reduced) mesh.rotation.z += delta * 0.045;
  });

  return (
    <mesh ref={meshRef} rotation={[0, 0, 0]}>
      <planeGeometry args={[15.5, 8.4, COLS - 1, ROWS - 1]} />
      <shaderMaterial
        ref={materialRef}
        uniforms={uniforms}
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        side={THREE.FrontSide}
        transparent
      />
    </mesh>
  );
}

export function IntroSequence(props: Props) {
  const reduced = useReducedMotion();
  const idle = reduced || props.settled;

  return (
    <Canvas
      dpr={[1, 1.5]}
      camera={{ position: [0, 0, 5.2], fov: 46 }}
      gl={{ antialias: false, alpha: true, powerPreference: 'high-performance' }}
      style={{ position: 'absolute', inset: 0 }}
      // Under reduced motion this draws one still frame and stops. While the
      // sequence plays it must run every frame; Landing unmounts the whole
      // canvas the moment it finishes.
      frameloop={idle ? 'demand' : 'always'}
    >
      <Surface {...props} />
    </Canvas>
  );
}
