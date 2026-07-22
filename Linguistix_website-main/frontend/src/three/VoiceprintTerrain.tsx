/**
 * The voiceprint terrain — the piece this interface is built around.
 *
 * A spectrogram is already three quantities: frequency, time, and energy. The
 * 2D strip flattens the third into colour; this renders it as height, so a
 * voice becomes a landscape you can look across. Ridges are formants, valleys
 * are the pauses between words.
 *
 * Four states, one object, so the user watches a single thing transform rather
 * than panels appearing and disappearing:
 *
 *   idle       a slow ambient swell, gently rotating
 *   recording  terrain scrolls toward the viewer, built live from the analyser
 *   analysing  rotates flat to face front while a light sweeps across it once
 *   resolved   frozen as the analysed clip, orbitable, saliency lifting the
 *              ridges that carried the identification
 *
 * Displacement happens in the vertex shader from a data texture, so updating a
 * frame costs one texture row rather than rebuilding geometry.
 */

import { useEffect, useMemo, useRef } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

import { NOISE_GLSL, RAMP_GLSL, rampUniforms } from './shaders/ramp';

export type TerrainState = 'idle' | 'recording' | 'analysing' | 'resolved';

interface Props {
  state: TerrainState;
  analyser?: AnalyserNode | null;
  /** Server spectrogram [n_mels][frames], 0-255. */
  spectrogram?: number[][] | null;
  /** Optional saliency, same shape, 0-255. */
  saliency?: number[][] | null;
  theme: 'scope' | 'print';
  height?: number;
  className?: string;
}

// Grid resolution. 128 x 96 is ~12k vertices — smooth on integrated graphics
// and still fine enough to resolve individual formant ridges.
const COLS = 128;
const ROWS = 96;

const vertexShader = /* glsl */ `
uniform sampler2D uData;
uniform float uTime;
uniform float uHeight;
uniform float uIdle;      // 1 while idle, 0 once real audio drives it
varying vec2 vUv;
varying float vEnergy;

${NOISE_GLSL}

void main() {
  vUv = uv;

  // Soft-knee compression: keeps quiet detail readable while stopping loud
  // frames spiking past the top of the frame.
  float raw = texture2D(uData, uv).r;
  float sampled = raw < 0.55 ? raw : 0.55 + (raw - 0.55) * 0.52;

  // Idle is a generated swell rather than silence: the instrument should look
  // powered on, not broken, before anyone speaks into it. Two octaves at
  // different rates keep it from looking like a repeating pattern, and the
  // ridged term gives it formant-like crests instead of soft blobs.
  float slow = fbm(vec3(uv * 2.4, uTime * 0.07));
  float fast = fbm(vec3(uv * 6.5 + 11.0, uTime * 0.15));
  float ridged = 1.0 - abs(slow);
  float ambient = ridged * 0.62 + fast * 0.3;

  // Energy falls off toward high frequencies, the way real speech does.
  ambient *= mix(1.0, 0.35, uv.y);
  ambient = clamp(ambient * 1.15 - 0.12, 0.0, 1.0);

  float energy = mix(sampled, ambient, uIdle);
  // Colour reads from the uncompressed value so the ramp still uses its full
  // range; only the geometry is compressed.
  vEnergy = mix(raw, ambient, uIdle);

  vec3 displaced = position;
  displaced.z += energy * uHeight;

  gl_Position = projectionMatrix * modelViewMatrix * vec4(displaced, 1.0);
}
`;

const fragmentShader = /* glsl */ `
uniform float uMode;      // 0 scope, 1 print
uniform float uSweep;     // -1 disabled, else 0..1 scan position
uniform float uOpacity;
varying vec2 vUv;
varying float vEnergy;

${RAMP_GLSL}

void main() {
  // Cheap slope shading from screen-space derivatives: ridges catch light and
  // valleys fall away, which is what makes this read as terrain rather than a
  // coloured sheet. No lights needed, so the material stays unlit and fast.
  float slope = clamp(length(vec2(dFdx(vEnergy), dFdy(vEnergy))) * 34.0, 0.0, 1.0);

  vec3 color = rampColor(vEnergy, uMode);
  color *= mix(0.72, 1.35, slope);
  color = mix(color, rampColor(min(1.0, vEnergy + 0.22), uMode), slope * 0.55);

  // Hairline grid, so the surface reads as a measurement rather than terrain
  // art. Spacing matches the axis ticks on the 2D strip.
  vec2 grid = abs(fract(vUv * vec2(16.0, 12.0) - 0.5) - 0.5) / fwidth(vUv * vec2(16.0, 12.0));
  float line = 1.0 - min(min(grid.x, grid.y), 1.0);
  color = mix(color, mix(color, vec3(1.0 - uMode), 0.35), line * 0.16);

  // The analysing sweep: one bright band crossing once. The glow is added here,
  // in linear space, because it is light being emitted.
  float ahead = 0.0;
  if (uSweep >= 0.0) {
    float d = abs(vUv.x - uSweep);
    float glow = smoothstep(0.045, 0.0, d);
    color += glow * (1.0 - uMode) * 0.85;
    color = mix(color, vec3(0.0), glow * uMode * 0.55);
    ahead = step(uSweep, vUv.x);
  }

  vec3 lit = toSRGB(color);

  // Dim the un-measured region AFTER gamma encoding. Doing it in linear space
  // and then encoding turned an intended 18% into roughly 46%, which is why
  // the unscanned area read as washed-out pale rather than as dark.
  lit = mix(lit, lit * 0.2 + vec3(0.02, 0.03, 0.05) * (1.0 - uMode), ahead);

  gl_FragColor = vec4(lit, uOpacity);
}
`;

function Terrain({ state, analyser, spectrogram, saliency, theme }: Props) {
  const meshRef = useRef<THREE.Mesh>(null);
  const materialRef = useRef<THREE.ShaderMaterial>(null);
  const sweepRef = useRef(0);
  const writeRef = useRef(0);
  // Reused across frames; allocating this per frame while recording churned
  // 512 bytes of garbage every 16 ms.
  const frameRef = useRef<Uint8Array<ArrayBuffer> | null>(null);
  const { invalidate } = useThree();

  // One R-channel float texture holding the whole grid. Recording writes a
  // single column per frame; resolved uploads the clip in one go.
  const { texture, data } = useMemo(() => {
    const buffer = new Float32Array(COLS * ROWS);
    const tex = new THREE.DataTexture(buffer, COLS, ROWS, THREE.RedFormat, THREE.FloatType);
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.wrapS = THREE.ClampToEdgeWrapping;
    tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.needsUpdate = true;
    return { texture: tex, data: buffer };
  }, []);

  useEffect(() => () => texture.dispose(), [texture]);

  const uniforms = useMemo(
    () => ({
      uData: { value: texture },
      uTime: { value: 0 },
      uHeight: { value: 2.35 },
      uIdle: { value: 1 },
      uMode: { value: theme === 'print' ? 1 : 0 },
      uSweep: { value: -1 },
      uOpacity: { value: 1 },
      ...rampUniforms(THREE),
    }),
    [texture, theme],
  );

  useEffect(() => {
    if (materialRef.current) materialRef.current.uniforms.uMode.value = theme === 'print' ? 1 : 0;
  }, [theme]);

  // Reset when a new take starts so the previous one does not scroll through.
  useEffect(() => {
    if (state === 'recording') {
      data.fill(0);
      writeRef.current = 0;
      texture.needsUpdate = true;
    }
    if (state === 'analysing') sweepRef.current = 0;
  }, [state, data, texture]);

  /** Upload a server spectrogram, resampled onto the grid. */
  useEffect(() => {
    if (!spectrogram || (state !== 'resolved' && state !== 'analysing')) return;

    const bins = spectrogram.length;
    const frames = spectrogram[0]?.length ?? 0;
    if (!bins || !frames) return;

    for (let row = 0; row < ROWS; row++) {
      const b = Math.min(bins - 1, Math.floor((row / ROWS) * bins));
      for (let col = 0; col < COLS; col++) {
        const f = Math.min(frames - 1, Math.floor((col / COLS) * frames));
        let v = (spectrogram[b][f] ?? 0) / 255;
        if (saliency) {
          // Saliency lifts rather than recolours: the regions that carried the
          // identification physically rise out of the surface.
          const w = (saliency[b]?.[f] ?? 0) / 255;
          v = Math.min(1, v * (0.55 + w * 1.15));
        }
        data[row * COLS + col] = v;
      }
    }
    texture.needsUpdate = true;
    invalidate();
  }, [spectrogram, saliency, state, data, texture, invalidate]);

  useFrame((_, delta) => {
    const material = materialRef.current;
    const mesh = meshRef.current;
    if (!material || !mesh) return;

    material.uniforms.uTime.value += delta;

    // Idle noise is gentle and wants amplitude; real audio is spiky and does not.
    const heightTarget = state === 'idle' ? 2.35 : 1.45;
    material.uniforms.uHeight.value +=
      (heightTarget - material.uniforms.uHeight.value) * Math.min(1, delta * 2.5);

    if (state === 'recording' && analyser) {
      const bins = analyser.frequencyBinCount;
      if (!frameRef.current || frameRef.current.length !== bins) {
        frameRef.current = new Uint8Array(new ArrayBuffer(bins));
      }
      const frame = frameRef.current;
      analyser.getByteFrequencyData(frame);

      // Scroll left one column, then write the newest frame at the right edge.
      for (let row = 0; row < ROWS; row++) {
        const base = row * COLS;
        data.copyWithin(base, base + 1, base + COLS);
        // Only the lower ~62% of FFT bins carry speech at 16 kHz.
        const b = Math.min(bins - 1, Math.floor((row / ROWS) * bins * 0.62));
        data[base + COLS - 1] = frame[b] / 255;
      }
      texture.needsUpdate = true;
      material.uniforms.uIdle.value = 0;
      material.uniforms.uSweep.value = -1;
    } else if (state === 'idle') {
      material.uniforms.uIdle.value = 1;
      material.uniforms.uSweep.value = -1;
    } else {
      material.uniforms.uIdle.value = 0;
    }

    if (state === 'analysing') {
      // Paced to the 1.6s scan window in Bench, so the light finishes crossing
      // exactly as the verdict is revealed rather than 900ms early.
      sweepRef.current = Math.min(1, sweepRef.current + delta / 1.6);
      material.uniforms.uSweep.value = sweepRef.current;
    } else if (state === 'resolved') {
      material.uniforms.uSweep.value = -1;
    }

    // Idle drifts; analysing rotates flat to face the viewer; resolved holds
    // still so the user can orbit it themselves.
    const targetX = state === 'idle' ? -1.02 : state === 'recording' ? -0.92 : -0.78;
    mesh.rotation.x += (targetX - mesh.rotation.x) * Math.min(1, delta * 2.4);

    if (state === 'idle') {
      mesh.rotation.z += delta * 0.045;
    } else {
      mesh.rotation.z += (0 - mesh.rotation.z) * Math.min(1, delta * 2.4);
    }
  });

  return (
    <mesh ref={meshRef} rotation={[-1.02, 0, 0]}>
      <planeGeometry args={[11.2, 6.2, COLS - 1, ROWS - 1]} />
      <shaderMaterial
        ref={materialRef}
        uniforms={uniforms}
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        side={THREE.DoubleSide}
        transparent
      />
    </mesh>
  );
}

export function VoiceprintTerrain(props: Props) {
  const { height = 340, className, state } = props;

  return (
    <div
      className={`terrain terrain--${state} ${className ?? ''}`}
      style={{ height }}
      role="img"
      aria-label={
        {
          idle: 'Three-dimensional spectrogram display, idle',
          recording: 'Live three-dimensional spectrogram of microphone input',
          analysing: 'Analysing the recorded audio',
          resolved: 'Three-dimensional spectrogram of the analysed clip',
        }[state]
      }
    >
      <Canvas
        // Clamped: a 3x retina buffer on a full-width canvas costs far more than
        // it shows, and this runs on whatever laptop a visitor has.
        dpr={[1, 2]}
        camera={{ position: [0, 1.1, 6.6], fov: 46 }}
        gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
        // Only the resolved state is static. Idle drifts, recording scrolls,
        // and analysing sweeps — putting those on 'demand' starved them of
        // frames, so `delta` arrived in large jumps and the scan lurched ahead
        // of its own progress bar instead of crossing smoothly.
        frameloop={state === 'resolved' ? 'demand' : 'always'}
      >
        <Terrain {...props} />
        {/* Orbit only once there is a real result worth inspecting. Letting the
            user drag an idle ambient surface implies it means something. */}
        {state === 'resolved' ? (
          <OrbitControls
            enablePan={false}
            enableZoom={false}
            minPolarAngle={Math.PI * 0.15}
            maxPolarAngle={Math.PI * 0.62}
            rotateSpeed={0.55}
          />
        ) : null}
      </Canvas>
    </div>
  );
}
