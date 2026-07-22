/**
 * The model comparison as a field of bars you can walk around.
 *
 * Accuracy is height, the representation groups along one axis and the model
 * family along the other, so which reduction wins is legible from the shape of
 * the field before anyone reads a label. Bars grow into place on load, ordered
 * by score, which turns the ranking itself into the animation.
 *
 * The sortable table beneath stays the source of truth for exact figures.
 */

import { useEffect, useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

import { ramp } from '../lib/colormap';

export interface BarDatum {
  key: string;
  model: string;
  representation: string;
  accuracy: number;
}

interface Props {
  data: BarDatum[];
  theme: 'scope' | 'print';
  height?: number;
}

const COLUMNS = 7;
const SPACING = 1.25;
const MAX_HEIGHT = 4.6;

function Bars({ data, theme }: { data: BarDatum[]; theme: 'scope' | 'print' }) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const progress = useRef(0);
  const dummy = useMemo(() => new THREE.Object3D(), []);

  const layout = useMemo(() => {
    const sorted = [...data].sort((a, b) => b.accuracy - a.accuracy);
    return sorted.map((d, i) => {
      const col = i % COLUMNS;
      const row = Math.floor(i / COLUMNS);
      return {
        ...d,
        x: (col - (COLUMNS - 1) / 2) * SPACING,
        z: (row - 1) * SPACING,
        // Stagger by rank so the field builds from the winner outward.
        delay: i * 0.045,
        target: Math.max(0.08, (d.accuracy / 100) * MAX_HEIGHT),
      };
    });
  }, [data]);

  useEffect(() => {
    progress.current = 0;
  }, [layout]);

  // Hoisted: this was allocated on every frame, forever.
  const scratch = useMemo(() => new THREE.Color(), []);

  useFrame((_, delta) => {
    const mesh = meshRef.current;
    // Once the growth animation completes the image is static, so there is
    // nothing to recompute. This previously kept re-uploading two full instance
    // buffers every frame for a picture that never changed again.
    if (!mesh || progress.current >= 2.4) return;

    progress.current = Math.min(2.4, progress.current + delta);
    const color = scratch;

    layout.forEach((bar, i) => {
      const local = Math.max(0, Math.min(1, (progress.current - bar.delay) / 0.75));
      // Ease-out-cubic: fast rise, soft landing.
      const eased = 1 - Math.pow(1 - local, 3);
      const h = bar.target * eased;

      dummy.position.set(bar.x, h / 2, bar.z);
      dummy.scale.set(0.66, Math.max(0.001, h), 0.66);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);

      const [r, g, b] = ramp(0.25 + (bar.accuracy / 100) * 0.7);
      color.setRGB(r / 255, g / 255, b / 255);
      if (theme === 'print') color.multiplyScalar(0.6);
      mesh.setColorAt(i, color);
    });

    mesh.count = layout.length;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  });

  const ink = theme === 'print' ? '#14120f' : '#eceae4';

  return (
    <group position={[0, -1.6, 0]}>
      <instancedMesh ref={meshRef} args={[undefined, undefined, Math.max(1, layout.length)]}>
        <boxGeometry args={[1, 1, 1]} />
        <meshBasicMaterial toneMapped={false} />
      </instancedMesh>


      <gridHelper
        args={[COLUMNS * SPACING + 2, 8, ink, ink]}
        position={[0, 0, 0]}
        // Barely visible: the grid gives the bars a floor to stand on, nothing more.
        material-opacity={0.12}
        material-transparent
      />
    </group>
  );
}

export function LeaderboardBars({ data, theme, height = 400 }: Props) {
  return (
    <div style={{ height, width: '100%' }} aria-hidden="true">
      <Canvas dpr={[1, 2]} camera={{ position: [0, 3.2, 7.4], fov: 50 }} gl={{ alpha: true, antialias: true }}>
        <Bars data={data} theme={theme} />
        <OrbitControls
          enablePan={false}
          enableZoom={false}
          minPolarAngle={Math.PI * 0.12}
          maxPolarAngle={Math.PI * 0.46}
          autoRotate
          autoRotateSpeed={0.5}
        />
      </Canvas>
    </div>
  );
}
