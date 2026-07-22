/**
 * The embedding space, as a place you can move through.
 *
 * 2511 clips as points in the encoder's own 192-dimension space, projected to
 * three. Each cluster is one speaker; the distance between clusters is exactly
 * what identification measures, so the picture is the algorithm rather than an
 * illustration of it.
 *
 * Held-out speakers — the ten the encoder never trained on — render brighter and
 * larger. That they still form their own tight clusters is the whole argument
 * for open-set recognition, and it is visible at a glance here in a way no
 * table of numbers manages.
 *
 * One InstancedMesh draws all of it in a single call.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

import { ramp } from '../lib/colormap';
import type { SpeakerMap } from '../lib/api';

interface Props {
  map: SpeakerMap;
  theme: 'scope' | 'print';
  showHeldout: boolean;
  /** Where the user's own clip landed, if they have analysed one. */
  marker?: { x: number; y: number; z: number } | null;
  onHover?: (info: { speaker: string; heldout: boolean } | null) => void;
  height?: number;
}

const SCALE = 8;

function Points({ map, theme, showHeldout, onHover }: Omit<Props, 'height'>) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const groupRef = useRef<THREE.Group>(null);
  const [hovered, setHovered] = useState<number | null>(null);
  const { invalidate } = useThree();

  const visible = useMemo(
    () => map.points.filter((p) => showHeldout || !p.h),
    [map.points, showHeldout],
  );

  const dummy = useMemo(() => new THREE.Object3D(), []);

  // Position and colour every instance. Re-runs only when the visible set or
  // the theme changes, never per frame.
  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;

    const color = new THREE.Color();
    const total = Math.max(1, map.speakers.length - 1);

    visible.forEach((point, i) => {
      dummy.position.set(point.x * SCALE, (point.z ?? 0) * SCALE, point.y * SCALE);
      const size = point.h ? 0.20 : 0.13;
      dummy.scale.setScalar(size);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);

      const [r, g, b] = ramp(0.2 + (point.s / total) * 0.75);
      color.setRGB(r / 255, g / 255, b / 255);
      // Paper mode needs the points darker than the ramp's bright end or they
      // vanish against a light background.
      if (theme === 'print') color.multiplyScalar(0.55);
      mesh.setColorAt(i, color);
    });

    mesh.count = visible.length;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    invalidate();
    // `hovered` is deliberately NOT a dependency. Including it rebuilt all 2511
    // instances and re-uploaded both GPU buffers on every pointer move; the
    // hover highlight is a single separate mesh instead (see Highlight below).
  }, [visible, theme, dummy, map.speakers.length, invalidate]);

  // No idle drift. This canvas renders on demand so 2511 instances are not
  // redrawn 60 times a second for a static picture, which means a useFrame
  // rotation here would simply never run. Depth is carried by the fog and by
  // dragging instead; the caption says "Drag to orbit" for that reason.

  return (
    <group ref={groupRef}>
      {hovered !== null && visible[hovered] ? (
        <mesh
          position={[
            visible[hovered].x * SCALE,
            (visible[hovered].z ?? 0) * SCALE,
            visible[hovered].y * SCALE,
          ]}
        >
          <sphereGeometry args={[0.3, 12, 12]} />
          <meshBasicMaterial color="#ffffff" toneMapped={false} transparent opacity={0.85} />
        </mesh>
      ) : null}

      <instancedMesh
        ref={meshRef}
        args={[undefined, undefined, Math.max(1, visible.length)]}
        onPointerMove={(event) => {
          event.stopPropagation();
          const id = event.instanceId ?? null;
          setHovered(id);
          if (id !== null && onHover) {
            const point = visible[id];
            onHover({ speaker: map.speakers[point.s], heldout: Boolean(point.h) });
          }
        }}
        onPointerOut={() => {
          setHovered(null);
          onHover?.(null);
        }}
      >
        <sphereGeometry args={[1, 8, 8]} />
        <meshBasicMaterial toneMapped={false} transparent opacity={0.92} />
      </instancedMesh>
    </group>
  );
}

/** The user's own clip, dropped into the cloud and pulsing so it is findable. */
function Marker({ position }: { position: [number, number, number] }) {
  const ref = useRef<THREE.Mesh>(null);
  const t = useRef(0);

  useFrame((_, delta) => {
    t.current += delta;
    if (!ref.current) return;
    // Ease down from above on arrival, then breathe.
    const drop = Math.min(1, t.current * 1.6);
    const eased = 1 - Math.pow(1 - drop, 3);
    ref.current.position.set(position[0], position[1] + (1 - eased) * 6, position[2]);
    ref.current.scale.setScalar(0.16 + Math.sin(t.current * 3) * 0.02);
  });

  return (
    <mesh ref={ref}>
      <sphereGeometry args={[1, 16, 16]} />
      <meshBasicMaterial color="#ffffff" toneMapped={false} />
    </mesh>
  );
}

export function EmbeddingGalaxy({ height = 560, ...props }: Props) {
  const { map, marker } = props;

  return (
    <div style={{ height, width: '100%' }}>
      <Canvas
        dpr={[1, 2]}
        camera={{ position: [0, 3.5, 11.5], fov: 50 }}
        gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
        // Renders on invalidate() rather than spinning at 60 fps when nothing
        // has changed. The invalidate() calls were already here and were doing
        // nothing, because R3F defaults to 'always' without this prop.
        frameloop="demand"
      >
        <Points {...props} />
        {/* Depth cue only: fog fades distant clusters so the third dimension
            reads even in a still frame. */}
        <fog attach="fog" args={[props.theme === 'print' ? '#f1eee7' : '#070d18', 12, 30]} />
        {marker ? <Marker position={[marker.x * SCALE, (marker.z ?? 0) * SCALE, marker.y * SCALE]} /> : null}
        <OrbitControls
          enablePan={false}
          minDistance={4}
          maxDistance={34}
          rotateSpeed={0.6}
          zoomSpeed={0.7}
          makeDefault
        />
      </Canvas>
      <span className="visually-hidden">
        Three-dimensional scatter of {map.points.length} voice clips across {map.speakers.length} speakers.
      </span>
    </div>
  );
}
