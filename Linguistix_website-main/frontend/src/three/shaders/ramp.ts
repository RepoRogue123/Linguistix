/**
 * The data ramp, as a GLSL chunk.
 *
 * Ember: silence is deep navy, energy climbs through steel blue and crosses
 * into amber at the peak. It encodes spectrogram energy, match confidence, and
 * speaker identity, so it is the one palette that must mean the same thing
 * everywhere in the app.
 *
 * The stops arrive as uniforms read from CSS custom properties rather than
 * being written here. That matters: the previous version encoded the ramp
 * twice — once as JS stops and once as a hand-fitted polynomial in GLSL — so
 * changing one without refitting the other silently desynced the 2D canvases
 * from the 3D surfaces. There is now one source of truth, and a palette change
 * is a token edit.
 *
 * PRINT swaps to ink density: high energy burns dark, the way a sound
 * spectrograph marked paper. The theme changes what the surface *means*, not
 * just its colour.
 */

export const RAMP_GLSL = /* glsl */ `
uniform vec3 uRamp0;
uniform vec3 uRamp1;
uniform vec3 uRamp2;
uniform vec3 uRamp3;
uniform vec3 uRamp4;
uniform vec3 uRamp5;

vec3 rampColor(float t) {
  t = clamp(t, 0.0, 1.0);
  float s = t * 5.0;
  vec3 c = mix(uRamp0, uRamp1, clamp(s, 0.0, 1.0));
  c = mix(c, uRamp2, clamp(s - 1.0, 0.0, 1.0));
  c = mix(c, uRamp3, clamp(s - 2.0, 0.0, 1.0));
  c = mix(c, uRamp4, clamp(s - 3.0, 0.0, 1.0));
  c = mix(c, uRamp5, clamp(s - 4.0, 0.0, 1.0));
  return c;
}

vec3 inkDensity(float t) {
  t = clamp(t, 0.0, 1.0);
  // Gamma matches lib/colormap.ts so paper reads identically in 2D and 3D.
  float d = pow(t, 0.75);
  return mix(vec3(0.945, 0.933, 0.906), vec3(0.078, 0.071, 0.059), d);
}

// mode 0 = scope (colour), 1 = print (ink on paper)
vec3 rampColor(float t, float mode) {
  return mix(rampColor(t), inkDensity(t), mode);
}

/**
 * THREE.Color converts hex to linear on construction, but a raw ShaderMaterial
 * writing gl_FragColor gets no automatic output conversion — so linear values
 * would be written as though they were already sRGB and land far too dark.
 */
vec3 toSRGB(vec3 c) {
  return pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.2));
}
`;

/**
 * Value noise, used for the terrain relief.
 *
 * Two octaves, not four. The four-octave version cost 8 lattice corners x 3
 * sin() x 4 octaves per call, and with five calls per pixel in the old
 * fullscreen backdrop that reached roughly 2.2 billion transcendentals per
 * frame. Nothing in this app needs more than two octaves of detail on a
 * surface that is already displaced by real spectrogram data.
 */
export const NOISE_GLSL = /* glsl */ `
vec3 hash3(vec3 p) {
  p = vec3(dot(p, vec3(127.1, 311.7, 74.7)),
           dot(p, vec3(269.5, 183.3, 246.1)),
           dot(p, vec3(113.5, 271.9, 124.6)));
  return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

float noise(vec3 p) {
  vec3 i = floor(p);
  vec3 f = fract(p);
  vec3 u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(mix(dot(hash3(i + vec3(0,0,0)), f - vec3(0,0,0)),
            dot(hash3(i + vec3(1,0,0)), f - vec3(1,0,0)), u.x),
        mix(dot(hash3(i + vec3(0,1,0)), f - vec3(0,1,0)),
            dot(hash3(i + vec3(1,1,0)), f - vec3(1,1,0)), u.x), u.y),
    mix(mix(dot(hash3(i + vec3(0,0,1)), f - vec3(0,0,1)),
            dot(hash3(i + vec3(1,0,1)), f - vec3(1,0,1)), u.x),
        mix(dot(hash3(i + vec3(0,1,1)), f - vec3(0,1,1)),
            dot(hash3(i + vec3(1,1,1)), f - vec3(1,1,1)), u.x), u.y),
    u.z);
}

float fbm(vec3 p) {
  float total = noise(p) * 0.5;
  p *= 2.02;
  total += noise(p) * 0.25;
  return total;
}
`;

/** The stops, in ramp order, as CSS custom property names. */
export const RAMP_VARS = ['--ramp-0', '--ramp-1', '--ramp-2', '--ramp-3', '--ramp-4', '--ramp-5'] as const;

/** Fallbacks matching tokens.css, so a missing stylesheet degrades rather than breaks. */
export const RAMP_FALLBACK = ['#0a1428', '#1d4e78', '#4b87a8', '#9ab4a8', '#e8944a', '#fbe3b0'] as const;

/**
 * Build the ramp uniforms from the CSS tokens.
 *
 * Every shader that draws data calls this, so all of them read the same stops
 * the 2D canvases do. Colours are constructed from hex, which THREE converts to
 * linear — shaders must therefore call `toSRGB()` before writing gl_FragColor.
 */
export function rampUniforms(THREE: typeof import('three')) {
  const styles = typeof window !== 'undefined' ? getComputedStyle(document.documentElement) : null;
  const uniforms: Record<string, { value: InstanceType<typeof THREE.Color> }> = {};

  RAMP_VARS.forEach((name, i) => {
    const hex = styles?.getPropertyValue(name).trim() || RAMP_FALLBACK[i];
    uniforms[`uRamp${i}`] = { value: new THREE.Color(hex) };
  });
  return uniforms;
}
