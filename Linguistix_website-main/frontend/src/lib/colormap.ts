/**
 * The Ember ramp — the one palette that encodes data everywhere in the app.
 *
 * Silence is deep navy; energy climbs through steel blue and crosses into amber
 * at the peak. Perceptually ordered and monotonic in lightness, so it degrades
 * correctly to grayscale and a brighter pixel always means more energy.
 *
 * These stops are the single source of truth. The GLSL side reads the same
 * values through CSS custom properties as uniforms rather than re-encoding them
 * (see three/shaders/ramp.ts), which is what keeps the 2D canvases and the 3D
 * surfaces from drifting apart.
 */

type RGB = [number, number, number];

/** Read the ramp from CSS so the tokens stay authoritative. */
function readStops(): RGB[] {
  if (typeof window === 'undefined') return FALLBACK_STOPS;

  const styles = getComputedStyle(document.documentElement);
  const stops: RGB[] = [];

  for (let i = 0; i < 6; i++) {
    const hex = styles.getPropertyValue(`--ramp-${i}`).trim();
    const parsed = hexToRgb(hex);
    if (!parsed) return FALLBACK_STOPS;
    stops.push(parsed);
  }
  return stops;
}

function hexToRgb(hex: string): RGB | null {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex);
  if (!m) return null;
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

const FALLBACK_STOPS: RGB[] = [
  [10, 20, 40], // #0a1428  silence
  [29, 78, 120], // #1d4e78  deep steel
  [75, 135, 168], // #4b87a8  steel
  [154, 180, 168], // #9ab4a8  pale slate (the cool-to-warm crossover)
  [232, 148, 74], // #e8944a  amber
  [251, 227, 176], // #fbe3b0  cream
];

// Resolved once. A theme switch changes ink-vs-colour, not the ramp itself, so
// there is nothing to invalidate.
let STOPS: RGB[] | null = null;

export function rampStops(): RGB[] {
  if (!STOPS) STOPS = readStops();
  return STOPS;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/** Sample the ramp at t in [0, 1]. */
export function ramp(t: number): RGB {
  const stops = rampStops();
  const clamped = Math.min(1, Math.max(0, t));
  const scaled = clamped * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(scaled));
  const f = scaled - i;
  const a = stops[i];
  const b = stops[i + 1];
  return [
    Math.round(lerp(a[0], b[0], f)),
    Math.round(lerp(a[1], b[1], f)),
    Math.round(lerp(a[2], b[2], f)),
  ];
}

/**
 * Ink density on paper: the PRINT theme's rendering.
 *
 * A sound spectrograph burned paper darker where there was more energy, so this
 * inverts — high energy means dark ink, not bright light.
 */
export function ink(t: number, paper: RGB = [241, 238, 231], darkest: RGB = [20, 18, 15]): RGB {
  const clamped = Math.min(1, Math.max(0, t));
  // Gamma matches three/shaders/ramp.ts so 2D and 3D paper agree.
  const density = Math.pow(clamped, 0.75);
  return [
    Math.round(lerp(paper[0], darkest[0], density)),
    Math.round(lerp(paper[1], darkest[1], density)),
    Math.round(lerp(paper[2], darkest[2], density)),
  ];
}

/**
 * Precompute a 256-entry lookup so per-pixel canvas rendering avoids
 * interpolating on every sample.
 */
export function buildLut(theme: 'scope' | 'print'): Uint8ClampedArray {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const [r, g, b] = theme === 'print' ? ink(i / 255) : ramp(i / 255);
    lut[i * 3] = r;
    lut[i * 3 + 1] = g;
    lut[i * 3 + 2] = b;
  }
  return lut;
}

export function rgbCss(c: RGB): string {
  return `rgb(${c[0]} ${c[1]} ${c[2]})`;
}

/**
 * Colour for a confidence value in [0, 1], drawn from the same ramp.
 *
 * Deliberately not red/amber/green. Confidence here is a similarity score and
 * belongs on the same perceptual scale as the spectrogram energy beside it; a
 * separate traffic-light palette would imply a different kind of quantity, and
 * would also fail for red-green colour blindness without the lightness ordering
 * this ramp provides.
 */
export function confidenceColor(value: number, theme: 'scope' | 'print' = 'scope'): string {
  const t = 0.25 + Math.min(1, Math.max(0, value)) * 0.75;
  return rgbCss(theme === 'print' ? ink(t) : ramp(t));
}

/**
 * Same encoding, restricted to the part of the ramp that stays legible as text.
 *
 * The full ramp is built for fills, where a dark forest green on near-black is
 * fine. As type it fails at one end or the other depending on the surface, so
 * each theme uses only the half of its ramp that contrasts with its own
 * background — a low value stays readable rather than merely faint.
 */
export function confidenceTextColor(value: number, theme: 'scope' | 'print' = 'scope'): string {
  const clamped = Math.min(1, Math.max(0, value));
  return theme === 'print'
    ? // On paper: mid-grey through to near-black.
      rgbCss(ink(0.55 + clamped * 0.45))
    : // On the console: 0.42 is the measured floor at which the lowest value
      // still clears 4.5:1 against the *panel* surface, which is lighter than
      // the ground and therefore the harder of the two. Solved rather than
      // guessed — 0.34 looked reasonable and came out at 3.54:1.
      rgbCss(ramp(0.42 + clamped * 0.58));
}
