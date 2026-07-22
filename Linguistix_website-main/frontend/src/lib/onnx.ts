/**
 * The speaker encoder, running in the browser.
 *
 * The deployed Space is free-tier CPU, so anything interactive — a slider the
 * user drags, a rolling window updating several times a second — cannot afford
 * a server round trip per frame. Those paths run the exported ONNX graph here
 * instead. The server keeps the authoritative calls (enrolment, the jury,
 * saliency), where one request per action is fine.
 *
 * This is only safe because the mel front-end is inside the graph, so the
 * browser feeds raw 16 kHz PCM and never reimplements feature extraction.
 * training/export_onnx.py asserts the two agree to under 1e-4 before shipping.
 */

import * as ort from 'onnxruntime-web';

import { TARGET_SAMPLE_RATE } from './audio';

export interface EncoderMeta {
  sample_rate: number;
  embedding_dim: number;
  cosine_threshold: number | null;
  heldout_eer: number | null;
  size_mb: number;
  max_parity_drift?: number;
}

let sessionPromise: Promise<{ session: ort.InferenceSession; meta: EncoderMeta }> | null = null;

/** Load the graph once per page and reuse it. */
export function loadEncoder(): Promise<{ session: ort.InferenceSession; meta: EncoderMeta }> {
  if (sessionPromise) return sessionPromise;

  sessionPromise = (async () => {
    ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 1);
    ort.env.wasm.simd = true;

    const [session, meta] = await Promise.all([
      ort.InferenceSession.create('/models/encoder.onnx', {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      }),
      fetch('/models/encoder.json').then((r) => r.json() as Promise<EncoderMeta>),
    ]);

    return { session, meta };
  })();

  // Let a failed load be retried rather than caching the rejection forever.
  sessionPromise.catch(() => {
    sessionPromise = null;
  });

  return sessionPromise;
}

/** Embed one mono 16 kHz buffer. Returns an L2-normalized vector. */
export async function embed(samples: Float32Array): Promise<Float32Array> {
  const { session } = await loadEncoder();

  // Below ~0.4 s the attentive pooling has too few frames to be stable.
  const minimum = Math.floor(TARGET_SAMPLE_RATE * 0.4);
  let input = samples;
  if (input.length < minimum) {
    const padded = new Float32Array(minimum);
    padded.set(input);
    input = padded;
  }

  const tensor = new ort.Tensor('float32', input, [1, input.length]);
  const output = await session.run({ [session.inputNames[0]]: tensor });
  return output[session.outputNames[0]].data as Float32Array;
}

/** Cosine similarity. Both inputs are already unit length, so this is a dot product. */
export function cosine(a: Float32Array | number[], b: Float32Array | number[]): number {
  let sum = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) sum += a[i] * b[i];
  return sum;
}

// ---- degradations for the robustness lab ---------------------------------

export interface Degradation {
  /** Additive white noise, expressed as SNR in dB. Infinity leaves it clean. */
  snrDb: number;
  /** One-pole lowpass cutoff in Hz. Infinity disables it. */
  lowpassHz: number;
  /** Playback rate multiplier: shifts pitch and tempo together. */
  speed: number;
  /** Trim to this many seconds. Infinity keeps the whole clip. */
  seconds: number;
  /** Resample through this rate and back, reproducing a sample-rate mismatch. */
  viaRate: number | null;
}

export const CLEAN: Degradation = {
  snrDb: Infinity,
  lowpassHz: Infinity,
  speed: 1,
  seconds: Infinity,
  viaRate: null,
};

function addNoise(samples: Float32Array, snrDb: number): Float32Array {
  if (!isFinite(snrDb)) return samples;
  let power = 0;
  for (let i = 0; i < samples.length; i++) power += samples[i] * samples[i];
  power = power / Math.max(1, samples.length);

  const noisePower = power / Math.pow(10, snrDb / 10);
  const scale = Math.sqrt(noisePower);
  const out = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    // Box-Muller for genuinely Gaussian noise; summed uniforms would have the
    // wrong tails and understate how bad real noise is.
    const u = Math.max(1e-9, Math.random());
    const v = Math.random();
    const gauss = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    out[i] = samples[i] + gauss * scale;
  }
  return out;
}

function lowpass(samples: Float32Array, cutoffHz: number): Float32Array {
  if (!isFinite(cutoffHz) || cutoffHz >= TARGET_SAMPLE_RATE / 2) return samples;

  const dt = 1 / TARGET_SAMPLE_RATE;
  const rc = 1 / (2 * Math.PI * cutoffHz);
  const alpha = dt / (rc + dt);

  const out = new Float32Array(samples.length);
  let previous = samples[0];
  for (let i = 0; i < samples.length; i++) {
    previous += alpha * (samples[i] - previous);
    out[i] = previous;
  }
  return out;
}

function resampleLinear(samples: Float32Array, ratio: number): Float32Array {
  if (ratio === 1) return samples;
  const length = Math.max(1, Math.floor(samples.length / ratio));
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const src = i * ratio;
    const i0 = Math.floor(src);
    const i1 = Math.min(samples.length - 1, i0 + 1);
    const f = src - i0;
    out[i] = samples[i0] * (1 - f) + samples[i1] * f;
  }
  return out;
}

/**
 * Apply a degradation chain in the browser.
 *
 * `viaRate` round-trips through another sample rate with a naive resampler, so
 * the cost of a rate mismatch is visible rather than theoretical.
 */
export function degrade(samples: Float32Array, options: Degradation): Float32Array {
  let out = samples;

  if (isFinite(options.seconds)) {
    out = out.slice(0, Math.max(1, Math.floor(options.seconds * TARGET_SAMPLE_RATE)));
  }
  if (options.viaRate) {
    const up = resampleLinear(out, TARGET_SAMPLE_RATE / options.viaRate);
    out = resampleLinear(up, options.viaRate / TARGET_SAMPLE_RATE);
  }
  if (options.speed !== 1) out = resampleLinear(out, options.speed);
  out = lowpass(out, options.lowpassHz);
  out = addNoise(out, options.snrDb);

  for (let i = 0; i < out.length; i++) out[i] = Math.max(-1, Math.min(1, out[i]));
  return out;
}
