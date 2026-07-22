/**
 * Typed client for the Linguistix API.
 *
 * Every response carries `ok`, and errors arrive as JSON rather than an HTML
 * error page, so one helper can normalize both transport and application
 * failures into a single thrown ApiError.
 */

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly payload?: unknown,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, init);
  } catch (cause) {
    throw new ApiError('Could not reach the server. Is the backend running?', 0, cause);
  }

  const text = await response.text();
  let payload: any = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      throw new ApiError(`Server returned non-JSON (${response.status}).`, response.status, text);
    }
  }

  if (!response.ok || payload?.ok === false) {
    throw new ApiError(payload?.error ?? `Request failed (${response.status}).`, response.status, payload);
  }
  return payload as T;
}

// ---- shapes ---------------------------------------------------------------

export interface Candidate {
  speaker: string;
  score: number;
  similarity_pct: number;
  source: 'reference' | 'enrolled';
  samples: number;
}

export interface Identification {
  matched: boolean;
  reason: string;
  speaker: string | null;
  closest?: string;
  score?: number;
  margin?: number;
  threshold: number;
  source?: string;
  candidates: Candidate[];
  /** Present when a match was downgraded because the audio was not speech. */
  speech_likeness?: number;
  explanation?: string;
}

export interface AudioMeta {
  duration_seconds: number;
  voiced_seconds: number;
  crops: number;
  crop_consistency: number;
  speech_likeness?: number;
  embed_ms: number;
}

export interface Spectrogram {
  n_mels: number;
  frames: number;
  duration_seconds: number;
  data: number[][];
}

export interface IdentifyResult {
  ok: true;
  identification: Identification;
  audio: AudioMeta;
  embedding: number[];
  spectrogram?: Spectrogram;
  total_ms: number;
}

export interface Vote {
  model: string;
  key: string;
  family: 'open-set' | 'closed-set';
  speaker: string | null;
  confidence: number | null;
  calibrated: boolean;
  accepted?: boolean;
  note?: string;
  ms: number;
}

export interface JuryResult {
  ok: true;
  votes: Vote[];
  unavailable: { model: string; reason: string }[];
  consensus: {
    speaker: string | null;
    votes_for: number;
    total_voters: number;
    agreement: number;
    unanimous: boolean;
    distinct_answers: number;
    tally: Record<string, number>;
  };
  audio: AudioMeta;
  note: string;
}

export interface ExplainResult {
  ok: true;
  n_mels: number;
  frames: number;
  duration_seconds: number;
  analysed_seconds: number;
  saliency: number[][];
  spectrogram: number[][];
  band_profile: number[];
  frame_profile: number[];
  top_bands_hz: { hz: number; weight: number }[];
  top_moments_seconds: { at: number; weight: number }[];
  method: string;
  caveat: string;
}

export interface SpeakerMap {
  ok: true;
  projection: string;
  source: string;
  note: string;
  speakers: string[];
  heldout_speakers: string[];
  dims?: number;
  points: { x: number; y: number; z?: number; s: number; h: number }[];
}

export interface Health {
  ok: true;
  status: 'ok' | 'degraded';
  torch: string;
  device: string;
  sample_rate: number;
  crops_per_request: number;
  crop_seconds: number;
  components: Record<string, any>;
}

export interface Metrics {
  ok: true;
  benchmarks: any;
  encoder_history: any;
  dataset?: {
    clips: number;
    speakers: string[];
    clips_per_speaker: Record<string, number>;
    split_sizes: Record<string, number>;
    heldout_speakers: string[];
  };
}

export interface GalleryState {
  ok: true;
  enrolled: { name: string; samples: number; created_at: string; updated_at: string }[];
  reference_speakers: number;
  enrolled_speakers: number;
  enrolled_samples: number;
  threshold: number;
  embedding_dim: number | null;
  persistent: boolean;
  storage_note: string;
}

// ---- calls ----------------------------------------------------------------

function audioForm(blob: Blob, field = 'audio', filename = 'clip.wav'): FormData {
  const form = new FormData();
  form.append(field, blob, filename);
  return form;
}

export const api = {
  health: () => request<Health>('/api/health'),

  identify: (clip: Blob, topK = 5) =>
    request<IdentifyResult>(`/api/identify?top_k=${topK}`, { method: 'POST', body: audioForm(clip) }),

  embed: (clip: Blob) =>
    request<{ ok: true; embedding: number[]; audio: AudioMeta }>('/api/embed', {
      method: 'POST',
      body: audioForm(clip),
    }),

  jury: (clip: Blob) => request<JuryResult>('/api/jury', { method: 'POST', body: audioForm(clip) }),

  explain: (clip: Blob) => request<ExplainResult>('/api/explain', { method: 'POST', body: audioForm(clip) }),

  verify: (a: Blob, b: Blob) => {
    const form = new FormData();
    form.append('audio_a', a, 'a.wav');
    form.append('audio_b', b, 'b.wav');
    return request<{ ok: true; verification: any; audio_a: AudioMeta; audio_b: AudioMeta }>('/api/verify', {
      method: 'POST',
      body: form,
    });
  },

  enroll: (name: string, clips: Blob[], replace = false) => {
    const form = new FormData();
    form.append('name', name);
    if (replace) form.append('replace', '1');
    clips.forEach((clip, i) => form.append('audio', clip, `clip-${i}.wav`));
    return request<{ ok: true; enrolled: any; accepted: any[]; rejected: any[]; warning?: string }>(
      '/api/enroll',
      { method: 'POST', body: form },
    );
  },

  gallery: () => request<GalleryState>('/api/gallery'),
  removeSpeaker: (name: string) =>
    request<{ ok: true; deleted: string }>(`/api/gallery/${encodeURIComponent(name)}`, { method: 'DELETE' }),

  map: () => request<SpeakerMap>('/api/map'),
  metrics: () => request<Metrics>('/api/metrics'),
  speakers: () => request<{ ok: true; speakers: any[] }>('/api/speakers'),
};
