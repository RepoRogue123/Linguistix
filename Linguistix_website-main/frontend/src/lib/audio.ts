/**
 * Microphone capture and WAV encoding.
 *
 * The browser's MediaRecorder produces WebM/Opus, which soundfile and librosa
 * cannot read, so the server would reject it. Rather than adding a transcoding
 * dependency to the container, we capture raw samples through Web Audio and
 * encode WAV in the browser. That also gives us the float32 PCM the in-browser
 * ONNX encoder needs, with no second decode.
 *
 * Everything is resampled to 16 kHz because that is what the model was trained
 * on. Any other rate silently shifts the features out of distribution.
 */

export const TARGET_SAMPLE_RATE = 16000;

export interface CaptureHandle {
  stream: MediaStream;
  context: AudioContext;
  analyser: AnalyserNode;
  stop: () => Promise<Float32Array>;
  cancel: () => void;
}

/** Average channels down to mono. */
function toMono(buffer: AudioBuffer): Float32Array {
  const { numberOfChannels, length } = buffer;
  if (numberOfChannels === 1) return buffer.getChannelData(0).slice();

  const mono = new Float32Array(length);
  for (let c = 0; c < numberOfChannels; c++) {
    const data = buffer.getChannelData(c);
    for (let i = 0; i < length; i++) mono[i] += data[i];
  }
  for (let i = 0; i < length; i++) mono[i] /= numberOfChannels;
  return mono;
}

/**
 * Resample to 16 kHz via OfflineAudioContext, which uses the browser's own
 * high-quality resampler rather than naive index dropping.
 */
export async function resampleTo16k(samples: Float32Array, sourceRate: number): Promise<Float32Array> {
  if (sourceRate === TARGET_SAMPLE_RATE) return samples;

  const targetLength = Math.max(1, Math.round((samples.length * TARGET_SAMPLE_RATE) / sourceRate));
  const offline = new OfflineAudioContext(1, targetLength, TARGET_SAMPLE_RATE);

  const source = offline.createBufferSource();
  const buffer = offline.createBuffer(1, samples.length, sourceRate);
  // Copy through a fresh view: copyToChannel requires an ArrayBuffer-backed
  // array, and a Float32Array sliced out of Web Audio may be SharedArrayBuffer-backed.
  buffer.copyToChannel(new Float32Array(samples), 0);
  source.buffer = buffer;
  source.connect(offline.destination);
  source.start();

  const rendered = await offline.startRendering();
  return rendered.getChannelData(0).slice();
}

/** Decode any file the browser can read, returning mono 16 kHz float32. */
export async function decodeFileTo16k(file: File | Blob): Promise<Float32Array> {
  const bytes = await file.arrayBuffer();
  const context = new AudioContext();
  try {
    const decoded = await context.decodeAudioData(bytes);
    return await resampleTo16k(toMono(decoded), decoded.sampleRate);
  } finally {
    void context.close();
  }
}

/** Wrap 16 kHz mono float32 as a 16-bit PCM WAV blob for upload. */
export function encodeWav(samples: Float32Array, sampleRate = TARGET_SAMPLE_RATE): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  const writeString = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true); // PCM header size
  view.setUint16(20, 1, true); // format: PCM
  view.setUint16(22, 1, true); // channels
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeString(36, 'data');
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += 2;
  }

  return new Blob([view], { type: 'audio/wav' });
}

/**
 * Open the microphone and start accumulating samples.
 *
 * Uses a ScriptProcessor-free path where available. The returned analyser is
 * live, so the sonagram strip can draw from it while recording continues.
 */
export async function startCapture(): Promise<CaptureHandle> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: false,
      noiseSuppression: false,
      // Leaving AGC on would fight the random-gain augmentation the encoder was
      // trained with, and would change loudness mid-recording.
      autoGainControl: false,
    },
  });

  const context = new AudioContext();
  const source = context.createMediaStreamSource(stream);

  const analyser = context.createAnalyser();
  analyser.fftSize = 1024;
  analyser.smoothingTimeConstant = 0.6;
  source.connect(analyser);

  const chunks: Float32Array[] = [];
  const recorder = context.createScriptProcessor(4096, 1, 1);
  recorder.onaudioprocess = (event) => {
    chunks.push(event.inputBuffer.getChannelData(0).slice());
  };
  source.connect(recorder);
  // ScriptProcessor only runs while connected to a destination. Routing through
  // a muted gain node keeps it alive without playing the mic back at the user.
  const mute = context.createGain();
  mute.gain.value = 0;
  recorder.connect(mute);
  mute.connect(context.destination);

  const teardown = () => {
    recorder.disconnect();
    mute.disconnect();
    source.disconnect();
    stream.getTracks().forEach((track) => track.stop());
  };

  return {
    stream,
    context,
    analyser,
    async stop() {
      teardown();
      const total = chunks.reduce((sum, c) => sum + c.length, 0);
      const merged = new Float32Array(total);
      let offset = 0;
      for (const chunk of chunks) {
        merged.set(chunk, offset);
        offset += chunk.length;
      }
      const resampled = await resampleTo16k(merged, context.sampleRate);
      void context.close();
      return resampled;
    },
    cancel() {
      teardown();
      void context.close();
    },
  };
}

/** Peak and RMS level of a buffer, for the input meter. */
export function levels(samples: Float32Array): { peak: number; rms: number } {
  let peak = 0;
  let sumSquares = 0;
  for (let i = 0; i < samples.length; i++) {
    const v = Math.abs(samples[i]);
    if (v > peak) peak = v;
    sumSquares += samples[i] * samples[i];
  }
  return { peak, rms: Math.sqrt(sumSquares / Math.max(1, samples.length)) };
}
