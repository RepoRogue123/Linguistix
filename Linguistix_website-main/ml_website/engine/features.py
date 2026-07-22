"""Audio to features. The only place in Linguistix where this conversion happens.

Two representations live here:

``mfcc_4000``  the 4000-dim flattened MFCC vector consumed by the dense network
               and by every classical model.

``log_mel``    log-mel spectrogram, the input to the TDNN speaker encoder.

Both start from :func:`load_audio`, which always resamples to ``SAMPLE_RATE``.
Keeping that in one place is what guarantees training and serving see identical
features.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO, Union

import librosa
import numpy as np

SAMPLE_RATE = 16_000

# 4000-dim MFCC path. These values reproduce extract_archive_features.py exactly;
# changing any of them invalidates ann_model.pth and every classical model
# trained on Extracted Data/X_features.npy.
LEGACY_N_MFCC = 40
LEGACY_TARGET_DIM = 4000

# Log-mel path for the speaker encoder. 25 ms window / 10 ms hop is the standard
# speech framing and matches what torchaudio's MelSpectrogram is configured with
# in engine/encoder.py, so the two front-ends stay in agreement.
N_MELS = 80
N_FFT = 400
HOP_LENGTH = 160
WIN_LENGTH = 400
F_MIN = 20.0

# Deliberately below the usual 7600 Hz. Clips arrive at mixed source rates, and
# resampling anything above 16 kHz down leaves an anti-aliasing rolloff around
# 7.2-7.8 kHz. Mel bins in that band therefore encode recording conditions rather
# than voice, so the cutoff sits under it. Do not raise this without re-measuring.
F_MAX = 7000.0

AudioSource = Union[str, Path, bytes, BinaryIO]


def load_audio(source: AudioSource, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio as mono float32 at ``sample_rate``, resampling when needed.

    Accepts a path, raw bytes, or an open binary stream so the server can decode
    an upload without touching disk.
    """
    if isinstance(source, bytes):
        source = io.BytesIO(source)

    audio, _ = librosa.load(source, sr=sample_rate, mono=True)
    audio = np.asarray(audio, dtype=np.float32)

    if audio.size == 0:
        raise ValueError("Audio is empty after decoding.")

    return audio


def preemphasis(audio: np.ndarray, coefficient: float = 0.97) -> np.ndarray:
    """Standard first-order high-pass, flattening the spectral tilt of voiced speech."""
    return np.append(audio[0], audio[1:] - coefficient * audio[:-1]).astype(np.float32)


def trim_silence(audio: np.ndarray, top_db: float = 30.0) -> np.ndarray:
    """Drop leading and trailing silence, keeping the original if trimming empties it."""
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed if trimmed.size > 0 else audio


def mfcc_4000(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_mfcc: int = LEGACY_N_MFCC,
    target_dim: int = LEGACY_TARGET_DIM,
) -> np.ndarray:
    """Flattened fixed-length MFCC vector for the dense network and classical models.

    Frames are truncated or zero-padded to ``ceil(target_dim / n_mfcc)`` before
    flattening, then the flat vector is clamped to exactly ``target_dim``.
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)

    target_frames = max(1, int(np.ceil(target_dim / n_mfcc)))
    if mfcc.shape[1] < target_frames:
        mfcc = np.pad(mfcc, ((0, 0), (0, target_frames - mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:, :target_frames]

    flat = mfcc.flatten().astype(np.float32)
    if flat.shape[0] < target_dim:
        flat = np.pad(flat, (0, target_dim - flat.shape[0]), mode="constant")
    elif flat.shape[0] > target_dim:
        flat = flat[:target_dim]

    return flat


def log_mel(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    win_length: int = WIN_LENGTH,
) -> np.ndarray:
    """Log-mel spectrogram of shape ``(n_mels, frames)`` for the speaker encoder."""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=F_MIN,
        fmax=F_MAX,
        power=2.0,
    )
    return np.log(mel + 1e-6).astype(np.float32)


def cmvn(features: np.ndarray) -> np.ndarray:
    """Per-utterance mean and variance normalization across the time axis.

    Removes channel and recording-level offsets, which is what lets an enrolled
    voice recorded on a laptop mic match the same voice over a phone.
    """
    mean = features.mean(axis=1, keepdims=True)
    std = features.std(axis=1, keepdims=True) + 1e-5
    return ((features - mean) / std).astype(np.float32)


def extract_legacy(source: AudioSource, target_dim: int = LEGACY_TARGET_DIM) -> np.ndarray:
    """Path to the 4000-dim MFCC vector, from a file or upload straight through."""
    return mfcc_4000(load_audio(source), target_dim=target_dim)


def extract_mel(source: AudioSource, apply_cmvn: bool = True) -> np.ndarray:
    """Path to the encoder's log-mel input, from a file or upload straight through."""
    features = log_mel(trim_silence(load_audio(source)))
    return cmvn(features) if apply_cmvn else features
