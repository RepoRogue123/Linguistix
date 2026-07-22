"""Which parts of a voice identified it: saliency over the mel spectrogram.

Speaker-recognition demos generally stop at a name and a percentage. This answers
the question people actually ask next, which is *what about my voice gave me
away*. Nearly no public speaker-ID demo attempts it.

Method. Take the gradient of the match score with respect to the mel
spectrogram, multiply by the input (gradient x input), and reduce over frequency
and time. Where the magnitude is large, perturbing that region would move the
embedding most, so that region is carrying the identity.

An honest caveat that the UI should repeat: gradient saliency shows local
sensitivity, not causation. It says the model is reading that region, not that
the region uniquely determines who is speaking.
"""

from __future__ import annotations

import numpy as np

from .features import SAMPLE_RATE, load_audio, trim_silence

# Cap the analysed window. Saliency needs a backward pass, and on the free CPU
# tier a full 60-second clip would take long enough to feel broken.
MAX_SECONDS = 6.0


def explain_embedding(service, audio_bytes: bytes, max_frames: int = 300) -> dict:
    """Saliency map plus per-band and per-frame profiles for one clip."""
    import torch

    encoder = service.encoder

    audio = trim_silence(load_audio(audio_bytes))
    if len(audio) < SAMPLE_RATE * 0.4:
        raise ValueError("Too little voiced audio to explain. Speak for at least a second.")

    audio = audio[: int(MAX_SECONDS * SAMPLE_RATE)]
    waveform = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)

    # Explicitly enabled: this is the one endpoint that needs a backward pass,
    # and every other inference path scopes itself with no_grad().
    with torch.enable_grad():
        mel = encoder.frontend(waveform)
        mel.requires_grad_(True)

        # Run the body directly; calling the model would recompute the front-end
        # and detach us from the tensor we need gradients on.
        x = torch.relu(encoder.bn1(encoder.conv1(mel)))
        h1 = encoder.block1(x)
        h2 = encoder.block2(h1)
        h3 = encoder.block3(h2)
        x = torch.relu(encoder.mfa(torch.cat([h1, h2, h3], dim=1)))
        x = encoder.bn_pool(encoder.pooling(x))
        embedding = torch.nn.functional.normalize(encoder.bn_embed(encoder.fc(x)), p=2, dim=1)

        # Differentiate the similarity to the embedding's own direction. Using a
        # detached copy of the output as the target makes this "what supports the
        # identity this clip was given", rather than requiring a gallery match
        # that may not exist for an unknown speaker.
        target = embedding.detach()
        score = (embedding * target).sum()

        encoder.zero_grad(set_to_none=True)
        score.backward()

    saliency = (mel.grad * mel).abs().squeeze(0).detach().numpy()

    if saliency.shape[1] > max_frames:
        idx = np.linspace(0, saliency.shape[1] - 1, max_frames).astype(int)
        saliency = saliency[:, idx]
        mel_display = mel.detach().squeeze(0).numpy()[:, idx]
    else:
        mel_display = mel.detach().squeeze(0).numpy()

    def normalize(a: np.ndarray) -> np.ndarray:
        lo, hi = float(a.min()), float(a.max())
        return (a - lo) / max(hi - lo, 1e-9)

    saliency_norm = normalize(saliency)
    band_profile = normalize(saliency.mean(axis=1))
    frame_profile = normalize(saliency.mean(axis=0))

    n_mels = saliency.shape[0]
    duration = len(audio) / SAMPLE_RATE

    # Report the loudest bands in Hz rather than mel-bin indices, which are
    # meaningless outside the model.
    from .features import F_MAX, F_MIN

    mel_hz = np.linspace(F_MIN, F_MAX, n_mels)
    top_bands = np.argsort(band_profile)[::-1][:5]
    top_frames = np.argsort(frame_profile)[::-1][:5]

    return {
        "n_mels": int(n_mels),
        "frames": int(saliency.shape[1]),
        "duration_seconds": round(duration, 2),
        "analysed_seconds": round(min(duration, MAX_SECONDS), 2),
        "saliency": (saliency_norm * 255).astype(np.uint8).tolist(),
        "spectrogram": (normalize(mel_display) * 255).astype(np.uint8).tolist(),
        "band_profile": [round(float(v), 4) for v in band_profile],
        "frame_profile": [round(float(v), 4) for v in frame_profile],
        "top_bands_hz": [
            {"hz": int(mel_hz[i]), "weight": round(float(band_profile[i]), 4)} for i in top_bands
        ],
        "top_moments_seconds": [
            {
                "at": round(float(i / max(saliency.shape[1] - 1, 1) * min(duration, MAX_SECONDS)), 2),
                "weight": round(float(frame_profile[i]), 4),
            }
            for i in top_frames
        ],
        "method": "gradient x input on the log-mel front-end",
        "caveat": (
            "Saliency shows where the model is locally sensitive, not what causes identity. "
            "Bright regions are the ones that would move the embedding most if changed."
        ),
    }
