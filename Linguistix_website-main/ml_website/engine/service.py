"""Process-wide inference service: loads models once, answers requests.

Holds the speaker encoder, the gallery, and the MFCC network the model jury
votes with. Everything is loaded lazily on first use and cached, so importing
this module is cheap and a missing artifact degrades the one endpoint that needs
it rather than taking down the whole process.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import numpy as np

from .features import SAMPLE_RATE, extract_legacy, load_audio, log_mel, trim_silence
from .gallery import SpeakerGallery

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

# Enrollments are written here. On Hugging Face Spaces this is container-local
# and therefore ephemeral; the value is overridable so a persistent volume can be
# mounted without a code change.
GALLERY_DB = Path(os.environ.get("LINGUISTIX_GALLERY_DB", MODELS_DIR / "gallery.sqlite3"))

# Crops averaged per request. More crops means a steadier embedding and a slower
# response; five is comfortably under a second on the free CPU tier.
INFERENCE_CROPS = int(os.environ.get("LINGUISTIX_CROPS", "5"))
CROP_SECONDS = float(os.environ.get("LINGUISTIX_CROP_SECONDS", "3.0"))


class ModelNotAvailable(RuntimeError):
    """Raised when an endpoint needs an artifact that was never built."""


class InferenceService:
    """Lazily-loaded singleton wrapping every model the API can reach."""

    def __init__(self, models_dir: Path = MODELS_DIR) -> None:
        self.models_dir = Path(models_dir)

        # Reentrant on purpose. The lazy loaders call each other: resolving the
        # gallery needs the encoder's calibrated threshold, so `gallery` acquires
        # this lock and then reaches `encoder`, which acquires it again. With a
        # plain Lock that self-deadlocks, and only on request orderings that
        # touch the gallery before the encoder — /api/match does, /api/identify
        # does not, so it hides until something calls them in the wrong order.
        self._lock = threading.RLock()
        self._encoder = None
        self._encoder_meta: dict = {}
        self._gallery: SpeakerGallery | None = None
        self._legacy = None
        self._jury = None
        self._load_errors: dict[str, str] = {}

    # ---------------------------------------------------------------- loading

    @property
    def encoder(self):
        if self._encoder is None:
            with self._lock:
                if self._encoder is None:
                    self._load_encoder()
        return self._encoder

    def _load_encoder(self) -> None:
        import torch

        from .encoder import SpeakerEncoder, load_encoder_state

        path = self.models_dir / "encoder.pt"
        if not path.exists():
            raise ModelNotAvailable(
                "encoder.pt not found. Run training/train_encoder.py to build it."
            )

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})

        model = SpeakerEncoder(
            channels=config.get("channels", 256),
            embedding_dim=config.get("embedding_dim", 192),
        )
        load_encoder_state(model, checkpoint["state_dict"])
        model.eval()

        # Note: no torch.set_grad_enabled(False) here. That setting is
        # thread-local, so applying it on the thread that happens to load the
        # model does nothing for the other threads Flask serves requests on.
        # Every inference site scopes its own torch.no_grad() instead.

        self._encoder = model
        self._encoder_meta = {
            "embedding_dim": model.embedding_dim,
            "channels": config.get("channels", 256),
            "sample_rate": SAMPLE_RATE,
            "heldout_eer": checkpoint.get("eer"),
            "heldout_min_dcf": checkpoint.get("min_dcf"),
            "cosine_threshold": checkpoint.get("threshold"),
            "trained_epoch": checkpoint.get("epoch"),
            "train_speakers": len(checkpoint.get("train_speakers", [])),
            "heldout_speakers": checkpoint.get("heldout_speakers", []),
        }

    @property
    def gallery(self) -> SpeakerGallery:
        if self._gallery is None:
            with self._lock:
                if self._gallery is None:
                    threshold = None
                    try:
                        threshold = self._encoder_meta.get("cosine_threshold") or self.encoder_meta().get(
                            "cosine_threshold"
                        )
                    except ModelNotAvailable:
                        pass
                    self._gallery = SpeakerGallery(
                        db_path=GALLERY_DB,
                        reference_path=self.models_dir / "gallery_reference.npz",
                        threshold=threshold,
                    )
        return self._gallery

    def encoder_meta(self) -> dict:
        self.encoder  # force load
        return dict(self._encoder_meta)

    # -------------------------------------------------------------- embedding

    def _crops(self, audio: np.ndarray, n_crops: int) -> np.ndarray:
        """Evenly spaced fixed-length windows spanning the clip."""
        width = int(CROP_SECONDS * SAMPLE_RATE)

        if len(audio) <= width:
            return np.pad(audio, (0, width - len(audio)))[None, :]

        starts = np.linspace(0, len(audio) - width, num=n_crops, dtype=int)
        return np.stack([audio[s : s + width] for s in starts])

    @staticmethod
    def speech_likeness(audio: np.ndarray) -> float:
        """How speech-shaped this audio is, in [0, 1].

        The acceptance threshold is calibrated on speech-against-speech trials,
        so anything that is not speech sits outside the distribution it was fitted
        on and can land anywhere: a pure 440 Hz tone measures around 0.40 cosine
        against a dataset speaker, which clears the 0.391 threshold on its own.

        Rather than raise the threshold (which would cost real rejections on real
        voices), this checks the property separating the two cases: speech
        spreads energy across many frequency bands and keeps moving, while a tone
        occupies one band and holds still. Reported alongside every result and
        used only to downgrade a match to "inconclusive", never to override a
        rejection.
        """
        mel = log_mel(audio)
        power = np.exp(mel)

        # Spectral flatness: near 1 for noise-like, near 0 for a pure tone.
        # Speech sits in between, so both extremes are penalised.
        geometric = np.exp(np.mean(np.log(power + 1e-10), axis=0))
        arithmetic = np.mean(power, axis=0) + 1e-10
        flatness = float(np.mean(geometric / arithmetic))

        # Spread: fraction of bands within 20 dB of the frame peak. A tone lights
        # up one or two; speech lights up many.
        peak = power.max(axis=0, keepdims=True)
        spread = float(np.mean(power > peak * 0.01))

        # Movement: how much the spectrum changes frame to frame. A sustained
        # tone barely changes; speech is articulated and therefore non-stationary.
        movement = float(np.mean(np.abs(np.diff(mel, axis=1)))) if mel.shape[1] > 1 else 0.0

        return float(
            np.clip(
                min(spread / 0.25, 1.0) * 0.5
                + min(movement / 0.6, 1.0) * 0.35
                + min(flatness / 0.15, 1.0) * 0.15,
                0.0,
                1.0,
            )
        )

    def embed(self, source, n_crops: int | None = None) -> tuple[np.ndarray, dict]:
        """Audio to one averaged, L2-normalized embedding, with timing metadata."""
        import torch

        started = time.perf_counter()
        audio = load_audio(source)
        raw_duration = len(audio) / SAMPLE_RATE

        trimmed = trim_silence(audio)
        if len(trimmed) < SAMPLE_RATE * 0.4:
            # Under ~0.4 s there is not enough voiced material for a stable
            # embedding, so say so rather than returning a confident-looking guess.
            raise ValueError(
                f"Only {len(trimmed)/SAMPLE_RATE:.2f}s of audio survived silence trimming. "
                "Speak for at least a second."
            )

        crops = self._crops(trimmed, n_crops or INFERENCE_CROPS)
        batch = torch.from_numpy(crops.astype(np.float32))
        with torch.no_grad():
            vectors = self.encoder(batch).numpy()

        mean = vectors.mean(axis=0)
        embedding = mean / max(float(np.linalg.norm(mean)), 1e-8)

        # Spread across crops is a usable confidence signal on its own: a clip
        # with one speaker throughout gives tightly agreeing crops, while a noisy
        # or multi-speaker clip does not.
        consistency = float(np.mean(vectors @ embedding)) if len(vectors) > 1 else 1.0

        return embedding, {
            "duration_seconds": round(raw_duration, 2),
            "voiced_seconds": round(len(trimmed) / SAMPLE_RATE, 2),
            "crops": int(crops.shape[0]),
            "crop_consistency": round(consistency, 4),
            "speech_likeness": round(self.speech_likeness(trimmed), 4),
            "embed_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    # ------------------------------------------------------------ spectrogram

    def spectrogram(self, source, max_frames: int = 400) -> dict:
        """Downsampled log-mel for the sonagram strip, as small integers.

        Quantized to 0-255 and sent as plain lists: the client only needs enough
        resolution to draw, and shipping float64 would multiply the payload for
        no visible gain.
        """
        audio = load_audio(source)
        mel = log_mel(trim_silence(audio))

        if mel.shape[1] > max_frames:
            idx = np.linspace(0, mel.shape[1] - 1, max_frames).astype(int)
            mel = mel[:, idx]

        lo, hi = float(mel.min()), float(mel.max())
        scaled = ((mel - lo) / max(hi - lo, 1e-6) * 255).astype(np.uint8)

        return {
            "n_mels": int(mel.shape[0]),
            "frames": int(mel.shape[1]),
            "duration_seconds": round(len(audio) / SAMPLE_RATE, 2),
            "data": scaled.tolist(),
        }

    # ------------------------------------------------------------ legacy path

    @property
    def legacy(self):
        """The dense 4000-dim MFCC network, one of the jury's voters."""
        if self._legacy is None:
            with self._lock:
                if self._legacy is None:
                    self._load_legacy()
        return self._legacy

    def _load_legacy(self) -> None:
        import torch

        from ..model import ANN

        weights = self.models_dir / "ann_model.pth"
        names = self.models_dir / "speaker_names.npy"
        mean = self.models_dir / "feature_mean.npy"
        std = self.models_dir / "feature_std.npy"

        missing = [p.name for p in (weights, names, mean, std) if not p.exists()]
        if missing:
            raise ModelNotAvailable(f"MFCC network artifacts missing: {', '.join(missing)}")

        speaker_names = np.load(names, allow_pickle=True)
        model = ANN(input_size=4000, hidden_size=128, output_size=len(speaker_names))
        model.load_state_dict(torch.load(weights, map_location="cpu"))
        model.eval()

        self._legacy = {
            "model": model,
            "names": [str(n) for n in speaker_names],
            "mean": np.load(mean).astype(np.float32),
            "std": np.load(std).astype(np.float32),
        }

    def legacy_predict(self, source, top_k: int = 5) -> dict:
        """Run the closed-set MFCC network, for comparison against the encoder."""
        import torch

        legacy = self.legacy
        features = extract_legacy(source)
        standardized = (features - legacy["mean"]) / np.maximum(legacy["std"], 1e-8)

        with torch.no_grad():
            logits = legacy["model"](torch.from_numpy(standardized.astype(np.float32)).unsqueeze(0))
            probabilities = torch.softmax(logits, dim=1)[0].numpy()

        order = np.argsort(probabilities)[::-1][:top_k]
        return {
            "model": "MFCC Neural Net (4000-d, closed set)",
            "closed_set": True,
            "predictions": [
                {
                    "rank": rank + 1,
                    "speaker": legacy["names"][i],
                    "confidence": round(float(probabilities[i]) * 100, 2),
                }
                for rank, i in enumerate(order)
            ],
        }

    # ------------------------------------------------------------------ state

    def health(self) -> dict:
        """Report what loaded and what did not, without raising."""
        import torch

        status: dict = {
            "status": "ok",
            "torch": torch.__version__,
            "device": "cpu",
            "sample_rate": SAMPLE_RATE,
            "crops_per_request": INFERENCE_CROPS,
            "crop_seconds": CROP_SECONDS,
            "components": {},
        }

        try:
            status["components"]["encoder"] = {"loaded": True, **self.encoder_meta()}
        except Exception as exc:
            status["components"]["encoder"] = {"loaded": False, "error": str(exc)}
            status["status"] = "degraded"

        try:
            status["components"]["gallery"] = {"loaded": True, **self.gallery.stats()}
        except Exception as exc:
            status["components"]["gallery"] = {"loaded": False, "error": str(exc)}
            status["status"] = "degraded"

        try:
            status["components"]["mfcc_net"] = {"loaded": True, "classes": len(self.legacy["names"])}
        except Exception as exc:
            status["components"]["mfcc_net"] = {"loaded": False, "error": str(exc)}

        for name in ("benchmarks.json", "encoder_history.json", "speaker_map.json", "manifest.json"):
            status["components"][name] = {"available": (self.models_dir / name).exists()}

        return status


_service: InferenceService | None = None


def get_service() -> InferenceService:
    global _service
    if _service is None:
        _service = InferenceService()
    return _service
