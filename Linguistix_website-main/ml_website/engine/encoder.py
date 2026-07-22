"""Speaker encoder: raw waveform in, fixed-length speaker embedding out.

This is the architectural change that makes Linguistix open-set. The original
model was a 50-way softmax classifier, so recognizing a new person meant
retraining. Here the network learns an embedding space where clips of the same
speaker land close together, and identification becomes nearest-neighbor against
an enrollment gallery. Adding a speaker costs one forward pass, and rejecting an
unknown one becomes a distance threshold instead of the hand-tuned confidence and
margin gate the old server used.

Two design decisions worth knowing:

**The mel front-end lives inside the model.** The browser runs this same network
via ONNX for the live and interactive paths, and reimplementing librosa's mel
extraction in JavaScript is the classic way to get a train/serve skew that is
almost impossible to debug. Instead the model takes raw 16 kHz PCM and computes
its own features, so the browser only has to supply samples.

**The STFT is built from explicit matrix multiplies** rather than ``torch.stft``
or ``torchaudio``. Those export to ONNX unreliably across opsets; framing with
``unfold`` and applying a precomputed DFT matrix uses only MatMul and Mul, which
every ONNX runtime supports. It is also bit-comparable with the Python path,
which is what the export parity check asserts.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import F_MAX, F_MIN, HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE


class MelFrontend(nn.Module):
    """Raw waveform to log-mel, using only ONNX-portable operations.

    Mirrors ``features.log_mel`` followed by ``features.cmvn``.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        n_mels: int = N_MELS,
        f_min: float = F_MIN,
        f_max: float = F_MAX,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_bins = n_fft // 2 + 1

        window = torch.hann_window(n_fft, periodic=True)

        # Real DFT expressed as convolution kernels, shaped (n_bins, 1, n_fft).
        # A windowed DFT is a bank of FIR filters evaluated every hop, which is
        # exactly what a strided Conv1d computes, so framing and transforming
        # collapse into one op. This is also the reason the graph exports: the
        # obvious `unfold` + matmul formulation is not ONNX-representable, while
        # Conv1d is supported by every runtime including onnxruntime-web.
        k = torch.arange(self.n_bins).unsqueeze(1)
        n = torch.arange(n_fft).unsqueeze(0)
        angle = 2.0 * torch.pi * k * n / n_fft
        # persistent=False: these are deterministic constants rebuilt in __init__,
        # so keeping them out of the state dict means a checkpoint stays loadable
        # even if the front-end's internal layout changes.
        self.register_buffer("dft_cos", (torch.cos(angle) * window).float().unsqueeze(1), persistent=False)
        self.register_buffer("dft_sin", (torch.sin(angle) * window).float().unsqueeze(1), persistent=False)

        # librosa's slaney-normalized filterbank, so the Python and torch paths
        # agree to floating-point noise rather than approximately.
        import librosa

        mel_fb = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=f_min, fmax=f_max
        )
        self.register_buffer("mel_fb", torch.from_numpy(mel_fb).float(), persistent=False)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """``(batch, samples)`` to ``(batch, n_mels, frames)``, per-utterance normalized."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Center-pad to match librosa's center=True. The zero fill is deliberate:
        # librosa 0.10+ defaults to pad_mode="constant", and using reflect here
        # instead put the two front-ends visibly out of agreement at clip edges.
        pad = self.n_fft // 2
        x = F.pad(waveform, (pad, pad), mode="constant", value=0.0).unsqueeze(1)

        real = F.conv1d(x, self.dft_cos, stride=self.hop_length)  # (B, n_bins, T)
        imag = F.conv1d(x, self.dft_sin, stride=self.hop_length)
        power = real.pow(2) + imag.pow(2)

        # The mel filterbank is a 1x1 convolution over the frequency axis.
        mel = F.conv1d(power, self.mel_fb.unsqueeze(-1))  # (B, n_mels, T)
        log_mel = torch.log(mel + 1e-6)

        mean = log_mel.mean(dim=2, keepdim=True)
        std = log_mel.std(dim=2, keepdim=True) + 1e-5
        return (log_mel - mean) / std


class SpecAugment(nn.Module):
    """Randomly mask frequency bands and time spans during training.

    Lives inside the model rather than the training loop so it is governed by
    ``self.training``: it is automatically inert under ``eval()`` and therefore
    absent from the ONNX export, which cannot silently disagree with the server.

    With only 40 training speakers this is doing real work. Masking whole
    frequency bands stops the network leaning on one narrow cue (a single
    speaker's formant peak, or channel coloration from their microphone) that
    would not transfer to a voice enrolled later on different hardware.
    """

    def __init__(self, freq_mask: int = 12, time_mask: int = 20, n_freq: int = 2, n_time: int = 2) -> None:
        super().__init__()
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.n_freq = n_freq
        self.n_time = n_time

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x

        batch, n_mels, n_frames = x.shape
        x = x.clone()

        for _ in range(self.n_freq):
            width = torch.randint(0, self.freq_mask + 1, (batch,), device=x.device)
            start = (torch.rand(batch, device=x.device) * (n_mels - width).clamp(min=1)).long()
            idx = torch.arange(n_mels, device=x.device).unsqueeze(0)
            mask = (idx >= start.unsqueeze(1)) & (idx < (start + width).unsqueeze(1))
            x = x.masked_fill(mask.unsqueeze(2), 0.0)

        for _ in range(self.n_time):
            width = torch.randint(0, self.time_mask + 1, (batch,), device=x.device)
            start = (torch.rand(batch, device=x.device) * (n_frames - width).clamp(min=1)).long()
            idx = torch.arange(n_frames, device=x.device).unsqueeze(0)
            mask = (idx >= start.unsqueeze(1)) & (idx < (start + width).unsqueeze(1))
            x = x.masked_fill(mask.unsqueeze(1), 0.0)

        return x


class SEBlock(nn.Module):
    """Squeeze-and-excitation over channels, conditioned on the whole utterance."""

    def __init__(self, channels: int, bottleneck: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Conv1d(channels, bottleneck, kernel_size=1)
        self.fc2 = nn.Conv1d(bottleneck, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.mean(dim=2, keepdim=True)
        scale = torch.relu(self.fc1(scale))
        return x * torch.sigmoid(self.fc2(scale))


class SERes2Block(nn.Module):
    """Dilated Res2Net block with SE, the core repeating unit of ECAPA-TDNN.

    The Res2Net split gives multiple receptive-field scales inside one block,
    which matters for speech because speaker identity lives at several timescales
    at once: glottal pulse shape over milliseconds, vowel formants over tens of
    milliseconds, and prosody over hundreds.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int, scale: int = 8) -> None:
        super().__init__()
        assert channels % scale == 0, "channels must divide evenly into scale"
        self.scale = scale
        width = channels // scale

        self.conv_in = nn.Conv1d(channels, channels, kernel_size=1)
        self.bn_in = nn.BatchNorm1d(channels)

        padding = (kernel_size - 1) // 2 * dilation
        self.convs = nn.ModuleList(
            nn.Conv1d(width, width, kernel_size, dilation=dilation, padding=padding)
            for _ in range(scale - 1)
        )
        self.bns = nn.ModuleList(nn.BatchNorm1d(width) for _ in range(scale - 1))

        self.conv_out = nn.Conv1d(channels, channels, kernel_size=1)
        self.bn_out = nn.BatchNorm1d(channels)
        self.se = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = torch.relu(self.bn_in(self.conv_in(x)))

        chunks = torch.chunk(out, self.scale, dim=1)
        processed = [chunks[0]]
        running = chunks[1]
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if i > 0:
                running = running + chunks[i + 1]
            running = torch.relu(bn(conv(running)))
            processed.append(running)

        out = torch.cat(processed, dim=1)
        out = torch.relu(self.bn_out(self.conv_out(out)))
        return self.se(out) + residual


class AttentiveStatsPooling(nn.Module):
    """Pool frames to one vector via attention-weighted mean and standard deviation.

    Attention lets the model down-weight silence and noise rather than averaging
    them in, which plain mean pooling cannot do.
    """

    def __init__(self, channels: int, attention_dim: int = 128) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels * 3, attention_dim, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(attention_dim),
            nn.Tanh(),
            nn.Conv1d(attention_dim, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.size(2)
        mean = x.mean(dim=2, keepdim=True).expand(-1, -1, t)
        std = x.std(dim=2, keepdim=True).expand(-1, -1, t)

        alpha = torch.softmax(self.attention(torch.cat([x, mean, std], dim=1)), dim=2)
        weighted_mean = (alpha * x).sum(dim=2)
        weighted_std = ((alpha * x.pow(2)).sum(dim=2) - weighted_mean.pow(2)).clamp(min=1e-6).sqrt()
        return torch.cat([weighted_mean, weighted_std], dim=1)


class SpeakerEncoder(nn.Module):
    """ECAPA-TDNN style encoder producing L2-normalized speaker embeddings."""

    def __init__(
        self,
        channels: int = 256,
        embedding_dim: int = 192,
        n_mels: int = N_MELS,
        include_frontend: bool = True,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.frontend = MelFrontend(n_mels=n_mels) if include_frontend else None
        self.spec_augment = SpecAugment()

        self.conv1 = nn.Conv1d(n_mels, channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(channels)

        self.block1 = SERes2Block(channels, kernel_size=3, dilation=2)
        self.block2 = SERes2Block(channels, kernel_size=3, dilation=3)
        self.block3 = SERes2Block(channels, kernel_size=3, dilation=4)

        # Multi-layer feature aggregation: concatenating all block outputs lets
        # the pooling layer see shallow and deep representations together.
        self.mfa = nn.Conv1d(channels * 3, channels * 3, kernel_size=1)
        self.pooling = AttentiveStatsPooling(channels * 3)
        self.bn_pool = nn.BatchNorm1d(channels * 6)
        self.fc = nn.Linear(channels * 6, embedding_dim)
        self.bn_embed = nn.BatchNorm1d(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw waveform ``(batch, samples)`` to L2-normalized ``(batch, embedding_dim)``.

        When built without a front-end, ``x`` is expected to already be log-mel
        of shape ``(batch, n_mels, frames)``.
        """
        if self.frontend is not None:
            x = self.frontend(x)
        x = self.spec_augment(x)

        x = torch.relu(self.bn1(self.conv1(x)))
        h1 = self.block1(x)
        h2 = self.block2(h1)
        h3 = self.block3(h2)

        x = torch.relu(self.mfa(torch.cat([h1, h2, h3], dim=1)))
        x = self.bn_pool(self.pooling(x))
        x = self.bn_embed(self.fc(x))

        # Normalizing here makes cosine similarity a plain dot product, both in
        # the gallery and in the browser.
        return F.normalize(x, p=2, dim=1)


class AAMSoftmax(nn.Module):
    """Additive angular margin (ArcFace) classification head, training only.

    Plain softmax only has to make classes separable, which leaves same-speaker
    clips loosely clustered. The angular margin forces a gap between classes on
    the unit sphere, so the embedding is tight enough that a distance threshold
    can reject a speaker the model has never seen. That property is what open-set
    enrollment depends on, and it is why the margin is worth the extra complexity.
    """

    def __init__(self, embedding_dim: int, n_classes: int, margin: float = 0.2, scale: float = 30.0) -> None:
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(n_classes, embedding_dim))
        nn.init.xavier_normal_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight)).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cosine)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        return self.scale * torch.cos(theta + self.margin * one_hot)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Front-end constants that older checkpoints stored before they were made
# non-persistent. They are recomputed in __init__, so finding them in a
# checkpoint is expected and harmless.
_FRONTEND_CONSTANTS = {"frontend.dft_cos", "frontend.dft_sin", "frontend.mel_fb"}


def load_encoder_state(model: SpeakerEncoder, state_dict: dict) -> None:
    """Load weights, tolerating only the front-end constants.

    Deliberately not a bare ``strict=False``: that would also swallow a genuine
    architecture mismatch and leave randomly-initialized layers in a model that
    still appears to load, which is close to impossible to notice from the
    outside. Anything unexpected other than the known constants raises.
    """
    result = model.load_state_dict(state_dict, strict=False)

    unexpected = set(result.unexpected_keys) - _FRONTEND_CONSTANTS
    if unexpected or result.missing_keys:
        raise RuntimeError(
            "Checkpoint does not match this architecture. "
            f"Missing: {sorted(result.missing_keys)}. Unexpected: {sorted(unexpected)}."
        )
