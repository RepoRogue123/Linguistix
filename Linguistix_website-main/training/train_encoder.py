"""Train the open-set speaker encoder with AAM-Softmax.

Trains on the ``open`` split's speakers only. The held-out speakers are never
seen during training, so the EER reported against them measures what actually
matters here: whether a voice the network was never trained on can still be
enrolled and recognized.

Audio is read as short random crops via soundfile's frame offsets rather than
decoding whole files. The clips average 59 seconds and total 41 hours, so
decoding a full file to use three seconds of it would make the loader, not the
GPU, the bottleneck.

Usage
-----
    python training/train_encoder.py
    python training/train_encoder.py --epochs 60 --batch-size 96 --crop-seconds 3.0
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import _bootstrap  # noqa: F401  (puts ml_website on sys.path)
from ml_website.engine.encoder import AAMSoftmax, SpeakerEncoder, count_parameters
from ml_website.engine.features import SAMPLE_RATE

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(__file__).resolve().parents[1] / "ml_website" / "models"
MANIFEST = MODELS_DIR / "manifest.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--crop-seconds", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=2e-5)
    p.add_argument("--channels", type=int, default=256)
    p.add_argument("--embedding-dim", type=int, default=192)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--scale", type=float, default=30.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps-per-epoch", type=int, default=200, help="Sampled crops define epoch length, not file count.")
    p.add_argument("--out", type=Path, default=MODELS_DIR / "encoder.pt")
    return p.parse_args()


class CropDataset(Dataset):
    """Random fixed-length crops, one per __getitem__, drawn from the clip pool.

    Sampling is speaker-balanced rather than clip-balanced. The dataset is badly
    imbalanced (10 clips for the smallest speaker, 120 for the largest) and
    sampling by clip would let the largest speakers dominate the angular margin
    and push the small ones into a corner of the sphere.
    """

    def __init__(
        self,
        clips: list[dict],
        clip_ids: list[int],
        speakers: list[str],
        crop_samples: int,
        length: int,
        augment: bool = True,
        seed: int = 42,
    ) -> None:
        self.clips = clips
        self.crop_samples = crop_samples
        self.length = length
        self.augment = augment
        self.speaker_to_index = {s: i for i, s in enumerate(speakers)}

        self.by_speaker: dict[str, list[int]] = {}
        for cid in clip_ids:
            self.by_speaker.setdefault(clips[cid]["speaker"], []).append(cid)
        self.speakers = sorted(self.by_speaker)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.length

    def _read_crop(self, clip: dict) -> np.ndarray:
        path = REPO_ROOT / clip["path"]
        source_rate = clip["source_rate"]

        # Read at the file's native rate, then resample the crop. Reading a
        # slightly longer window first means resampling cannot leave us short.
        need = int(math.ceil(self.crop_samples * source_rate / SAMPLE_RATE)) + 64
        total = int(clip["duration"] * source_rate)

        start = self.rng.randint(0, max(0, total - need)) if total > need else 0
        audio, _ = sf.read(str(path), start=start, frames=need, dtype="float32", always_2d=False)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if source_rate != SAMPLE_RATE:
            import soxr

            audio = soxr.resample(audio, source_rate, SAMPLE_RATE)

        if len(audio) < self.crop_samples:
            audio = np.pad(audio, (0, self.crop_samples - len(audio)))
        return audio[: self.crop_samples].astype(np.float32)

    def _augment(self, audio: np.ndarray) -> np.ndarray:
        # Additive noise across a wide SNR range. Enrollment happens on whatever
        # microphone the visitor has, so the encoder should not assume a clean one.
        if self.rng.random() < 0.6:
            snr_db = self.rng.uniform(5.0, 25.0)
            signal_power = float(np.mean(audio**2)) + 1e-10
            noise_power = signal_power / (10 ** (snr_db / 10))
            audio = audio + np.random.randn(len(audio)).astype(np.float32) * math.sqrt(noise_power)

        # Random gain, so loudness never becomes a speaker cue.
        if self.rng.random() < 0.5:
            audio = audio * self.rng.uniform(0.5, 1.5)

        return np.clip(audio, -1.0, 1.0).astype(np.float32)

    def __getitem__(self, _: int):
        speaker = self.rng.choice(self.speakers)
        clip = self.clips[self.rng.choice(self.by_speaker[speaker])]

        try:
            audio = self._read_crop(clip)
        except Exception:
            audio = np.zeros(self.crop_samples, dtype=np.float32)

        if self.augment:
            audio = self._augment(audio)

        return torch.from_numpy(audio), self.speaker_to_index[speaker]


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Equal error rate and its threshold, from same/different-speaker trial scores."""
    order = np.argsort(scores)[::-1]
    labels = labels[order]
    scores_sorted = scores[order]

    n_target = max(1, int(labels.sum()))
    n_nontarget = max(1, int((1 - labels).sum()))

    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)

    fnr = 1.0 - tp / n_target      # missed same-speaker pairs
    fpr = fp / n_nontarget         # accepted different-speaker pairs

    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fnr[idx] + fpr[idx]) / 2 * 100), float(scores_sorted[idx])


def min_dcf(scores: np.ndarray, labels: np.ndarray, p_target: float = 0.01,
            c_miss: float = 1.0, c_fa: float = 1.0) -> float:
    """Minimum detection cost, the metric speaker verification is normally judged on.

    EER weights both error types equally. minDCF assumes genuine trials are rare
    (p_target=0.01), which is closer to how identification is actually used and
    punishes false accepts much harder.
    """
    order = np.argsort(scores)[::-1]
    labels_sorted = labels[order]

    n_target = max(1, int(labels.sum()))
    n_nontarget = max(1, int((1 - labels).sum()))

    fnr = 1.0 - np.cumsum(labels_sorted) / n_target
    fpr = np.cumsum(1 - labels_sorted) / n_nontarget

    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    return float(dcf.min() / min(c_miss * p_target, c_fa * (1 - p_target)))


@torch.no_grad()
def evaluate_eer(model, clips, clip_ids, crop_samples, device, n_trials=4000, seed=0):
    """Score same/different-speaker pairs among held-out speakers."""
    model.eval()
    rng = random.Random(seed)

    by_speaker: dict[str, list[int]] = {}
    for cid in clip_ids:
        by_speaker.setdefault(clips[cid]["speaker"], []).append(cid)
    speakers = sorted(by_speaker)

    # One embedding per clip, then pairs drawn from those.
    dataset = CropDataset(clips, clip_ids, speakers, crop_samples, len(clip_ids), augment=False, seed=seed)
    dataset.rng = random.Random(seed)

    embeddings: dict[int, np.ndarray] = {}
    batch, ids = [], []
    for cid in clip_ids:
        try:
            audio = dataset._read_crop(clips[cid])
        except Exception:
            continue
        batch.append(torch.from_numpy(audio))
        ids.append(cid)
        if len(batch) == 64:
            out = model(torch.stack(batch).to(device)).cpu().numpy()
            embeddings.update(dict(zip(ids, out)))
            batch, ids = [], []
    if batch:
        out = model(torch.stack(batch).to(device)).cpu().numpy()
        embeddings.update(dict(zip(ids, out)))

    valid = {s: [c for c in v if c in embeddings] for s, v in by_speaker.items()}
    valid = {s: v for s, v in valid.items() if len(v) >= 2}
    if len(valid) < 2:
        return float("nan"), float("nan"), float("nan")

    speakers = sorted(valid)
    scores, labels = [], []

    for _ in range(n_trials // 2):
        s = rng.choice(speakers)
        a, b = rng.sample(valid[s], 2)
        scores.append(float(embeddings[a] @ embeddings[b]))
        labels.append(1)

        s1, s2 = rng.sample(speakers, 2)
        a = rng.choice(valid[s1])
        b = rng.choice(valid[s2])
        scores.append(float(embeddings[a] @ embeddings[b]))
        labels.append(0)

    scores = np.array(scores)
    labels = np.array(labels)
    eer, threshold = compute_eer(scores, labels)
    return eer, threshold, min_dcf(scores, labels)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    clips = manifest["clips"]
    split = manifest["splits"]["open"]

    train_speakers = split["train_speakers"]
    crop_samples = int(args.crop_seconds * SAMPLE_RATE)

    print(f"Device: {device}  ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")
    print(f"Train: {len(train_speakers)} speakers / {len(split['train'])} clips")
    print(f"Held out: {len(split['heldout_speakers'])} speakers / {len(split['heldout'])} clips")
    print(f"Crop: {args.crop_seconds}s ({crop_samples} samples)\n")

    dataset = CropDataset(
        clips, split["train"], train_speakers, crop_samples,
        length=args.steps_per_epoch * args.batch_size, seed=args.seed,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=(device.type == "cuda"), drop_last=True, persistent_workers=args.workers > 0,
    )

    model = SpeakerEncoder(channels=args.channels, embedding_dim=args.embedding_dim).to(device)
    head = AAMSoftmax(args.embedding_dim, len(train_speakers), args.margin, args.scale).to(device)
    print(f"Encoder: {count_parameters(model)/1e6:.2f}M params, {args.embedding_dim}-d embeddings\n")

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * args.steps_per_epoch, pct_start=0.15
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    history: list[dict] = []
    best_eer = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        total_loss = correct = seen = 0

        for audio, labels in loader:
            audio = audio.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = head(model(audio), labels)
                loss = F.cross_entropy(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.size(0)

        train_loss = total_loss / seen
        train_acc = correct / seen * 100

        eer, threshold, dcf = evaluate_eer(model, clips, split["heldout"], crop_samples, device, seed=epoch)
        elapsed = time.perf_counter() - t0

        history.append({
            "epoch": epoch, "train_loss": round(train_loss, 4), "train_acc": round(train_acc, 2),
            "heldout_eer": round(eer, 3), "heldout_mindcf": round(dcf, 4),
            "lr": round(scheduler.get_last_lr()[0], 6), "seconds": round(elapsed, 1),
        })
        print(
            f"epoch {epoch:3d}/{args.epochs}  loss {train_loss:6.3f}  train_acc {train_acc:5.1f}%  "
            f"heldout_EER {eer:5.2f}%  minDCF {dcf:.3f}  ({elapsed:.0f}s)"
        )

        if eer < best_eer:
            best_eer = eer
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": {
                        "channels": args.channels, "embedding_dim": args.embedding_dim,
                        "crop_seconds": args.crop_seconds, "sample_rate": SAMPLE_RATE,
                    },
                    "train_speakers": train_speakers,
                    "heldout_speakers": split["heldout_speakers"],
                    "eer": eer, "min_dcf": dcf, "threshold": threshold, "epoch": epoch,
                },
                args.out,
            )
            print(f"          saved (best EER {eer:.2f}%, cosine threshold {threshold:.3f})")

    (MODELS_DIR / "encoder_history.json").write_text(
        json.dumps(
            {
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "config": vars(args) | {"out": str(args.out)},
                "train_speakers": train_speakers,
                "heldout_speakers": split["heldout_speakers"],
                "best_eer": best_eer,
                "history": history,
            },
            indent=2, default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nBest held-out EER: {best_eer:.2f}%  ->  {args.out}")


if __name__ == "__main__":
    main()
