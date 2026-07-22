"""Build the dataset manifest and the two split regimes every experiment reads from.

Run once. Everything downstream (classical models, encoder, evaluation) loads
``manifest.json`` rather than re-deriving splits, so no two experiments can
accidentally disagree about what "test" means.

Two regimes, because the project asks two different questions:

closed
    All 50 speakers, stratified per speaker into train/val/test. Answers "which
    of these known speakers is this?" and is what the classical models and the
    legacy ANN are scored on.

open
    Speakers themselves are split. The encoder trains on a subset and never sees
    the held-out speakers at all; those are used only to measure verification EER.
    Answers "can this generalize to a voice it was never trained on?", which is
    the question that matters once anyone can enroll.

Usage
-----
    python training/prepare_data.py
    python training/prepare_data.py --heldout-speakers 12 --seed 1337
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = REPO_ROOT / "archive" / "50_speakers_audio_data"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "ml_website" / "models" / "manifest.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.2, help="Closed-set validation fraction.")
    p.add_argument("--test-frac", type=float, default=0.1, help="Closed-set test fraction.")
    p.add_argument(
        "--heldout-speakers",
        type=int,
        default=10,
        help="Speakers withheld entirely from encoder training, for open-set evaluation.",
    )
    p.add_argument("--min-clips", type=int, default=4, help="Speakers with fewer clips are reported, not dropped.")
    return p.parse_args()


def canonical_speaker_name(folder_name: str) -> str:
    """Normalize the archive's inconsistent folder names to Speaker_XXXX.

    The archive mixes styles (``Speaker26``, ``Speaker_0026``), so the numeric id
    is the only reliable key. Note the dataset has no Speaker_0022 and numbering
    runs to 0050, so the class index and the numeric suffix diverge past 21.
    """
    match = re.search(r"(\d+)", folder_name)
    if not match:
        raise ValueError(f"No numeric speaker id in folder name: {folder_name}")
    return f"Speaker_{int(match.group(1)):04d}"


def scan_archive(archive_dir: Path) -> list[dict]:
    """Index every WAV with its speaker and duration, reading headers only."""
    if not archive_dir.exists():
        raise FileNotFoundError(f"Archive directory not found: {archive_dir}")

    clips: list[dict] = []
    speaker_dirs = sorted((p for p in archive_dir.iterdir() if p.is_dir()), key=lambda p: p.name)

    for speaker_dir in speaker_dirs:
        speaker = canonical_speaker_name(speaker_dir.name)
        for wav_path in sorted(speaker_dir.glob("*.wav"), key=lambda p: p.name):
            try:
                info = sf.info(str(wav_path))
            except Exception as exc:  # a corrupt file should not abort the scan
                print(f"  skipped {wav_path.name}: {exc}")
                continue

            clips.append(
                {
                    "id": len(clips),
                    "path": str(wav_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "speaker": speaker,
                    "duration": round(info.frames / info.samplerate, 3),
                    "source_rate": info.samplerate,
                    # Recorded because both of these turn out to be perfectly
                    # correlated with speaker identity; see report_confounds().
                    "subtype": info.subtype,
                }
            )

    if not clips:
        raise RuntimeError(f"No WAV files found under {archive_dir}")
    return clips


def stratified_split(
    clips: list[dict], val_frac: float, test_frac: float, rng: np.random.Generator
) -> dict[str, list[int]]:
    """Per-speaker split so every speaker appears in every fold.

    Stratifying matters here because the class imbalance is severe (10 clips for
    the smallest speaker, 120 for the largest). A global random split would leave
    the smallest speakers absent from test entirely and make the number meaningless.
    """
    by_speaker: dict[str, list[int]] = defaultdict(list)
    for clip in clips:
        by_speaker[clip["speaker"]].append(clip["id"])

    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}

    for speaker in sorted(by_speaker):
        ids = np.array(by_speaker[speaker])
        rng.shuffle(ids)
        n = len(ids)

        # Guarantee at least one clip in val and test whenever the speaker has
        # enough to spare, rather than letting rounding starve small speakers.
        n_test = max(1, int(round(n * test_frac))) if n >= 3 else 0
        n_val = max(1, int(round(n * val_frac))) if n >= 3 else 0
        if n_test + n_val >= n:
            n_test = min(1, n - 1)
            n_val = min(1, n - 1 - n_test)

        splits["test"].extend(ids[:n_test].tolist())
        splits["val"].extend(ids[n_test : n_test + n_val].tolist())
        splits["train"].extend(ids[n_test + n_val :].tolist())

    return {k: sorted(v) for k, v in splits.items()}


def speaker_disjoint_split(
    clips: list[dict], n_heldout: int, rng: np.random.Generator
) -> dict[str, list]:
    """Withhold whole speakers so open-set metrics measure real generalization.

    Held-out speakers are chosen from those with enough clips to form both
    enrollment and trial sets; withholding a speaker with 10 clips would give an
    EER estimate too noisy to mean anything.
    """
    counts = Counter(clip["speaker"] for clip in clips)
    eligible = sorted((s for s, c in counts.items() if c >= 20), key=lambda s: (-counts[s], s))

    if len(eligible) < n_heldout:
        raise ValueError(
            f"Only {len(eligible)} speakers have >=20 clips; cannot hold out {n_heldout}."
        )

    # Sample across the frequency range rather than taking the largest speakers,
    # so the held-out set is not systematically easier than the training set.
    picks = rng.choice(len(eligible), size=n_heldout, replace=False)
    heldout = sorted(eligible[i] for i in picks)
    heldout_set = set(heldout)

    train_speakers = sorted(s for s in counts if s not in heldout_set)

    return {
        "train_speakers": train_speakers,
        "heldout_speakers": heldout,
        "train": sorted(c["id"] for c in clips if c["speaker"] not in heldout_set),
        "heldout": sorted(c["id"] for c in clips if c["speaker"] in heldout_set),
    }


def report_confounds(clips: list[dict]) -> dict:
    """Check whether recording metadata alone can identify a speaker.

    It can, in this dataset, and badly. Every speaker was captured in a single
    session, so sample rate and codec are constant within a speaker and vary
    between them. Two consequences:

    - Sample rate splits the 50 speakers into a 24/26 partition with no overlap.
    - Seven speakers are stored entirely as MP3 inside a WAV container while the
      other 43 are entirely PCM, so compression artifacts uniquely tag those seven.

    A model reading raw spectra can score well on this by detecting the recording
    channel rather than the voice, which inflates any accuracy measured on it.
    The mitigations are in engine/features.py: the mel cutoff sits below the
    resampler rolloff, and per-utterance CMVN removes channel offset.
    """
    by_speaker: dict[str, dict[str, set]] = defaultdict(lambda: {"rates": set(), "subtypes": set()})
    for clip in clips:
        by_speaker[clip["speaker"]]["rates"].add(clip["source_rate"])
        by_speaker[clip["speaker"]]["subtypes"].add(clip.get("subtype", "?"))

    mixed_rate = [s for s, v in by_speaker.items() if len(v["rates"]) > 1]
    mixed_codec = [s for s, v in by_speaker.items() if len(v["subtypes"]) > 1]

    rate_groups: dict[int, list[str]] = defaultdict(list)
    codec_groups: dict[str, list[str]] = defaultdict(list)
    for speaker, v in by_speaker.items():
        if len(v["rates"]) == 1:
            rate_groups[next(iter(v["rates"]))].append(speaker)
        if len(v["subtypes"]) == 1:
            codec_groups[next(iter(v["subtypes"]))].append(speaker)

    print("\nRecording-metadata confound check")
    for rate, speakers in sorted(rate_groups.items()):
        print(f"  {len(speakers):2d} speakers recorded entirely at {rate} Hz")
    print(f"  speakers with mixed sample rates: {len(mixed_rate)}")
    for codec, speakers in sorted(codec_groups.items()):
        print(f"  {len(speakers):2d} speakers stored entirely as {codec}")
    print(f"  speakers with mixed codecs: {len(mixed_codec)}")

    if not mixed_rate and len(rate_groups) > 1:
        print("  WARNING: sample rate alone partitions the speakers perfectly.")
    if not mixed_codec and len(codec_groups) > 1:
        print("  WARNING: codec alone uniquely tags a subset of speakers.")
    print("  Mitigated in engine/features.py via the 7 kHz mel cutoff and per-utterance CMVN.")

    return {
        "rate_groups": {str(k): sorted(v) for k, v in rate_groups.items()},
        "codec_groups": {k: sorted(v) for k, v in codec_groups.items()},
        "speakers_with_mixed_rate": sorted(mixed_rate),
        "speakers_with_mixed_codec": sorted(mixed_codec),
        "perfectly_separable_by_rate": not mixed_rate and len(rate_groups) > 1,
        "perfectly_separable_by_codec": not mixed_codec and len(codec_groups) > 1,
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"Scanning {args.archive_dir} ...")
    clips = scan_archive(args.archive_dir)
    counts = Counter(c["speaker"] for c in clips)
    durations = np.array([c["duration"] for c in clips])
    rates = Counter(c["source_rate"] for c in clips)

    print(f"  {len(clips)} clips, {len(counts)} speakers")
    print(f"  duration: min {durations.min():.1f}s  mean {durations.mean():.1f}s  max {durations.max():.1f}s")
    print(f"  total audio: {durations.sum() / 3600:.2f} hours")
    print(f"  source sample rates: {dict(rates)}")

    smallest = min(counts.items(), key=lambda kv: kv[1])
    largest = max(counts.items(), key=lambda kv: kv[1])
    print(f"  clips per speaker: min {smallest[1]} ({smallest[0]})  max {largest[1]} ({largest[0]})")

    thin = sorted(s for s, c in counts.items() if c < args.min_clips)
    if thin:
        print(f"  WARNING: speakers with <{args.min_clips} clips: {thin}")

    closed = stratified_split(clips, args.val_frac, args.test_frac, rng)
    print(f"\nClosed-set split: train {len(closed['train'])} / val {len(closed['val'])} / test {len(closed['test'])}")

    covered = {
        fold: len({clips[i]["speaker"] for i in ids}) for fold, ids in closed.items()
    }
    print(f"  speakers present per fold: {covered}")
    if covered["test"] != len(counts):
        print(f"  NOTE: {len(counts) - covered['test']} speaker(s) too small to appear in test")

    openset = speaker_disjoint_split(clips, args.heldout_speakers, rng)
    print(
        f"\nOpen-set split: {len(openset['train_speakers'])} train speakers "
        f"({len(openset['train'])} clips) / {len(openset['heldout_speakers'])} held out "
        f"({len(openset['heldout'])} clips)"
    )
    print(f"  held out: {', '.join(openset['heldout_speakers'])}")

    confounds = report_confounds(clips)

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "archive_dir": str(args.archive_dir),
        "sample_rate": 16_000,
        "confounds": confounds,
        "speakers": sorted(counts),
        "clips_per_speaker": dict(sorted(counts.items())),
        "clips": clips,
        "splits": {"closed": closed, "open": openset},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
