"""Rebuild speaker features from raw WAV archive for training/inference consistency.

This script walks speaker folders under the archive dataset, extracts fixed-size MFCC
vectors, and writes:
- X_features.npy  -> shape [n_samples, target_dim]
- y_labels.npy    -> shape [n_samples], canonical speaker names like Speaker_0006
"""

import argparse
import json
import os
import re
from pathlib import Path

import librosa
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Extract fixed-size MFCC features from archive WAV files")
    parser.add_argument(
        "--archive-dir",
        default=os.path.join("..", "archive", "50_speakers_audio_data"),
        help="Path to raw speaker WAV archive directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("..", "Extracted Data"),
        help="Directory to write X_features.npy and y_labels.npy.",
    )
    parser.add_argument(
        "--n-mfcc",
        type=int,
        default=40,
        help="Number of MFCC coefficients per frame.",
    )
    parser.add_argument(
        "--target-dim",
        type=int,
        default=4000,
        help="Final flattened feature length per sample.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate used when loading WAVs.",
    )
    parser.add_argument(
        "--save-manifest",
        action="store_true",
        help="Also save extraction_manifest.json with summary info.",
    )
    return parser.parse_args()


def canonical_speaker_name(folder_name):
    """Normalize speaker folder naming variants to Speaker_XXXX."""
    m = re.search(r"(\d+)", folder_name)
    if not m:
        raise ValueError(f"Could not parse numeric speaker id from folder name: {folder_name}")
    return f"Speaker_{int(m.group(1)):04d}"


def extract_fixed_mfcc(audio_path, n_mfcc, target_dim, sample_rate):
    """Match the deployed inference MFCC strategy for high-dimensional vectors."""
    y, sr = librosa.load(audio_path, sr=sample_rate)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    target_frames = max(1, int(np.ceil(target_dim / n_mfcc)))
    if mfcc.shape[1] < target_frames:
        pad_width = target_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :target_frames]

    flat = mfcc.flatten().astype(np.float32)
    if flat.shape[0] < target_dim:
        flat = np.pad(flat, (0, target_dim - flat.shape[0]), mode="constant")
    elif flat.shape[0] > target_dim:
        flat = flat[:target_dim]

    return flat


def iter_wav_files(archive_dir):
    archive = Path(archive_dir)
    if not archive.exists():
        raise FileNotFoundError(f"Archive directory not found: {archive_dir}")

    speaker_dirs = sorted([p for p in archive.iterdir() if p.is_dir()], key=lambda p: p.name)
    for speaker_dir in speaker_dirs:
        speaker_name = canonical_speaker_name(speaker_dir.name)
        wav_files = sorted(speaker_dir.glob("*.wav"), key=lambda p: p.name)
        for wav_path in wav_files:
            yield speaker_name, wav_path


def main():
    args = parse_args()

    features = []
    labels = []
    failures = []

    for idx, (speaker_name, wav_path) in enumerate(iter_wav_files(args.archive_dir), start=1):
        try:
            vec = extract_fixed_mfcc(
                str(wav_path),
                n_mfcc=args.n_mfcc,
                target_dim=args.target_dim,
                sample_rate=args.sample_rate,
            )
            features.append(vec)
            labels.append(speaker_name)

            if idx % 250 == 0:
                print(f"Processed {idx} files...")
        except Exception as exc:
            failures.append({"file": str(wav_path), "error": str(exc)})

    if not features:
        raise RuntimeError("No features were extracted. Check archive path and WAV files.")

    X = np.vstack(features).astype(np.float32)
    y = np.asarray(labels)

    os.makedirs(args.output_dir, exist_ok=True)
    x_path = os.path.join(args.output_dir, "X_features.npy")
    y_path = os.path.join(args.output_dir, "y_labels.npy")

    np.save(x_path, X)
    np.save(y_path, y)

    unique_speakers = sorted(set(labels))

    print("\nExtraction complete")
    print(f"- X_features: {x_path} | shape={X.shape} | dtype={X.dtype}")
    print(f"- y_labels:   {y_path} | shape={y.shape} | unique_speakers={len(unique_speakers)}")
    if failures:
        print(f"- failures:   {len(failures)} (see manifest if saved)")

    if args.save_manifest:
        manifest_path = os.path.join(args.output_dir, "extraction_manifest.json")
        manifest = {
            "archive_dir": os.path.abspath(args.archive_dir),
            "output_dir": os.path.abspath(args.output_dir),
            "sample_count": int(X.shape[0]),
            "feature_dim": int(X.shape[1]),
            "speaker_count": int(len(unique_speakers)),
            "speakers": unique_speakers,
            "failures": failures,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"- manifest:   {manifest_path}")


if __name__ == "__main__":
    main()
