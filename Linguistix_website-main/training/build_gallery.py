"""Embed every clip once, then build the reference gallery and the speaker map.

Produces three artifacts the server and UI depend on:

``embeddings.npz``        one 192-d embedding per clip, plus labels and split tags.
``gallery_reference.npz`` per-speaker centroids, the read-only half of the gallery.
``speaker_map.json``      a 3D projection of every clip for the map view.

Each clip is embedded from several crops averaged together rather than one. A
single three-second window can land on a pause or a cough; averaging across the
clip gives a centroid that reflects the speaker instead of the moment.

The map is projected straight from these embeddings, so it shows the space
identification actually runs in rather than a separate view of the data.

Usage
-----
    python training/build_gallery.py
    python training/build_gallery.py --crops 6 --projection tsne
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import _bootstrap  # noqa: F401
from ml_website.engine.features import SAMPLE_RATE

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(__file__).resolve().parents[1] / "ml_website" / "models"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=MODELS_DIR / "encoder.pt")
    p.add_argument("--manifest", type=Path, default=MODELS_DIR / "manifest.json")
    p.add_argument("--crops", type=int, default=5, help="Crops averaged per clip.")
    p.add_argument("--crop-seconds", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--projection", choices=["umap", "tsne", "pca"], default="umap")
    p.add_argument("--dims", type=int, default=3, choices=[2, 3], help="Projection dimensionality for the map.")
    return p.parse_args()


def read_crops(clip: dict, n_crops: int, crop_samples: int) -> list[np.ndarray]:
    """Evenly spaced crops spanning the clip, resampled to 16 kHz."""
    path = REPO_ROOT / clip["path"]
    rate = clip["source_rate"]
    need = int(math.ceil(crop_samples * rate / SAMPLE_RATE)) + 64
    total = int(clip["duration"] * rate)

    if total <= need:
        starts = [0]
    else:
        starts = np.linspace(0, total - need, num=n_crops, dtype=int).tolist()

    crops = []
    for start in starts:
        try:
            audio, _ = sf.read(str(path), start=int(start), frames=need, dtype="float32", always_2d=False)
        except Exception:
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if rate != SAMPLE_RATE:
            import soxr

            audio = soxr.resample(audio, rate, SAMPLE_RATE)
        if len(audio) < crop_samples:
            audio = np.pad(audio, (0, crop_samples - len(audio)))
        crops.append(audio[:crop_samples].astype(np.float32))

    return crops or [np.zeros(crop_samples, dtype=np.float32)]


@torch.no_grad()
def embed_all(model, clips, n_crops, crop_samples, device, batch_size):
    """One averaged, renormalized embedding per clip."""
    embeddings = np.zeros((len(clips), model.embedding_dim), dtype=np.float32)
    pending: list[np.ndarray] = []
    owners: list[int] = []

    def flush() -> None:
        if not pending:
            return
        batch = torch.from_numpy(np.stack(pending)).to(device)
        out = model(batch).cpu().numpy()
        for owner, vector in zip(owners, out):
            embeddings[owner] += vector
        pending.clear()
        owners.clear()

    for index, clip in enumerate(clips):
        for crop in read_crops(clip, n_crops, crop_samples):
            pending.append(crop)
            owners.append(index)
            if len(pending) >= batch_size:
                flush()
        if (index + 1) % 250 == 0:
            print(f"  embedded {index + 1}/{len(clips)} clips")
    flush()

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-8)


def project(embeddings: np.ndarray, method: str, dims: int = 3, seed: int = 42) -> tuple[np.ndarray, str]:
    """Reduce embeddings for the map, falling back if a library is absent."""
    if method == "umap":
        try:
            import umap

            reducer = umap.UMAP(
                n_components=dims, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=seed
            )
            return reducer.fit_transform(embeddings), "umap"
        except ImportError:
            print("  umap-learn not installed, falling back to t-SNE")
            method = "tsne"

    if method == "tsne":
        from sklearn.manifold import TSNE

        # Barnes-Hut only supports 2 output dimensions; 3D needs the exact solver.
        reducer = TSNE(
            n_components=dims,
            metric="cosine",
            init="pca",
            perplexity=30,
            random_state=seed,
            method="exact" if dims > 2 else "barnes_hut",
        )
        return reducer.fit_transform(embeddings), "tsne"

    from sklearn.decomposition import PCA

    return PCA(n_components=dims, random_state=seed).fit_transform(embeddings), "pca"


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(f"No checkpoint at {args.checkpoint}. Run training/train_encoder.py first.")

    from ml_website.engine.encoder import SpeakerEncoder, load_encoder_state

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    model = SpeakerEncoder(
        channels=config.get("channels", 256), embedding_dim=config.get("embedding_dim", 192)
    ).to(device)
    load_encoder_state(model, checkpoint["state_dict"])
    model.eval()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    clips = manifest["clips"]
    crop_samples = int(args.crop_seconds * SAMPLE_RATE)

    print(f"Embedding {len(clips)} clips x {args.crops} crops on {device} ...")
    embeddings = embed_all(model, clips, args.crops, crop_samples, device, args.batch_size)

    labels = np.array([c["speaker"] for c in clips])
    heldout = set(manifest["splits"]["open"]["heldout_speakers"])
    seen = np.array(["heldout" if s in heldout else "train" for s in labels])

    np.savez_compressed(
        MODELS_DIR / "embeddings.npz",
        embeddings=embeddings, labels=labels, seen=seen,
        clip_ids=np.array([c["id"] for c in clips]),
    )
    print(f"Wrote {MODELS_DIR / 'embeddings.npz'}  {embeddings.shape}")

    # Centroids from training clips only. Building them from every clip would
    # fold the closed-set test data into the thing being evaluated.
    train_ids = set(manifest["splits"]["closed"]["train"])
    names, centroids, counts = [], [], []
    for speaker in manifest["speakers"]:
        rows = [i for i, c in enumerate(clips) if c["speaker"] == speaker and c["id"] in train_ids]
        if not rows:
            rows = [i for i, c in enumerate(clips) if c["speaker"] == speaker]
        vector = embeddings[rows].mean(axis=0)
        names.append(speaker)
        centroids.append(vector / max(float(np.linalg.norm(vector)), 1e-8))
        counts.append(len(rows))

    np.savez_compressed(
        MODELS_DIR / "gallery_reference.npz",
        names=np.array(names), centroids=np.stack(centroids).astype(np.float32), counts=np.array(counts),
    )
    print(f"Wrote {MODELS_DIR / 'gallery_reference.npz'}  {len(names)} speakers")

    print(f"Projecting to {args.dims}D via {args.projection} ...")
    coords, used = project(embeddings, args.projection, dims=args.dims)

    # Normalize into [-1, 1] so the client can scale to any canvas size.
    coords = np.asarray(coords, dtype=np.float32)
    span = np.maximum(coords.max(axis=0) - coords.min(axis=0), 1e-6)
    coords = (coords - coords.min(axis=0)) / span * 2 - 1

    speaker_index = {s: i for i, s in enumerate(manifest["speakers"])}
    (MODELS_DIR / "speaker_map.json").write_text(
        json.dumps(
            {
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "projection": used,
                "dims": int(args.dims),
                "source": "speaker encoder embeddings",
                "note": (
                    "Projected from the encoder's own embedding space, which is where "
                    "identification happens."
                ),
                "speakers": manifest["speakers"],
                "heldout_speakers": sorted(heldout),
                "points": [
                    {
                        "x": round(float(coords[i, 0]), 4),
                        "y": round(float(coords[i, 1]), 4),
                        **({"z": round(float(coords[i, 2]), 4)} if args.dims > 2 else {}),
                        "s": speaker_index[clips[i]["speaker"]],
                        "h": 1 if clips[i]["speaker"] in heldout else 0,
                    }
                    for i in range(len(clips))
                ],
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    print(f"Wrote {MODELS_DIR / 'speaker_map.json'}  {len(clips)} points in {args.dims}D")

    # Quick sanity signal: same-speaker similarity should clearly exceed
    # different-speaker similarity, or the gallery will not discriminate.
    matrix = np.stack(centroids)
    similarity = matrix @ matrix.T
    off_diagonal = similarity[~np.eye(len(names), dtype=bool)]
    print(f"\nCentroid separation: mean off-diagonal cosine {off_diagonal.mean():.3f}, max {off_diagonal.max():.3f}")
    worst = np.unravel_index(np.argmax(similarity - np.eye(len(names)) * 2), similarity.shape)
    print(f"Most confusable pair: {names[worst[0]]} / {names[worst[1]]} at {similarity[worst]:.3f}")


if __name__ == "__main__":
    main()
