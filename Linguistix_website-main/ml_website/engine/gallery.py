"""Speaker gallery: enrollment, identification, and verification over embeddings.

This is where open-set recognition actually happens. The old server ran a fixed
50-way softmax, so the set of recognizable people was baked into the weights.
Here a speaker is just a centroid in embedding space, enrolling one is a write,
and identification is a cosine search. Nothing is retrained.

Two populations share the space:

reference
    The 50 dataset speakers, centroids precomputed offline by
    ``training/build_gallery.py``. Read-only, always present.

enrolled
    Anyone added at runtime through the API. Stored in SQLite.

Both are searched together and results say which population a match came from,
because "you sound most like dataset Speaker_0031" and "you are the person who
enrolled as Priya" are different claims and the interface should not blur them.

The deployed Hugging Face Space has an ephemeral filesystem, so runtime
enrollments do not survive a restart. That is a deployment property, not a bug
to fix here, but the API surfaces it so the UI can say so plainly.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Cosine similarity below this is reported as "no confident match". Overwritten
# at load time by the EER threshold measured on held-out speakers, which is a
# calibrated operating point rather than a guess. The old server's 35% confidence
# and 8% margin gate had no such grounding.
DEFAULT_THRESHOLD = 0.45


@dataclass
class Match:
    """One candidate returned by a gallery search."""

    speaker: str
    score: float
    source: str  # "reference" or "enrolled"
    samples: int

    def as_dict(self) -> dict:
        return {
            "speaker": self.speaker,
            "score": round(float(self.score), 4),
            "similarity_pct": round(float((self.score + 1) / 2 * 100), 2),
            "source": self.source,
            "samples": self.samples,
        }


class SpeakerGallery:
    """Cosine-similarity search over reference and enrolled speaker centroids."""

    def __init__(self, db_path: Path, reference_path: Path | None = None, threshold: float | None = None) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.threshold = DEFAULT_THRESHOLD if threshold is None else float(threshold)
        self._lock = threading.Lock()

        self._reference_names: list[str] = []
        self._reference_matrix = np.zeros((0, 0), dtype=np.float32)
        self._reference_counts: dict[str, int] = {}

        self._init_db()
        if reference_path and Path(reference_path).exists():
            self._load_reference(Path(reference_path))

    # ---------------------------------------------------------------- storage

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS enrolled (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    name       TEXT NOT NULL UNIQUE,
                    centroid   BLOB NOT NULL,
                    dim        INTEGER NOT NULL,
                    samples    INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def _load_reference(self, path: Path) -> None:
        """Load precomputed dataset-speaker centroids."""
        payload = np.load(path, allow_pickle=True)
        names = [str(n) for n in payload["names"]]
        matrix = np.asarray(payload["centroids"], dtype=np.float32)
        counts = payload["counts"] if "counts" in payload else np.ones(len(names))

        # Renormalize defensively: averaging unit vectors does not yield a unit
        # vector, and the search treats the dot product as a cosine.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self._reference_matrix = matrix / np.maximum(norms, 1e-8)
        self._reference_names = names
        self._reference_counts = {n: int(c) for n, c in zip(names, counts)}

    # -------------------------------------------------------------- mutation

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        return embedding / max(float(np.linalg.norm(embedding)), 1e-8)

    def enroll(self, name: str, embeddings: np.ndarray, replace: bool = False) -> dict:
        """Add or extend a speaker from one or more embeddings.

        Re-enrolling an existing name merges into a running mean weighted by
        sample count rather than overwriting, so a speaker's centroid tightens as
        they contribute more audio instead of jumping to wherever the last clip landed.
        """
        name = name.strip()
        if not name:
            raise ValueError("Speaker name cannot be empty.")

        embeddings = np.atleast_2d(np.asarray(embeddings, dtype=np.float32))
        if embeddings.size == 0:
            raise ValueError("No embeddings supplied.")

        centroid = self._normalize(embeddings.mean(axis=0))
        n_new = int(embeddings.shape[0])
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")

        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT centroid, dim, samples FROM enrolled WHERE name = ?", (name,)).fetchone()

            if row is None:
                conn.execute(
                    "INSERT INTO enrolled (name, centroid, dim, samples, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                    (name, centroid.tobytes(), int(centroid.shape[0]), n_new, now, now),
                )
                total = n_new
            elif replace:
                conn.execute(
                    "UPDATE enrolled SET centroid=?, dim=?, samples=?, updated_at=? WHERE name=?",
                    (centroid.tobytes(), int(centroid.shape[0]), n_new, now, name),
                )
                total = n_new
            else:
                existing = np.frombuffer(row["centroid"], dtype=np.float32)
                n_old = int(row["samples"])
                merged = self._normalize((existing * n_old + centroid * n_new) / (n_old + n_new))
                total = n_old + n_new
                conn.execute(
                    "UPDATE enrolled SET centroid=?, samples=?, updated_at=? WHERE name=?",
                    (merged.tobytes(), total, now, name),
                )

        return {"name": name, "samples": total, "added": n_new, "updated_at": now}

    def delete(self, name: str) -> bool:
        with self._lock, self._connect() as conn:
            return conn.execute("DELETE FROM enrolled WHERE name = ?", (name,)).rowcount > 0

    def clear(self) -> int:
        with self._lock, self._connect() as conn:
            return conn.execute("DELETE FROM enrolled").rowcount

    # ---------------------------------------------------------------- queries

    def _enrolled_matrix(self) -> tuple[list[str], np.ndarray, dict[str, int]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT name, centroid, samples FROM enrolled ORDER BY name").fetchall()

        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32), {}

        names = [r["name"] for r in rows]
        matrix = np.stack([np.frombuffer(r["centroid"], dtype=np.float32) for r in rows])
        counts = {r["name"]: int(r["samples"]) for r in rows}
        return names, matrix, counts

    def list_enrolled(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name, samples, created_at, updated_at FROM enrolled ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def identify(self, embedding: np.ndarray, top_k: int = 5) -> dict:
        """Rank every known speaker against one embedding.

        Rejection is a threshold on cosine similarity, calibrated from the
        held-out EER operating point. The margin to the runner-up is reported
        too, because a confident-looking top score with a near-tie behind it
        means something quite different from a clear win.
        """
        query = self._normalize(embedding)
        matches: list[Match] = []

        if self._reference_matrix.size:
            for name, score in zip(self._reference_names, self._reference_matrix @ query):
                matches.append(Match(name, float(score), "reference", self._reference_counts.get(name, 0)))

        names, matrix, counts = self._enrolled_matrix()
        if matrix.size:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            for name, score in zip(names, (matrix / np.maximum(norms, 1e-8)) @ query):
                matches.append(Match(name, float(score), "enrolled", counts.get(name, 0)))

        if not matches:
            return {
                "matched": False,
                "reason": "gallery_empty",
                "speaker": None,
                "threshold": self.threshold,
                "candidates": [],
            }

        matches.sort(key=lambda m: m.score, reverse=True)
        best = matches[0]
        runner_up = matches[1].score if len(matches) > 1 else -1.0
        accepted = best.score >= self.threshold

        return {
            "matched": accepted,
            "reason": "accepted" if accepted else "below_threshold",
            "speaker": best.speaker if accepted else None,
            "closest": best.speaker,
            "score": round(best.score, 4),
            "margin": round(float(best.score - runner_up), 4),
            "threshold": self.threshold,
            "source": best.source,
            "candidates": [m.as_dict() for m in matches[:top_k]],
        }

    def verify(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> dict:
        """Score whether two clips are the same speaker."""
        score = float(self._normalize(embedding_a) @ self._normalize(embedding_b))
        return {
            "same_speaker": score >= self.threshold,
            "score": round(score, 4),
            "similarity_pct": round((score + 1) / 2 * 100, 2),
            "threshold": self.threshold,
        }

    def stats(self) -> dict:
        with self._connect() as conn:
            enrolled = conn.execute("SELECT COUNT(*) AS n, COALESCE(SUM(samples),0) AS s FROM enrolled").fetchone()
        return {
            "reference_speakers": len(self._reference_names),
            "enrolled_speakers": int(enrolled["n"]),
            "enrolled_samples": int(enrolled["s"]),
            "threshold": self.threshold,
            "embedding_dim": int(self._reference_matrix.shape[1]) if self._reference_matrix.size else None,
            "persistent": False,
            "storage_note": "Enrollments live on the container filesystem and reset when the Space restarts.",
        }
