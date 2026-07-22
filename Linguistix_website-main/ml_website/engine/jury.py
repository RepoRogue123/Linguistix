"""Model jury: run every trained model on one clip and surface their disagreement.

A dozen models were trained across several notebooks. Here they all vote on the
same clip at once.

Disagreement is the point. Two clips can both come back "Speaker_0031, 92%" while
one had every model agreeing and the other had a 3-2 split, and those are not
equally trustworthy answers. A unanimous verdict on a noisy clip says something
that a single model's confidence score cannot.

The classical voters run on the 4000-dim MFCC vector; the encoder runs on raw
audio and searches the gallery. They are deliberately different kinds of model,
which is why comparing them is interesting.
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import numpy as np

from .features import extract_legacy
from .service import ModelNotAvailable

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

_bundle = None
_bundle_error: str | None = None


def _load_bundle():
    """Load the persisted classical models once per process."""
    global _bundle, _bundle_error

    if _bundle is not None or _bundle_error is not None:
        return _bundle

    path = MODELS_DIR / "classical_models.joblib"
    if not path.exists():
        _bundle_error = (
            "classical_models.joblib not found. "
            "Run training/train_classical.py --save-models pca to build it."
        )
        return None

    try:
        import joblib

        _bundle = joblib.load(path)
    except Exception as exc:
        _bundle_error = f"Could not load classical models: {exc}"
    return _bundle


def _predict_one(model, features_2d: np.ndarray, classes: list[str]) -> dict:
    """Top-1 prediction with a confidence, however this model expresses one."""
    predicted = int(model.predict(features_2d)[0])
    confidence = None

    if hasattr(model, "predict_proba"):
        try:
            confidence = round(float(model.predict_proba(features_2d)[0].max()) * 100, 2)
        except Exception:
            confidence = None
    elif hasattr(model, "decision_function"):
        # SVC without probability=True has no calibrated probability. Squash the
        # decision margin into 0-100 for display, and mark it so the UI does not
        # present it as if it were the same quantity as a real probability.
        try:
            scores = np.atleast_2d(model.decision_function(features_2d))[0]
            gap = float(np.sort(scores)[-1] - np.sort(scores)[-2]) if scores.size > 1 else 0.0
            confidence = round(float(100 / (1 + np.exp(-gap))), 2)
        except Exception:
            confidence = None

    return {
        "speaker": classes[predicted] if 0 <= predicted < len(classes) else str(predicted),
        "confidence": confidence,
        "calibrated": hasattr(model, "predict_proba"),
    }


def run_jury(service, audio_bytes: bytes) -> dict:
    """Collect one vote per available model, plus a consensus summary."""
    votes: list[dict] = []
    unavailable: list[dict] = []

    # Voter 1: the open-set encoder searching the gallery.
    try:
        started = time.perf_counter()
        embedding, audio_meta = service.embed(audio_bytes)
        result = service.gallery.identify(embedding, top_k=3)
        votes.append(
            {
                "model": "Speaker encoder (TDNN + AAM-Softmax)",
                "key": "encoder",
                "family": "open-set",
                "speaker": result.get("closest"),
                "accepted": result.get("matched"),
                "confidence": round((result.get("score", 0) + 1) / 2 * 100, 2),
                "calibrated": False,
                "note": "cosine similarity against the gallery, not a class probability",
                "ms": round((time.perf_counter() - started) * 1000, 1),
            }
        )
    except ModelNotAvailable as exc:
        raise
    except Exception as exc:
        unavailable.append({"model": "Speaker encoder", "reason": str(exc)})
        audio_meta = {}

    # Voter 2: the dense MFCC network, closed set.
    try:
        started = time.perf_counter()
        legacy = service.legacy_predict(audio_bytes, top_k=1)
        top = legacy["predictions"][0]
        votes.append(
            {
                "model": "MFCC Neural Net",
                "key": "mfcc_net",
                "family": "closed-set",
                "speaker": top["speaker"],
                "confidence": top["confidence"],
                "calibrated": True,
                "ms": round((time.perf_counter() - started) * 1000, 1),
            }
        )
    except Exception as exc:
        unavailable.append({"model": "MFCC Neural Net", "reason": str(exc)})

    # Voters 3+: the classical lineup.
    bundle = _load_bundle()
    if bundle is None:
        unavailable.append({"model": "Classical models", "reason": _bundle_error})
    else:
        try:
            features = extract_legacy(audio_bytes, target_dim=bundle["feature_dim"])
            scaled = bundle["scaler"].transform(features.reshape(1, -1))
            reduced = scaled if bundle["reducer"] is None else bundle["reducer"].transform(scaled)

            for key, entry in bundle["models"].items():
                started = time.perf_counter()
                try:
                    prediction = _predict_one(entry["model"], reduced, bundle["classes"])
                except Exception as exc:
                    unavailable.append({"model": entry["name"], "reason": str(exc)})
                    continue
                votes.append(
                    {
                        "model": f"{entry['name']} + {bundle['representation'].upper()}",
                        "key": key,
                        "family": "closed-set",
                        "ms": round((time.perf_counter() - started) * 1000, 1),
                        **prediction,
                    }
                )
        except Exception as exc:
            unavailable.append({"model": "Classical models", "reason": str(exc)})

    named = [v["speaker"] for v in votes if v.get("speaker")]
    tally = Counter(named)
    winner, winner_count = (tally.most_common(1)[0] if tally else (None, 0))

    return {
        "votes": sorted(votes, key=lambda v: (v.get("confidence") or 0), reverse=True),
        "unavailable": unavailable,
        "consensus": {
            "speaker": winner,
            "votes_for": winner_count,
            "total_voters": len(named),
            "agreement": round(winner_count / len(named) * 100, 1) if named else 0.0,
            "unanimous": bool(named) and winner_count == len(named),
            "distinct_answers": len(tally),
            "tally": dict(tally.most_common()),
        },
        "audio": audio_meta,
        "note": (
            "Closed-set voters must name one of the 50 dataset speakers even when the "
            "true speaker is not among them, so on an unfamiliar voice their agreement "
            "means less than the encoder's decision to reject."
        ),
    }
