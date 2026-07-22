"""JSON endpoints. Every route returns JSON, including on error.

The React build is served separately, so the frontend can be developed against a
running backend without templates in the way.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from flask import Blueprint, jsonify, request

from ..engine.service import ModelNotAvailable, get_service

api = Blueprint("api", __name__, url_prefix="/api")

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

# The browser records WebM/Opus by default, which soundfile cannot read, so the
# frontend encodes to WAV before upload. The wider list here is for people
# dropping their own files in.
ALLOWED_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aiff", ".aif"}
MAX_FILES_PER_ENROLLMENT = 10

# Below this, audio is treated as non-speech and a positive match is downgraded.
# Set low on purpose: the cost of wrongly rejecting a real quiet voice is much
# worse here than the cost of accepting an unusual but genuine recording.
SPEECH_LIKENESS_FLOOR = 0.35


def _error(message: str, status: int = 400, **extra):
    return jsonify({"ok": False, "error": message, **extra}), status


def _read_upload(field: str = "audio"):
    """Pull an uploaded file off the request and hand back its bytes."""
    if field not in request.files:
        raise ValueError(f"No file in field '{field}'.")

    upload = request.files[field]
    if not upload.filename:
        raise ValueError("Uploaded file has no name.")

    suffix = Path(upload.filename).suffix.lower()
    if suffix and suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{suffix}'. Accepted: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    data = upload.read()
    if not data:
        raise ValueError("Uploaded file is empty.")
    return data


def _load_json_artifact(name: str):
    path = MODELS_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@api.errorhandler(Exception)
def _unhandled(exc):
    return jsonify({"ok": False, "error": str(exc), "type": type(exc).__name__}), 500


# --------------------------------------------------------------------- status


@api.get("/health")
def health():
    return jsonify({"ok": True, **get_service().health()})


# ----------------------------------------------------------------- identifica


@api.post("/identify")
def identify():
    """Audio in, ranked speaker candidates out.

    Returns the spectrogram alongside the result so the UI can render the
    sonagram strip from exactly the audio the model scored, rather than
    recomputing it client-side and showing something subtly different.
    """
    service = get_service()
    started = time.perf_counter()

    try:
        audio_bytes = _read_upload()
    except ValueError as exc:
        return _error(str(exc))

    try:
        embedding, meta = service.embed(audio_bytes)
    except ModelNotAvailable as exc:
        return _error(str(exc), 503)
    except ValueError as exc:
        return _error(str(exc), 422)

    top_k = min(int(request.args.get("top_k", 5)), 50)
    result = service.gallery.identify(embedding, top_k=top_k)

    # Downgrade a match on audio that does not look like speech. The threshold
    # was calibrated on speech trials only, so a tone or a music clip can score
    # just over it without meaning anything. This only ever weakens a positive
    # result; it never turns a rejection into a match.
    likeness = meta.get("speech_likeness", 1.0)
    if result.get("matched") and likeness < SPEECH_LIKENESS_FLOOR:
        result["matched"] = False
        result["speaker"] = None
        result["reason"] = "not_speech"
        result["speech_likeness"] = likeness
        result["explanation"] = (
            f"This scored {result.get('score')} against {result.get('closest')}, which clears the "
            f"{result.get('threshold')} threshold, but the audio does not look like speech "
            f"(speech-likeness {likeness:.2f}). The threshold is calibrated on voice-against-voice "
            "comparisons, so a match on non-speech input is not meaningful."
        )

    payload = {
        "ok": True,
        "identification": result,
        "audio": meta,
        "embedding": [round(float(v), 6) for v in embedding],
        "total_ms": round((time.perf_counter() - started) * 1000, 1),
    }

    if request.args.get("spectrogram", "1") != "0":
        try:
            payload["spectrogram"] = service.spectrogram(audio_bytes)
        except Exception as exc:
            payload["spectrogram_error"] = str(exc)

    return jsonify(payload)


@api.post("/embed")
def embed():
    """Just the embedding. Used by the map to place a clip without ranking it."""
    try:
        audio_bytes = _read_upload()
    except ValueError as exc:
        return _error(str(exc))

    try:
        embedding, meta = get_service().embed(audio_bytes)
    except ModelNotAvailable as exc:
        return _error(str(exc), 503)
    except ValueError as exc:
        return _error(str(exc), 422)

    return jsonify({"ok": True, "embedding": [round(float(v), 6) for v in embedding], "audio": meta})


@api.post("/match")
def match():
    """Match an embedding the client already computed against the gallery.

    The split that makes live streaming viable on a free CPU tier. The browser
    runs the encoder locally via ONNX, which is the expensive part, and sends
    only the 192 floats here. Gallery search is a single matrix multiply, so
    this stays a few milliseconds even at several calls per second — and keeping
    it server-side means runtime enrolments are matched against too, which a
    fully client-side search could not do.
    """
    payload = request.get_json(silent=True) or {}
    vector = payload.get("embedding")

    if not isinstance(vector, list) or not vector:
        return _error("Body must be JSON with an 'embedding' array.")

    try:
        embedding = np.asarray(vector, dtype=np.float32)
    except (TypeError, ValueError):
        return _error("'embedding' must be an array of numbers.")

    if embedding.ndim != 1:
        return _error("'embedding' must be one-dimensional.")

    service = get_service()
    expected = service.gallery.stats().get("embedding_dim")
    if expected and embedding.shape[0] != expected:
        return _error(f"Expected a {expected}-dimension embedding, got {embedding.shape[0]}.")

    top_k = min(int(request.args.get("top_k", 3)), 50)
    return jsonify({"ok": True, "identification": service.gallery.identify(embedding, top_k=top_k)})


@api.post("/verify")
def verify():
    """Two clips in, a same-speaker decision out."""
    service = get_service()

    try:
        first = _read_upload("audio_a")
        second = _read_upload("audio_b")
    except ValueError as exc:
        return _error(str(exc))

    try:
        embedding_a, meta_a = service.embed(first)
        embedding_b, meta_b = service.embed(second)
    except ModelNotAvailable as exc:
        return _error(str(exc), 503)
    except ValueError as exc:
        return _error(str(exc), 422)

    return jsonify(
        {
            "ok": True,
            "verification": service.gallery.verify(embedding_a, embedding_b),
            "audio_a": meta_a,
            "audio_b": meta_b,
        }
    )


# -------------------------------------------------------------------- gallery


@api.post("/enroll")
def enroll():
    """Add a speaker from one or more clips. No retraining involved."""
    service = get_service()

    name = (request.form.get("name") or "").strip()
    if not name:
        return _error("A speaker name is required.")
    if len(name) > 64:
        return _error("Speaker name must be 64 characters or fewer.")

    uploads = request.files.getlist("audio")
    if not uploads:
        return _error("At least one audio file is required.")
    if len(uploads) > MAX_FILES_PER_ENROLLMENT:
        return _error(f"At most {MAX_FILES_PER_ENROLLMENT} clips per enrollment.")

    embeddings, accepted, rejected = [], [], []
    for upload in uploads:
        data = upload.read()
        if not data:
            rejected.append({"file": upload.filename, "reason": "empty file"})
            continue
        try:
            vector, meta = service.embed(data)
        except ModelNotAvailable as exc:
            return _error(str(exc), 503)
        except Exception as exc:
            rejected.append({"file": upload.filename, "reason": str(exc)})
            continue
        embeddings.append(vector)
        accepted.append({"file": upload.filename, **meta})

    if not embeddings:
        return _error("No usable audio in the upload.", 422, rejected=rejected)

    # Warn when the supplied clips disagree with each other. Enrolling two
    # different people under one name silently poisons the centroid, and the
    # spread across clips is the cheapest way to notice.
    stacked = np.stack(embeddings)
    coherence = None
    if len(stacked) > 1:
        centroid = stacked.mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-8)
        coherence = round(float(np.mean(stacked @ centroid)), 4)

    replace = request.form.get("replace", "").lower() in {"1", "true", "yes"}
    record = service.gallery.enroll(name, stacked, replace=replace)

    response = {"ok": True, "enrolled": record, "accepted": accepted, "rejected": rejected}
    if coherence is not None:
        response["clip_coherence"] = coherence
        if coherence < 0.5:
            response["warning"] = (
                f"The supplied clips only agree at {coherence:.2f} cosine. "
                "They may not all be the same speaker, which would blur this enrollment."
            )
    return jsonify(response)


@api.get("/gallery")
def gallery_list():
    service = get_service()
    return jsonify({"ok": True, "enrolled": service.gallery.list_enrolled(), **service.gallery.stats()})


@api.delete("/gallery/<path:name>")
def gallery_delete(name: str):
    if get_service().gallery.delete(name):
        return jsonify({"ok": True, "deleted": name})
    return _error(f"No enrolled speaker named '{name}'.", 404)


@api.delete("/gallery")
def gallery_clear():
    return jsonify({"ok": True, "deleted": get_service().gallery.clear()})


# ----------------------------------------------------------------- comparison


@api.post("/jury")
def jury():
    """Run several models on one clip and show where they disagree.

    Disagreement is informative rather than embarrassing: clips the models split
    on are the genuinely hard ones, and a unanimous verdict means something
    different from a 3-2 result even at identical top-1 confidence.
    """
    try:
        audio_bytes = _read_upload()
    except ValueError as exc:
        return _error(str(exc))

    from ..engine.jury import run_jury

    try:
        return jsonify({"ok": True, **run_jury(get_service(), audio_bytes)})
    except ModelNotAvailable as exc:
        return _error(str(exc), 503)


@api.post("/explain")
def explain():
    """Saliency over the mel spectrogram: which regions drove the identification."""
    try:
        audio_bytes = _read_upload()
    except ValueError as exc:
        return _error(str(exc))

    from ..engine.explain import explain_embedding

    try:
        return jsonify({"ok": True, **explain_embedding(get_service(), audio_bytes)})
    except ModelNotAvailable as exc:
        return _error(str(exc), 503)
    except ValueError as exc:
        return _error(str(exc), 422)


# ------------------------------------------------------------------ artifacts


@api.get("/map")
def speaker_map():
    """Precomputed 2D projection of every dataset clip."""
    data = _load_json_artifact("speaker_map.json")
    if data is None:
        return _error("speaker_map.json not built. Run training/build_gallery.py.", 503)
    return jsonify({"ok": True, **data})


@api.get("/metrics")
def metrics():
    """Every evaluation artifact the UI charts, in one response."""
    benchmarks = _load_json_artifact("benchmarks.json")

    # The artifact on disk can carry several evaluation regimes; only the one
    # where dimensionality reduction was fitted inside the training fold is a
    # real measurement, so only that one is served.
    if benchmarks and isinstance(benchmarks.get("results"), list):
        benchmarks = {
            **benchmarks,
            "results": [r for r in benchmarks["results"] if r.get("regime") != "leaked"],
        }

    payload = {
        "ok": True,
        "benchmarks": benchmarks,
        "encoder_history": _load_json_artifact("encoder_history.json"),
    }

    manifest = _load_json_artifact("manifest.json")
    if manifest:
        payload["dataset"] = {
            "clips": len(manifest.get("clips", [])),
            "speakers": manifest.get("speakers", []),
            "clips_per_speaker": manifest.get("clips_per_speaker", {}),
            "split_sizes": {k: len(v) for k, v in manifest.get("splits", {}).get("closed", {}).items()},
            "heldout_speakers": manifest.get("splits", {}).get("open", {}).get("heldout_speakers", []),
        }

    return jsonify(payload)



@api.get("/speakers")
def speakers():
    """The 50 dataset speakers with clip counts and held-out status."""
    manifest = _load_json_artifact("manifest.json")
    if manifest is None:
        return _error("manifest.json not built. Run training/prepare_data.py.", 503)

    heldout = set(manifest.get("splits", {}).get("open", {}).get("heldout_speakers", []))
    counts = manifest.get("clips_per_speaker", {})

    return jsonify(
        {
            "ok": True,
            "speakers": [
                {"name": name, "clips": counts.get(name, 0), "heldout": name in heldout}
                for name in manifest.get("speakers", [])
            ],
        }
    )

