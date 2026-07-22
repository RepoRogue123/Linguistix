"""Flask entry point: JSON API plus the built React app.

Deliberately thin. Inference lives in engine/, routes live in api/, and this
module only wires them together and serves static files.

Model loading is lazy (see engine/service.py). Importing this module touches no
checkpoint, so a missing artifact degrades only the endpoints that need it
instead of preventing the process from starting.
"""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from .api import api

# Vite writes here; see frontend/vite.config.ts.
DIST_DIR = Path(__file__).resolve().parent / "static" / "dist"

MAX_UPLOAD_MB = int(os.environ.get("LINGUISTIX_MAX_UPLOAD_MB", "32"))


def create_app() -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
    app.config["JSON_SORT_KEYS"] = False

    app.register_blueprint(api)

    @app.errorhandler(413)
    def too_large(_):
        return jsonify({"ok": False, "error": f"Upload exceeds the {MAX_UPLOAD_MB} MB limit."}), 413

    @app.errorhandler(404)
    def not_found(_):
        # API 404s stay JSON. Unknown page routes fall through to the SPA below
        # so client-side routing survives a hard refresh.
        if request.path.startswith("/api/"):
            return jsonify({"ok": False, "error": f"No such endpoint: {request.path}"}), 404
        return serve_spa(request.path.lstrip("/"))

    # /health is what .github/workflows/keep-space-alive.yml pings. Deliberately
    # liveness-only: a keep-alive ping should not force every model to load.
    @app.get("/healthz")
    @app.get("/health")
    def healthz():
        return jsonify({"ok": True, "service": "linguistix", "detail": "/api/health"})

    @app.get("/", defaults={"path": ""})
    @app.get("/<path:path>")
    def serve_spa(path: str):
        if not DIST_DIR.exists():
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": "Frontend not built.",
                        "hint": "cd frontend && npm install && npm run build",
                        "api": "/api/health",
                    }
                ),
                503,
            )

        candidate = DIST_DIR / path
        if path and candidate.is_file():
            return send_from_directory(DIST_DIR, path)

        return send_from_directory(DIST_DIR, "index.html")

    return app


app = create_app()


if __name__ == "__main__":
    # 7860 is the Hugging Face Spaces convention.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)), debug=False)
