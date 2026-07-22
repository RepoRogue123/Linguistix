# Multi-stage: Node builds the React app, Python serves it alongside the API.
# One image, one origin, one URL. That matters beyond tidiness — microphone
# access requires a secure context, and keeping the frontend on the same origin
# as the API avoids both CORS and a cross-origin hop on every prediction.

# ---- stage 1: build the frontend -------------------------------------------
FROM node:22-slim AS frontend

WORKDIR /build

# Copy manifests first so `npm ci` is cached unless dependencies actually change.
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund

COPY frontend/ ./
# The exported ONNX graph is served from public/models; without it the Lab and
# streaming views have nothing to load.
COPY ml_website/models/encoder.onnx ml_website/models/encoder.json ./public/models/
RUN npm run build


# ---- stage 2: the server ----------------------------------------------------
FROM python:3.11-slim

# HF_HOME / NUMBA_CACHE_DIR / MPLCONFIGDIR: Spaces runs as a non-root user, so
# every library that wants a cache needs somewhere writable or it fails on import.
#
# OMP/MKL_NUM_THREADS: one math thread per worker. Torch otherwise starts a
# thread per core in every worker, so on a 2-vCPU Space the two workers fight
# over the same two cores under concurrent load. Inference is ~70 ms; the
# throughput this protects is worth more than shaving a few ms off one request.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/hf \
    NUMBA_CACHE_DIR=/tmp/numba \
    MPLCONFIGDIR=/tmp/mpl \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

# libsndfile for soundfile; libgomp for the sklearn/torch OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ml_website/ ./ml_website/
COPY training/ ./training/

# vite.config.ts writes to ../ml_website/static/dist relative to /build, which
# resolves to /ml_website/static/dist in the builder stage.
COPY --from=frontend /ml_website/static/dist ./ml_website/static/dist

# Enrolment writes here. The Spaces filesystem is ephemeral, so these reset on
# restart; that is surfaced in /api/gallery rather than hidden.
RUN mkdir -p ml_website/uploads /tmp/gallery \
    && chmod -R 777 /tmp/gallery ml_website/uploads

ENV LINGUISTIX_GALLERY_DB=/tmp/gallery/gallery.sqlite3

EXPOSE 7860

# Two workers with a generous timeout: model load is lazy and the first request
# after a cold start pays for it, so a short timeout would kill that request.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 2 --threads 4 --timeout 180 ml_website.app:app"]
