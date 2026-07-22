---
title: Linguistix Speaker Recognition
emoji: "🎤"
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Linguistix — Sonagraph

Identify who is speaking from a few seconds of audio. Fifty dataset speakers, plus anyone
enrolled at runtime, with no retraining.

The Space builds a React frontend and serves it from the same Flask process as the API, so
there is one origin and one URL. That is not only tidiness: microphone capture requires a
secure context, and a same-origin API avoids both CORS and a cross-origin hop per prediction.

## How it runs

`Dockerfile` is a two-stage build. Node compiles the frontend into
`ml_website/static/dist`; Python serves that bundle alongside the API under gunicorn on port
7860. Model loading is lazy, so the first request after a cold start pays for it — hence the
180 s worker timeout.

```bash
docker build -t linguistix .
docker run -p 7860:7860 linguistix
# http://localhost:7860
```

To run the two halves separately while developing:

```bash
pip install -r requirements.txt
python -m ml_website.app          # API on :7860, or $PORT

cd frontend && npm install && npm run dev   # Vite on :5173, proxying /api
```

## Model artifacts

These live in `ml_website/models/` and are committed, so the image builds from a clean clone
with no training step.

**Encoder** — the open-set speaker embedding model.

| File | Used by |
|---|---|
| `encoder.pt` | server-side inference |
| `encoder.json` | architecture + front-end constants, read by both server and browser |
| `encoder.onnx` | in-browser inference (the Lab and streaming views) |
| `encoder_history.json` | training curve and best held-out EER |

**Gallery** — who the system knows.

| File | Used by |
|---|---|
| `gallery_reference.npz` | the 50 reference centroids |
| `speaker_map.json` | 3-D projection behind the speaker map |
| `manifest.json` | per-clip dataset metadata |

**Model jury** — the secondary voters shown on a result.

| File | Used by |
|---|---|
| `classical_models.joblib` | KNN / SVM / ensemble voters |
| `ann_model.pth`, `feature_mean.npy`, `feature_std.npy`, `speaker_names.npy` | the MFCC network voter |
| `benchmarks.json` | the Arena leaderboard |

Two files in that directory are deliberately **not** committed. `gallery.sqlite3` is runtime
enrolment state, and the container points `LINGUISTIX_GALLERY_DB` at `/tmp` anyway.
`embeddings.npz` is written by `training/build_gallery.py` and never read at serve time.

### Rebuilding them

```bash
python -m training.prepare_data      # features + splits, reports dataset confounds
python -m training.train_classical   # the classical leaderboard
python -m training.train_encoder     # the embedding model
python -m training.build_gallery     # centroids + 3-D map
python -m training.export_onnx       # browser graph, with parity check against torch
```

`scikit-learn` must stay at the major version that pickled `classical_models.joblib`
(≥ 1.5). A major-version jump makes the jury fail to unpickle.

## Enrolment is ephemeral

The Spaces filesystem resets on restart, so enrolled speakers do not survive one. The app
reports this in `/api/gallery` rather than hiding it.

## Keeping the Space awake

Free `cpu-basic` Spaces pause after 48 h idle and stay down until restarted.
`.github/workflows/keep-space-alive.yml` pings every 6 h. It needs a repository variable
`HF_SPACE_ID` and a secret `HF_TOKEN` with write access to the Space.

---

CSL2050 Pattern Recognition and Machine Learning, Indian Institute of Technology Jodhpur.
Shashank Parchure, Atharva Honparkhe, Vyankatesh Deshpande, Abhinash Roy, Namya Dhingra,
and Damarasingu Akshaya Sree.
