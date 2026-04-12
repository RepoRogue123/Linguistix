---
title: Linguistix Speaker Recognition
emoji: "🎤"
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

# Linguistix Speaker Recognition

This Space runs the Flask backend in `ml_website/app.py` with real model inference.

## Required model artifacts

Place these files in `ml_website/models/`:

- `ann_model.pth`
- `feature_mean.npy`
- `feature_std.npy`
- `speaker_names.npy`

The app also reads labels fallback data from `ml_website/data/y_labels.npy` when needed.

## Train and export artifacts

Run from the repository root:

```bash
python generate_test_model.py \
  --features-path "../Extracted Data/X_features.npy" \
  --labels-path "../Extracted Data/y_labels.npy" \
  --models-dir ml_website/models
```

The `ml_website/data` files are small demo assets and may not be label-aligned for training.

After training new artifacts, restart the app or call `POST /reload-model`.

## Local run

```bash
docker build -t linguistix-speaker .
docker run -p 7860:7860 linguistix-speaker
```

Open `http://localhost:7860`.
