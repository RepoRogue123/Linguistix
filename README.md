# Linguistix — Speaker Recognition

**CSL2050 · Pattern Recognition and Machine Learning · Indian Institute of Technology Jodhpur**

Identify who is speaking from a few seconds of audio. The system recognizes 50 dataset speakers and
anyone you enrol at runtime, with no retraining.

## Team

Shashank Parchure (B23CM1059) · Atharva Honparkhe (B23EE1006) · Vyankatesh Deshpande (B23CS1079) ·
Abhinash Roy (B23CS1003) · Namya Dhingra (B23CS1040) · Damarasingu Akshaya Sree (B23EE1085)

---

## What this is

Two systems, evaluated separately because they answer different questions.

**The classical benchmark** compares KNN, SVM, Naive Bayes, decision trees, ensembles, K-Means and
GMMs across PCA, LDA, and PCA→LDA representations. Closed-set: every model must name one of the 50.

**The speaker encoder** is an ECAPA-TDNN-style network (2.05M parameters) trained with additive
angular margin loss. It maps a voice to a 192-dimension embedding, so identification is a
nearest-neighbour search against a gallery and enrolling a new speaker costs one forward pass.
Open-set: it can also decline to answer.

---

## Results

### Classical models

Dimensionality reduction fitted inside the training fold only, stratified 70/20/10 split,
2511 clips across 50 speakers.

| Model | Representation | Test accuracy | Macro F1 |
|---|---|---:|---:|
| Random Forest | PCA→LDA | **92.71%** | 91.52 |
| SVM (RBF) | PCA→LDA | 91.09% | 89.75 |
| Gaussian Naive Bayes | PCA→LDA | 90.69% | 88.91 |
| SVM (RBF) | PCA | 88.66% | 88.29 |
| Decision Tree + Bagging | PCA→LDA | 87.45% | 87.01 |
| KNN | PCA→LDA | 85.02% | 85.20 |
| AdaBoost (SAMME) | PCA→LDA | 82.19% | 81.89 |
| Random Forest | PCA | 81.38% | 79.25 |
| KNN | PCA | 52.63% | 48.99 |

Clustering, scored by purity: K-Means + PCA→LDA 85.02%, GMM + PCA→LDA 84.21%.

Macro F1 is worth reading alongside accuracy: clips per speaker range from 10 to 120, so a model can
look respectable on accuracy while failing the speakers with the least data.

### Speaker encoder, evaluated open-set

| Metric | Value |
|---|---|
| Equal error rate, speakers **never seen in training** | **10.15%** |
| minDCF (p_target = 0.01) | 0.70 |
| Embedding dimension | 192 |
| Parameters | 2.05M |
| Calibrated acceptance threshold | 0.391 cosine |

Trained on 40 speakers, evaluated on 10 withheld entirely. This number is **not** comparable to the
closed-set accuracies above — it answers the harder question of whether a voice the model has never
heard can still be enrolled and recognized.

---

## Running it

```bash
# 1. Environment. For training, install the CUDA build instead:
#    pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r Linguistix_website-main/requirements.txt

# 2. Data pipeline: manifest, stratified split, speaker-disjoint split, confound report
python Linguistix_website-main/training/prepare_data.py

# 3. Classical benchmark (writes benchmarks.json and the jury's models)
python Linguistix_website-main/training/train_classical.py --save-models pca_lda

# 4. Speaker encoder (~15 min on an RTX 4070)
python Linguistix_website-main/training/train_encoder.py

# 5. Gallery centroids and the 2D speaker map
python Linguistix_website-main/training/build_gallery.py

# 6. Export for in-browser inference. Fails loudly if the graph disagrees
#    with PyTorch by more than 1e-4.
python Linguistix_website-main/training/export_onnx.py --verify

# 7. Frontend
cd Linguistix_website-main/frontend && npm install && npm run build

# 8. Serve
cd Linguistix_website-main && python -m ml_website.app     # http://localhost:7860
```

## Layout

```
Linguistix_website-main/
  ml_website/
    app.py              Flask entry: JSON API + built SPA
    api/routes.py       identify · enrol · gallery · verify · map · metrics · jury · explain
    engine/
      features.py       the only place audio becomes features
      encoder.py        TDNN + AAM-Softmax, mel front-end inside the graph
      gallery.py        enrolment store and cosine search
      jury.py           multi-model consensus
      explain.py        gradient saliency over the mel spectrogram
    models/             checkpoints, benchmarks, manifest, ONNX graph
  training/             prepare_data · train_classical · train_encoder · build_gallery · export_onnx
  frontend/             Vite + React + TypeScript
ANN/ SVM/ KNN/ …        research notebooks
```

## Notes on the interface

The design is built around the **voiceprint terrain** — the spectrogram as a 3D landscape that is
the input meter while recording, the progress indicator while analysing, and the result afterwards,
with saliency lifting the identifying ridges. Two themes, SCOPE and PRINT, render it differently:
the magma colormap on a dark console, or grayscale ink density on paper, as a real sound
spectrograph printed it.

The **Map** renders all 2511 clips as an orbitable 3D point cloud of the encoder's embedding space.
The **Lab** runs the encoder in your browser via ONNX Runtime, so degradation sliders respond
instantly rather than queueing a request per frame against a free CPU tier. The mel front-end is
baked into the exported graph so the browser and server cannot disagree about features;
`export_onnx.py --verify` asserts they match to under 1e-4 before shipping.

## Deployment

Single Docker image on a Hugging Face Space: Node builds the frontend, Python serves it beside the
API on port 7860. Enrolments are stored on the container filesystem and reset when the Space
restarts; `/api/gallery` reports this rather than letting it surprise anyone.

## Dataset

[50-speaker recognition corpus](https://www.kaggle.com/datasets/vjcalling/speaker-recognition-audio-dataset)
— 2511 WAV clips, 50 speakers, 41.4 hours. Note the dataset has no `Speaker_0022`: numbering runs to
0050 while there are only 50 classes, so class index and numeric suffix diverge past index 21.

## Resources

- [Spotlight video](https://youtu.be/yORB3cY9WDA)
- [Project page](https://vyankateshd206.github.io/Linguistix/)
- Reports in [`Report/`](Report/)
