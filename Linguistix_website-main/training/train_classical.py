"""Re-run every classical model with dimensionality reduction fit inside the training fold.

The original notebooks fit LDA and PCA on the full dataset and only then split
into train/test. LDA is supervised, so fitting it on everything leaks the test
labels into the projection and the resulting accuracies (ANN+LDA 100.00%,
KNN+LDA 99.80%) are not measurements of generalization. The honestly-evaluated
deployed model sits at 88.49% on the same data, which is the size of the gap.

This script reports both numbers on purpose:

  leakfree   reducer fitted on train only, then applied to val/test. The real number.
  leaked     reducer fitted on everything, reproducing the notebooks. Kept so the
             site can show the gap rather than quietly restating better numbers.

Usage
-----
    python training/train_classical.py
    python training/train_classical.py --reps lda --models knn svm
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(__file__).resolve().parents[1] / "ml_website" / "models"
MANIFEST = MODELS_DIR / "manifest.json"
FEATURES = REPO_ROOT / "Extracted Data" / "X_features.npy"
LABELS = REPO_ROOT / "Extracted Data" / "y_labels.npy"

SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=MODELS_DIR / "benchmarks.json")
    p.add_argument(
        "--reps",
        nargs="+",
        default=["raw", "pca", "lda", "pca_lda"],
        choices=["raw", "pca", "lda", "pca_lda"],
    )
    p.add_argument("--models", nargs="+", default=None, help="Subset of model keys; default is all.")
    p.add_argument("--pca-components", type=int, default=150)
    p.add_argument("--skip-leaked", action="store_true", help="Skip the leakage comparison run.")
    p.add_argument(
        "--save-models",
        type=str,
        default=None,
        metavar="REP",
        help="Persist the leak-free models fitted on REP (e.g. pca) for the API's model jury.",
    )
    return p.parse_args()


def build_models(seed: int = SEED) -> dict:
    """The classifier lineup from the notebooks, with fixed seeds."""
    tree = DecisionTreeClassifier(random_state=seed)
    return {
        "knn": ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=5, weights="distance")),
        "svm": ("Support Vector Machine", SVC(kernel="rbf", C=10.0, gamma="scale", random_state=seed)),
        "bayes": ("Gaussian Naive Bayes", GaussianNB()),
        "tree": ("Decision Tree", tree),
        "bagging": ("Decision Tree + Bagging", BaggingClassifier(tree, n_estimators=50, random_state=seed, n_jobs=-1)),
        "forest": ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)),
        # SAMME is the only boosting algorithm in current sklearn; the old
        # `algorithm=` argument was removed, so it is no longer passed.
        "adaboost": (
            "AdaBoost (SAMME)",
            AdaBoostClassifier(
                DecisionTreeClassifier(max_depth=6, random_state=seed),
                n_estimators=100,
                random_state=seed,
            ),
        ),
    }


def fit_reducer(rep: str, X: np.ndarray, y: np.ndarray, n_pca: int):
    """Return a fitted (scaler, reducer) pair. Both see only the data passed in."""
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    if rep == "raw":
        return scaler, None
    if rep == "pca":
        # Cannot exceed min(n_samples, n_features).
        n = min(n_pca, Xs.shape[0], Xs.shape[1])
        return scaler, PCA(n_components=n, random_state=SEED).fit(Xs)
    if rep == "lda":
        # LDA yields at most n_classes - 1 components.
        n = min(len(np.unique(y)) - 1, Xs.shape[1])
        return scaler, LinearDiscriminantAnalysis(n_components=n).fit(Xs, y)
    if rep == "pca_lda":
        # PCA first, then LDA. Fitting LDA directly on 4000 features from ~1760
        # training samples leaves the within-class scatter matrix rank-deficient,
        # so the projection it finds is unstable and does not transfer. Reducing
        # to a well-conditioned subspace first is the textbook remedy, and it is
        # the only way to judge LDA fairly here rather than judging that failure.
        n_lda = min(len(np.unique(y)) - 1, Xs.shape[1])
        n_pca = min(n_pca, Xs.shape[0] - len(np.unique(y)), Xs.shape[1])
        return scaler, Pipeline(
            [
                ("pca", PCA(n_components=n_pca, random_state=SEED)),
                ("lda", LinearDiscriminantAnalysis(n_components=n_lda)),
            ]
        ).fit(Xs, y)
    raise ValueError(rep)


def apply_reducer(scaler, reducer, X: np.ndarray) -> np.ndarray:
    Xs = scaler.transform(X)
    return Xs if reducer is None else reducer.transform(Xs)


def score(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    """Accuracy plus the macro averages that expose per-class failure under imbalance."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    per_class_p, per_class_r, per_class_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(n_classes), average=None, zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)) * 100, 2),
        "macro_precision": round(float(precision) * 100, 2),
        "macro_recall": round(float(recall) * 100, 2),
        "macro_f1": round(float(f1) * 100, 2),
        "per_class": {
            "precision": [round(float(v) * 100, 2) for v in per_class_p],
            "recall": [round(float(v) * 100, 2) for v in per_class_r],
            "f1": [round(float(v) * 100, 2) for v in per_class_f1],
            "support": [int(v) for v in support],
        },
    }


def cluster_purity(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Fraction of samples in the majority class of their assigned cluster."""
    total = 0
    for cluster in np.unique(labels_pred):
        members = labels_true[labels_pred == cluster]
        if members.size:
            total += np.bincount(members).max()
    return float(total) / len(labels_true)


def run_supervised(models, reps, splits, X, y, n_classes, n_pca, tag, results):
    """Fit and score every (representation, model) pair for one leakage regime."""
    tr, va, te = splits

    for rep in reps:
        t0 = time.perf_counter()
        if tag == "leakfree":
            scaler, reducer = fit_reducer(rep, X[tr], y[tr], n_pca)
        else:
            # Deliberately wrong: the reducer sees val and test labels too.
            scaler, reducer = fit_reducer(rep, X, y, n_pca)

        Xtr, Xva, Xte = (apply_reducer(scaler, reducer, X[s]) for s in (tr, va, te))
        dims = Xtr.shape[1]
        print(f"  [{tag}] {rep:4s} -> {dims:4d} dims  (fit {time.perf_counter()-t0:.1f}s)")

        for key, (name, model) in models.items():
            t1 = time.perf_counter()
            try:
                model.fit(Xtr, y[tr])
                val = score(y[va], model.predict(Xva), n_classes)
                test = score(y[te], model.predict(Xte), n_classes)
            except Exception as exc:
                print(f"      {key:9s} FAILED: {exc}")
                continue

            entry = {
                "key": f"{key}_{rep}",
                "model": name,
                "representation": rep.upper(),
                "dims": int(dims),
                "regime": tag,
                "val": val,
                "test": test,
                "fit_seconds": round(time.perf_counter() - t1, 2),
            }
            if tag == "leakfree" and rep == "lda" and key == "knn":
                entry["confusion"] = confusion_matrix(
                    y[te], model.predict(Xte), labels=np.arange(n_classes)
                ).tolist()

            results.append(entry)
            print(f"      {key:9s} val {val['accuracy']:6.2f}%  test {test['accuracy']:6.2f}%  f1 {test['macro_f1']:6.2f}")


def run_clustering(reps, splits, X, y, n_classes, n_pca, results):
    """K-Means and GMM, scored by purity since cluster ids carry no label."""
    tr, _, te = splits

    for rep in reps:
        if rep == "raw":
            continue  # clustering 4000 raw dims is not informative and is slow
        scaler, reducer = fit_reducer(rep, X[tr], y[tr], n_pca)
        # float64 and a larger covariance floor: with 50 components over a
        # 49-dim LDA projection some clusters collapse to near-singletons and the
        # EM fit fails outright on float32 with a small reg_covar.
        Xtr, Xte = (apply_reducer(scaler, reducer, X[s]).astype(np.float64) for s in (tr, te))

        for key, name, model in (
            ("kmeans", "K-Means", KMeans(n_clusters=n_classes, random_state=SEED, n_init=10)),
            ("gmm", "Gaussian Mixture", GaussianMixture(n_components=n_classes, covariance_type="diag", random_state=SEED, reg_covar=1e-2)),
        ):
            t0 = time.perf_counter()
            try:
                model.fit(Xtr)
                purity = cluster_purity(y[te], model.predict(Xte))
            except Exception as exc:
                print(f"      {key:9s} FAILED: {exc}")
                continue

            results.append(
                {
                    "key": f"{key}_{rep}",
                    "model": name,
                    "representation": rep.upper(),
                    "dims": int(Xtr.shape[1]),
                    "regime": "leakfree",
                    "metric": "purity",
                    "test": {"accuracy": round(purity * 100, 2)},
                    "fit_seconds": round(time.perf_counter() - t0, 2),
                }
            )
            print(f"      {key:9s} purity {purity*100:6.2f}%")


def main() -> None:
    args = parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    X = np.load(FEATURES).astype(np.float32)
    y_names = np.load(LABELS, allow_pickle=True)

    # X_features.npy and the manifest are built by walking the archive in the same
    # sorted order, so row i must correspond to clip i. Verify rather than trust:
    # a silent misalignment here would poison every number below.
    manifest_speakers = [c["speaker"] for c in manifest["clips"]]
    if len(manifest_speakers) != len(y_names) or manifest_speakers != list(y_names):
        raise SystemExit(
            "Manifest clip order does not match X_features.npy label order. "
            "Re-run extract_archive_features.py and prepare_data.py together."
        )

    encoder = LabelEncoder().fit(sorted(set(y_names)))
    y = encoder.transform(y_names)
    n_classes = len(encoder.classes_)

    closed = manifest["splits"]["closed"]
    splits = (np.array(closed["train"]), np.array(closed["val"]), np.array(closed["test"]))
    print(f"Features {X.shape} | {n_classes} classes | train {len(splits[0])} val {len(splits[1])} test {len(splits[2])}\n")

    models = build_models()
    if args.models:
        models = {k: v for k, v in models.items() if k in args.models}

    results: list[dict] = []

    print("Leak-free regime (reducer fit on train fold only) -- these are the real numbers")
    run_supervised(models, args.reps, splits, X, y, n_classes, args.pca_components, "leakfree", results)
    print("\n  clustering")
    run_clustering(args.reps, splits, X, y, n_classes, args.pca_components, results)

    if not args.skip_leaked:
        print("\nLeaked regime (reducer fit on all data) -- reproduces the notebooks, not a real result")
        run_supervised(models, args.reps, splits, X, y, n_classes, args.pca_components, "leaked", results)

    payload = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "feature_dim": int(X.shape[1]),
        "n_classes": n_classes,
        "classes": encoder.classes_.tolist(),
        "split_sizes": {k: len(v) for k, v in zip(("train", "val", "test"), splits)},
        "note": (
            "leakfree fits LDA/PCA on the training fold only and is the honest measurement. "
            "leaked fits on the full dataset, reproducing the original notebooks, and is "
            "reported only to quantify how much the original numbers were inflated."
        ),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nWrote {args.out}")

    if args.save_models:
        # Refit on the chosen representation and persist, so /api/jury can run
        # these models on a live clip. Only the leak-free fit is ever saved:
        # shipping a reducer that was fitted on the test set would put the
        # inflated numbers back into the product through the side door.
        import joblib

        rep = args.save_models
        scaler, reducer = fit_reducer(rep, X[splits[0]], y[splits[0]], args.pca_components)
        Xtr = apply_reducer(scaler, reducer, X[splits[0]])

        fitted = {}
        for key, (name, model) in build_models().items():
            model.fit(Xtr, y[splits[0]])
            fitted[key] = {"name": name, "model": model}

        bundle_path = args.out.parent / "classical_models.joblib"
        joblib.dump(
            {
                "representation": rep,
                "scaler": scaler,
                "reducer": reducer,
                "classes": encoder.classes_.tolist(),
                "models": fitted,
                "feature_dim": int(X.shape[1]),
                "regime": "leakfree",
            },
            bundle_path,
            compress=3,
        )
        print(f"Saved {len(fitted)} fitted models ({rep}) to {bundle_path}")

    lf = [r for r in results if r["regime"] == "leakfree" and "metric" not in r]
    if lf:
        best = max(lf, key=lambda r: r["test"]["accuracy"])
        print(f"Best leak-free: {best['model']} + {best['representation']} -> {best['test']['accuracy']}% test")

    leaked = [r for r in results if r["regime"] == "leaked"]
    if leaked:
        pairs = {(r["key"]) : r["test"]["accuracy"] for r in leaked}
        gaps = [
            (r["key"], pairs[r["key"]] - r["test"]["accuracy"])
            for r in lf
            if r["key"] in pairs
        ]
        if gaps:
            worst = max(gaps, key=lambda kv: kv[1])
            print(f"Largest leakage inflation: {worst[0]} +{worst[1]:.2f} points")


if __name__ == "__main__":
    main()
