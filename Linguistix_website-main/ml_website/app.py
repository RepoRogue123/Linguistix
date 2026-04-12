from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import torch
import librosa
from werkzeug.utils import secure_filename

try:
    from .model import ANN, standardize_data
except ImportError:
    from model import ANN, standardize_data

app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_EXTENSIONS = {"wav"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload size

# Artifact locations for inference
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "ann_model.pth")
FEATURE_MEAN_PATH = os.path.join(MODEL_DIR, "feature_mean.npy")
FEATURE_STD_PATH = os.path.join(MODEL_DIR, "feature_std.npy")
SPEAKER_NAMES_PATH = os.path.join(MODEL_DIR, "speaker_names.npy")

X_FEATURES_PATH = os.path.join(DATA_DIR, "X_features.npy")
Y_LABELS_PATH = os.path.join(DATA_DIR, "y_labels.npy")


def _float_env(name, default_value):
    raw = os.getenv(name)
    if raw is None:
        return float(default_value)
    try:
        return float(raw)
    except ValueError:
        return float(default_value)


def _int_env(name, default_value):
    raw = os.getenv(name)
    if raw is None:
        return int(default_value)
    try:
        return int(raw)
    except ValueError:
        return int(default_value)


# Inference decision controls.
TOP_K_PREDICTIONS = max(1, _int_env("TOP_K_PREDICTIONS", 3))
UNKNOWN_MIN_CONFIDENCE = _float_env("UNKNOWN_MIN_CONFIDENCE", 35.0)
UNKNOWN_MIN_MARGIN = _float_env("UNKNOWN_MIN_MARGIN", 8.0)

# Global variables for model
model = None
feature_mean = None
feature_std = None
speaker_names = None
model_input_size = None
model_output_size = None
last_model_error = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_audio_features(audio_path, expected_dim):
    """Extract features from WAV and adapt shape to trained model input size."""
    y, sr = librosa.load(audio_path, sr=None)

    if expected_dim == 17:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)

        spectral_centroid_mean = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth_mean = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_rolloff_mean = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y))

        features = np.concatenate(
            [
                mfccs_mean,
                np.array(
                    [
                        spectral_centroid_mean,
                        spectral_bandwidth_mean,
                        spectral_rolloff_mean,
                        zcr_mean,
                    ]
                ),
            ]
        )
        return features.astype(np.float32)

    # Generic MFCC pipeline for high-dimensional models (for example 4000-D vectors).
    n_mfcc = 40
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    target_frames = max(1, int(np.ceil(expected_dim / n_mfcc)))

    if mfcc.shape[1] < target_frames:
        pad_width = target_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :target_frames]

    flat = mfcc.flatten().astype(np.float32)
    if flat.shape[0] < expected_dim:
        flat = np.pad(flat, (0, expected_dim - flat.shape[0]), mode="constant")
    elif flat.shape[0] > expected_dim:
        flat = flat[:expected_dim]

    return flat


def _load_speaker_names():
    """Load speaker names in class-index order."""
    if os.path.exists(SPEAKER_NAMES_PATH):
        names = np.load(SPEAKER_NAMES_PATH, allow_pickle=True)
        return np.asarray(names)

    if not os.path.exists(Y_LABELS_PATH):
        raise FileNotFoundError(
            f"Missing speaker labels at {Y_LABELS_PATH} and {SPEAKER_NAMES_PATH}."
        )

    labels = np.load(Y_LABELS_PATH, allow_pickle=True)
    labels = np.asarray(labels)
    if labels.size == 0:
        raise ValueError("y_labels.npy is empty.")

    if np.issubdtype(labels.dtype, np.str_) or labels.dtype == object:
        # Preserve first-seen order so class mapping stays stable.
        seen = []
        seen_set = set()
        for label in labels.tolist():
            if label not in seen_set:
                seen_set.add(label)
                seen.append(label)
        return np.asarray(seen)

    unique_labels = np.unique(labels)
    return np.asarray([f"Speaker_{int(x)}" for x in unique_labels])


def _load_standardization_stats(expected_input_size):
    """Load feature scaling stats from model artifacts or build from training features."""
    if os.path.exists(FEATURE_MEAN_PATH) and os.path.exists(FEATURE_STD_PATH):
        mean = np.load(FEATURE_MEAN_PATH).astype(np.float32)
        std = np.load(FEATURE_STD_PATH).astype(np.float32)
    else:
        if not os.path.exists(X_FEATURES_PATH):
            raise FileNotFoundError(
                f"Missing feature stats files and fallback features file at {X_FEATURES_PATH}."
            )
        X_features = np.load(X_FEATURES_PATH).astype(np.float32)
        mean = np.mean(X_features, axis=0)
        std = np.std(X_features, axis=0)

    std = np.where(std == 0, 1.0, std)

    if mean.shape[0] != expected_input_size or std.shape[0] != expected_input_size:
        raise ValueError(
            "Feature stats shape mismatch: "
            f"mean={mean.shape}, std={std.shape}, input_size={expected_input_size}."
        )

    return mean, std


def load_model():
    """Load model + preprocessing artifacts used by real inference."""
    global model
    global feature_mean
    global feature_std
    global speaker_names
    global model_input_size
    global model_output_size
    global last_model_error

    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

        speaker_names_local = _load_speaker_names()
        output_size = len(speaker_names_local)

        if os.path.exists(FEATURE_MEAN_PATH):
            input_size = int(np.load(FEATURE_MEAN_PATH).shape[0])
        elif os.path.exists(X_FEATURES_PATH):
            input_size = int(np.load(X_FEATURES_PATH).shape[1])
        else:
            input_size = 17

        model_local = ANN(input_size=input_size, hidden_size=128, output_size=output_size)
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model_local.load_state_dict(state_dict)
        model_local.to(device)
        model_local.eval()

        mean_local, std_local = _load_standardization_stats(input_size)

        model = model_local
        feature_mean = mean_local
        feature_std = std_local
        speaker_names = speaker_names_local
        model_input_size = input_size
        model_output_size = output_size
        last_model_error = None

        print(
            "Model loaded successfully "
            f"(input_size={model_input_size}, output_size={model_output_size}, device={device})"
        )
    except Exception as exc:
        model = None
        feature_mean = None
        feature_std = None
        speaker_names = None
        model_input_size = None
        model_output_size = None
        last_model_error = str(exc)
        print(f"Model load failed: {last_model_error}")


def _speaker_name_from_class(class_idx):
    if speaker_names is not None and class_idx < len(speaker_names):
        return str(speaker_names[class_idx])
    return f"Speaker_{class_idx}"


def _predict_from_feature_vector(features_1d):
    if model is None:
        raise RuntimeError("Model not loaded.")

    if features_1d.shape[0] != model_input_size:
        raise ValueError(
            f"Feature length mismatch: got {features_1d.shape[0]}, expected {model_input_size}."
        )

    features_array = features_1d.reshape(1, -1).astype(np.float32)
    features_std = standardize_data(features_array, feature_mean, feature_std)
    features_tensor = torch.tensor(features_std, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(features_tensor)
        probs = torch.softmax(outputs, dim=1)
        top_k = min(TOP_K_PREDICTIONS, probs.shape[1])
        top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)

    top_probs_np = top_probs.squeeze(0).cpu().numpy()
    top_indices_np = top_indices.squeeze(0).cpu().numpy()

    top_predictions = []
    for rank, (cls_idx, cls_prob) in enumerate(zip(top_indices_np, top_probs_np), start=1):
        cls_int = int(cls_idx)
        top_predictions.append(
            {
                "rank": rank,
                "prediction": cls_int,
                "speaker": _speaker_name_from_class(cls_int),
                "confidence": float(cls_prob * 100.0),
            }
        )

    pred_class = top_predictions[0]["prediction"]
    matched_speaker = top_predictions[0]["speaker"]
    confidence_val = top_predictions[0]["confidence"]
    confidence_margin = confidence_val
    if len(top_predictions) > 1:
        confidence_margin = confidence_val - top_predictions[1]["confidence"]

    unknown_reasons = []
    if confidence_val < UNKNOWN_MIN_CONFIDENCE:
        unknown_reasons.append(
            f"confidence<{UNKNOWN_MIN_CONFIDENCE:.1f}%"
        )
    if confidence_margin < UNKNOWN_MIN_MARGIN:
        unknown_reasons.append(
            f"margin<{UNKNOWN_MIN_MARGIN:.1f}%"
        )

    is_unknown = len(unknown_reasons) > 0
    speaker_name = "Unknown" if is_unknown else matched_speaker
    decision_reason = "accepted" if not is_unknown else ", ".join(unknown_reasons)

    return {
        "prediction": pred_class,
        "speaker": speaker_name,
        "matched_speaker": matched_speaker,
        "confidence": confidence_val,
        "confidence_margin": float(confidence_margin),
        "is_unknown": is_unknown,
        "decision_reason": decision_reason,
        "top_predictions": top_predictions,
    }


# Load model at startup so inference paths are ready immediately.
load_model()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/model-info")
def model_info():
    return render_template("model_info.html")


@app.route("/visualization")
def visualization():
    return render_template("visualization.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "model_loaded": model is not None,
            "input_size": model_input_size,
            "output_size": model_output_size,
            "device": str(device),
            "error": last_model_error,
            "top_k": TOP_K_PREDICTIONS,
            "unknown_min_confidence": UNKNOWN_MIN_CONFIDENCE,
            "unknown_min_margin": UNKNOWN_MIN_MARGIN,
        }
    )


@app.route("/reload-model", methods=["POST"])
def reload_model():
    load_model()
    status_code = 200 if model is not None else 500
    return jsonify({"success": model is not None, "error": last_model_error}), status_code


@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle WAV upload and run true model inference."""
    if model is None:
        return jsonify({"success": False, "error": f"Model not loaded: {last_model_error}"}), 503

    if "audio_file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files["audio_file"]

    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400

    if not (file and allowed_file(file.filename)):
        return jsonify({"success": False, "error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        file.save(file_path)
        audio_features = extract_audio_features(file_path, model_input_size)
        prediction = _predict_from_feature_vector(audio_features)
        return jsonify({"success": True, **prediction})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.route("/predict", methods=["POST"])
def predict():
    """Direct feature prediction endpoint for testing/debug use."""
    if model is None:
        return jsonify({"success": False, "error": f"Model not loaded: {last_model_error}"}), 503

    try:
        data = request.get_json(silent=True) or {}
        features = data.get("features", [])
        if len(features) == 0:
            return jsonify({"success": False, "error": "No features provided"}), 400

        features_1d = np.asarray(features, dtype=np.float32).reshape(-1)
        prediction = _predict_from_feature_vector(features_1d)
        return jsonify({"success": True, **prediction})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
