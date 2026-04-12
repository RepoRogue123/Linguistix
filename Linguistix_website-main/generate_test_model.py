"""Train ANN on extracted features and export runtime artifacts for the website.

This script is intentionally named the same as before so existing workflows keep working,
but it now performs real training instead of creating random demo weights.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from ml_website.model import ANN


def parse_args():
    parser = argparse.ArgumentParser(description="Train and export speaker model artifacts")
    parser.add_argument(
        "--features-path",
        default=os.path.join("ml_website", "data", "X_features.npy"),
        help="Path to feature matrix (.npy), shape [n_samples, n_features].",
    )
    parser.add_argument(
        "--labels-path",
        default=os.path.join("ml_website", "data", "y_labels.npy"),
        help="Path to speaker labels (.npy), shape [n_samples].",
    )
    parser.add_argument(
        "--models-dir",
        default=os.path.join("ml_website", "models"),
        help="Directory to write model artifacts.",
    )
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="L2 regularization strength for Adam optimizer.",
    )
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden layer size")
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Validation fraction from the full dataset.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Test fraction from the full dataset.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Early stopping patience in epochs based on validation accuracy.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Minimum validation accuracy improvement (percentage points) to reset patience.",
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=3,
        help="LR scheduler patience in epochs based on validation accuracy.",
    )
    parser.add_argument(
        "--lr-factor",
        type=float,
        default=0.5,
        help="Factor to reduce LR when validation accuracy plateaus.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_data(X, y):
    if X.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {X.shape}")
    if y.ndim != 1:
        y = y.reshape(-1)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Feature/label size mismatch: {X.shape[0]} vs {y.shape[0]}")
    if X.shape[0] < 20:
        raise ValueError("Dataset too small for a stable split (need at least 20 samples).")


def standardize_with_train_stats(X_train, X_val, X_test):
    mean = np.mean(X_train, axis=0).astype(np.float32)
    std = np.std(X_train, axis=0).astype(np.float32)
    std = np.where(std == 0, 1.0, std)

    X_train_std = ((X_train - mean) / std).astype(np.float32)
    X_val_std = ((X_val - mean) / std).astype(np.float32)
    X_test_std = ((X_test - mean) / std).astype(np.float32)
    return X_train_std, X_val_std, X_test_std, mean, std


def evaluate_metrics(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item() * y_batch.size(0)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    avg_loss = (total_loss / total) if total else 0.0
    accuracy = (100.0 * correct / total) if total else 0.0
    return avg_loss, accuracy


def save_learning_curves(history, output_path):
    epochs = history["epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].plot(epochs, history["test_loss"], label="Test")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Validation")
    axes[1].plot(epochs, history["test_acc"], label="Test")
    axes[1].set_title("Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.models_dir, exist_ok=True)

    if args.val_size <= 0 or args.test_size <= 0:
        raise ValueError("Validation and test sizes must be > 0.")
    if args.val_size + args.test_size >= 0.8:
        raise ValueError("val_size + test_size is too large; keep at least 20% for training.")
    if args.patience <= 0:
        raise ValueError("patience must be > 0.")
    if args.lr_patience <= 0:
        raise ValueError("lr-patience must be > 0.")
    if not (0.0 < args.lr_factor < 1.0):
        raise ValueError("lr-factor must be in (0, 1).")

    if not os.path.exists(args.features_path):
        raise FileNotFoundError(f"Features file not found: {args.features_path}")
    if not os.path.exists(args.labels_path):
        raise FileNotFoundError(f"Labels file not found: {args.labels_path}")

    print("Loading features and labels...")
    X = np.load(args.features_path).astype(np.float32)
    y_raw = np.load(args.labels_path, allow_pickle=True)
    y_raw = np.asarray(y_raw)

    validate_data(X, y_raw)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    speaker_names = label_encoder.classes_

    holdout_size = args.val_size + args.test_size
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=holdout_size,
        random_state=args.seed,
        stratify=y,
    )

    test_ratio_in_holdout = args.test_size / holdout_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=test_ratio_in_holdout,
        random_state=args.seed,
        stratify=y_holdout,
    )

    X_train_std, X_val_std, X_test_std, feature_mean, feature_std = standardize_with_train_stats(
        X_train,
        X_val,
        X_test,
    )

    X_train_tensor = torch.tensor(X_train_std, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val_std, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_std, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val_tensor, y_val_tensor),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(X_test_tensor, y_test_tensor),
        batch_size=args.batch_size,
        shuffle=False,
    )

    input_size = X.shape[1]
    output_size = len(speaker_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ANN(input_size=input_size, hidden_size=args.hidden_size, output_size=output_size).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_factor,
        patience=args.lr_patience,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_test_acc = -1.0
    best_epoch = -1
    stopped_early = False
    best_state_dict = None
    epochs_without_improvement = 0
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "train_acc": [],
        "val_acc": [],
        "test_acc": [],
    }

    print(
        f"Training ANN (input={input_size}, classes={output_size}, epochs={args.epochs}, device={device})"
    )

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss, train_acc = evaluate_metrics(model, train_loader, device, criterion)
        val_loss, val_acc = evaluate_metrics(model, val_loader, device, criterion)
        test_loss, test_acc = evaluate_metrics(model, test_loader, device, criterion)

        prev_best_val_acc = best_val_acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = epoch + 1
            best_state_dict = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

        scheduler.step(val_acc)

        if val_acc > prev_best_val_acc + args.min_delta:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["test_loss"].append(float(test_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))
        history["test_acc"].append(float(test_acc))

        avg_train_step_loss = running_loss / max(1, len(train_loader))
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:03d}/{args.epochs:03d} | "
            f"train_step_loss={avg_train_step_loss:.4f} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | test_loss={test_loss:.4f} | "
            f"train_acc={train_acc:.2f}% | val_acc={val_acc:.2f}% | test_acc={test_acc:.2f}% | "
            f"lr={current_lr:.6f}"
        )

        if epochs_without_improvement >= args.patience:
            stopped_early = True
            print(
                "Early stopping triggered: "
                f"no validation-accuracy improvement for {args.patience} epochs."
            )
            break

    if best_state_dict is None:
        raise RuntimeError("Training completed but no best model state was captured.")

    model_path = os.path.join(args.models_dir, "ann_model.pth")
    mean_path = os.path.join(args.models_dir, "feature_mean.npy")
    std_path = os.path.join(args.models_dir, "feature_std.npy")
    names_path = os.path.join(args.models_dir, "speaker_names.npy")
    metadata_path = os.path.join(args.models_dir, "training_metadata.json")
    curves_path = os.path.join(args.models_dir, "learning_curves.png")
    history_path = os.path.join(args.models_dir, "training_history.json")

    torch.save(best_state_dict, model_path)
    np.save(mean_path, feature_mean)
    np.save(std_path, feature_std)
    np.save(names_path, speaker_names)
    save_learning_curves(history, curves_path)

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    metadata = {
        "input_size": int(input_size),
        "output_size": int(output_size),
        "best_epoch": int(best_epoch),
        "best_val_accuracy": float(best_val_acc),
        "best_test_accuracy": float(best_test_acc),
        "stopped_early": bool(stopped_early),
        "epochs_ran": int(len(history["epoch"])),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "hidden_size": int(args.hidden_size),
        "val_size": float(args.val_size),
        "test_size": float(args.test_size),
        "patience": int(args.patience),
        "min_delta": float(args.min_delta),
        "lr_patience": int(args.lr_patience),
        "lr_factor": float(args.lr_factor),
        "features_path": args.features_path,
        "labels_path": args.labels_path,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nArtifacts exported:")
    print(f"- {model_path}")
    print(f"- {mean_path}")
    print(f"- {std_path}")
    print(f"- {names_path}")
    print(f"- {metadata_path}")
    print(f"- {history_path}")
    print(f"- {curves_path}")
    print("\nWebsite backend can now load these artifacts for real inference.")


if __name__ == "__main__":
    main()
