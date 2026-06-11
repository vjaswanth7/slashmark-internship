"""
Dogs vs Cats CNN Classifier
Reference-based implementation inspired by the provided zip project,
updated for modern TensorFlow/Keras workflows.

Requirements satisfied:
- CNN for cat vs dog classification
- Data generators and augmentation
- Training/validation curves
- Confusion matrix
- Single-image prediction
- Model saving
- Uses Python, TensorFlow/Keras, NumPy, Matplotlib, scikit-learn
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# Optional, but useful for nicer confusion matrix plots
try:
    import seaborn as sns
except Exception:
    sns = None


IMG_SIZE = (128, 128)
BATCH_SIZE = 32
DEFAULT_EPOCHS = 30
SEED = 42


def build_datasets(data_dir: Path, img_size: Tuple[int, int] = IMG_SIZE, batch_size: int = BATCH_SIZE):
    """
    Expects:
    data_dir/
      train/
        cats/
        dogs/
      validation/
        cats/
        dogs/
    """
    train_dir = data_dir / "train"
    val_dir = data_dir / "validation"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Expected folder structure not found under {data_dir}. "
            "Need train/ and validation/ folders, each with cats/ and dogs/ subfolders."
        )

    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    val_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)

    train_data = train_gen.flow_from_directory(
        train_dir.as_posix(),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
        seed=SEED,
    )

    val_data = val_gen.flow_from_directory(
        val_dir.as_posix(),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )

    return train_data, val_data


def build_model(input_shape=(128, 128, 3)) -> tf.keras.Model:
    """
    A compact CNN that performs well for a binary cat-vs-dog task.
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),

            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history: tf.keras.callbacks.History, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(history.history.get("accuracy", []), label="Train Accuracy")
    plt.plot(history.history.get("val_accuracy", []), label="Validation Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_curve.png", dpi=150)
    plt.close()

    # Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(history.history.get("loss", []), label="Train Loss")
    plt.plot(history.history.get("val_loss", []), label="Validation Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close()


def evaluate_and_plot(model: tf.keras.Model, val_data, out_dir: Path) -> Dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)

    val_data.reset()
    probs = model.predict(val_data, verbose=0)
    y_pred = (probs.ravel() >= 0.5).astype(int)
    y_true = val_data.classes

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    if sns is not None:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    else:
        plt.imshow(cm, interpolation="nearest")
        plt.colorbar()
        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, str(v), ha="center", va="center")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    report = classification_report(y_true, y_pred, target_names=list(val_data.class_indices.keys()), output_dict=True)
    with open(out_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    metrics = {
        "validation_accuracy_from_predictions": float(np.mean(y_true == y_pred)),
        "validation_precision_dog": float(report.get("dog", {}).get("precision", 0.0)),
        "validation_recall_dog": float(report.get("dog", {}).get("recall", 0.0)),
        "validation_precision_cat": float(report.get("cat", {}).get("precision", 0.0)),
        "validation_recall_cat": float(report.get("cat", {}).get("recall", 0.0)),
    }
    return metrics


def train(data_dir: Path, out_dir: Path, epochs: int = DEFAULT_EPOCHS):
    tf.keras.utils.set_random_seed(SEED)

    train_data, val_data = build_datasets(data_dir)

    model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=(out_dir / "best_model.keras").as_posix(),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "dogs_vs_cats_cnn.keras")
    model.save(out_dir / "dogs_vs_cats_cnn.h5")

    plot_history(history, out_dir)
    metrics = evaluate_and_plot(model, val_data, out_dir)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nFinal metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print(f"\nArtifacts saved to: {out_dir.resolve()}")
    return model, history, metrics, val_data


def predict_image(model_path: Path, image_path: Path):
    model = tf.keras.models.load_model(model_path.as_posix())
    img = tf.keras.preprocessing.image.load_img(image_path.as_posix(), target_size=IMG_SIZE)
    arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    prob = float(model.predict(arr, verbose=0)[0][0])
    label = "dog" if prob >= 0.5 else "cat"
    confidence = prob if prob >= 0.5 else 1.0 - prob

    print(f"Prediction: {label}  |  confidence: {confidence:.4f}")
    return label, confidence


def parse_args():
    parser = argparse.ArgumentParser(description="Dogs vs Cats CNN Classifier")
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--model-path", type=Path, default=Path("outputs/dogs_vs_cats_cnn.keras"))
    parser.add_argument("--image-path", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "train":
        train(args.data_dir, args.out_dir, args.epochs)
    else:
        if args.image_path is None:
            raise ValueError("--image-path is required in predict mode")
        predict_image(args.model_path, args.image_path)


if __name__ == "__main__":
    main()
