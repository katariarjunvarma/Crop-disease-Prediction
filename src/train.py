"""Training pipeline: preprocess -> build model -> train -> save artifacts."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")  # headless backend so plots save without a display
import matplotlib.pyplot as plt

from . import config
from .data import load_datasets
from .model import build_model


def _plot_history(history, out_path):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, acc, "o-", label="train")
    ax1.plot(epochs, val_acc, "o-", label="validation")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("epoch")
    ax1.legend()

    ax2.plot(epochs, loss, "o-", label="train")
    ax2.plot(epochs, val_loss, "o-", label="validation")
    ax2.set_title("Loss")
    ax2.set_xlabel("epoch")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def train(epochs: int = config.EPOCHS):
    print("[1/4] Loading and preprocessing data...")
    train_ds, val_ds, class_names = load_datasets()
    print(f"      Classes ({len(class_names)}): {class_names}")

    print("[2/4] Building CNN model...")
    model = build_model(num_classes=len(class_names))
    model.summary()

    print(f"[3/4] Training for {epochs} epochs...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    print("[4/4] Saving model and artifacts...")
    model.save(config.MODEL_PATH)
    with open(config.CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f, indent=2)
    history_plot = config.REPORTS_DIR / "training_history.png"
    _plot_history(history, history_plot)

    print(f"      Model saved to       {config.MODEL_PATH}")
    print(f"      Class names saved to {config.CLASS_NAMES_PATH}")
    print(f"      History plot saved to {history_plot}")
    return model, class_names


if __name__ == "__main__":
    train()
