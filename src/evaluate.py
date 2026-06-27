"""Performance evaluation using scikit-learn metrics.

Loads the trained model, runs it over the held-out validation set, and reports
accuracy, a full classification report (precision/recall/F1 per class) and a
confusion matrix.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from . import config
from .data import load_datasets


def _load_class_names():
    if config.CLASS_NAMES_PATH.exists():
        with open(config.CLASS_NAMES_PATH) as f:
            return json.load(f)
    return None


def evaluate():
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model at {config.MODEL_PATH}. Run `python main.py train` first."
        )

    print("Loading model and validation data...")
    model = tf.keras.models.load_model(config.MODEL_PATH)
    _, val_ds, class_names = load_datasets()
    class_names = _load_class_names() or class_names

    # Collect ground-truth labels and predictions across the validation set.
    y_true, y_pred = [], []
    for images, labels in val_ds:
        probs = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1))
        y_true.extend(labels.numpy())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    print("\n================ EVALUATION ================")
    print(f"Validation accuracy: {acc:.4f}\n")
    print(report)

    # Save the textual report.
    report_path = config.REPORTS_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Validation accuracy: {acc:.4f}\n\n{report}\n")

    # Save the confusion matrix as an image.
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Greens", xticks_rotation=45, colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    cm_path = config.REPORTS_DIR / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)

    print(f"\nReport saved to            {report_path}")
    print(f"Confusion matrix saved to  {cm_path}")
    return acc


if __name__ == "__main__":
    evaluate()
