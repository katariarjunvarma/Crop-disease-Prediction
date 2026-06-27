"""Run inference on a single leaf image."""
from __future__ import annotations

import json

import numpy as np
import tensorflow as tf

from . import config


def _load_class_names():
    if config.CLASS_NAMES_PATH.exists():
        with open(config.CLASS_NAMES_PATH) as f:
            return json.load(f)
    raise FileNotFoundError(
        f"No class names at {config.CLASS_NAMES_PATH}. Train the model first."
    )


def predict(image_path: str):
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model at {config.MODEL_PATH}. Run `python main.py train` first."
        )

    model = tf.keras.models.load_model(config.MODEL_PATH)
    class_names = _load_class_names()

    img = tf.keras.utils.load_img(image_path, target_size=config.IMAGE_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)  # add batch dimension

    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    confidence = float(probs[idx])

    print(f"\nImage:      {image_path}")
    print(f"Prediction: {class_names[idx]}")
    print(f"Confidence: {confidence:.2%}\n")

    print("Class probabilities:")
    for name, p in sorted(zip(class_names, probs), key=lambda x: -x[1]):
        print(f"  {name:<32} {p:.2%}")

    return class_names[idx], confidence


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.predict <path-to-leaf-image>")
        raise SystemExit(1)
    predict(sys.argv[1])
