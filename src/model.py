"""CNN model definition for leaf-disease classification.

A compact convolutional neural network acts as the feature extractor: the
stacked Conv/Pool blocks learn increasingly abstract visual features from the
leaf images, and the dense head maps those features to disease classes.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from . import config


def build_model(num_classes: int, image_size=config.IMAGE_SIZE) -> keras.Model:
    input_shape = (*image_size, 3)

    # On-the-fly augmentation makes the model robust to orientation/zoom changes.
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(x)        # normalise pixel values to [0, 1]

    # --- Convolutional feature extractor -----------------------------------
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)

    # --- Classification head ------------------------------------------------
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="crop_disease_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(config.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
