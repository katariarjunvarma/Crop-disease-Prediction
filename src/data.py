"""Data preprocessing: load leaf images from disk into tf.data pipelines.

The dataset is expected to follow the standard image-folder layout, which is
exactly how the PlantVillage dataset is organised::

    data/
        Tomato___Healthy/
            img1.jpg
            ...
        Tomato___Bacterial_spot/
            ...

Each sub-folder name is treated as a class label.
"""
from __future__ import annotations

import tensorflow as tf

from . import config


def _has_images(data_dir) -> bool:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
    return any(
        p.suffix.lower() in exts for p in data_dir.rglob("*") if p.is_file()
    )


def load_datasets(data_dir=config.DATA_DIR):
    """Return ``(train_ds, val_ds, class_names)``.

    Images are split into a training and a held-out validation set. The
    validation set is later used for performance evaluation.
    """
    if not _has_images(data_dir):
        raise FileNotFoundError(
            f"No images found under {data_dir}. Generate the demo dataset with "
            "`python main.py generate-demo` or place the PlantVillage dataset "
            "there (one sub-folder per class)."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=config.VALIDATION_SPLIT,
        subset="training",
        seed=config.SEED,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="int",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=config.VALIDATION_SPLIT,
        subset="validation",
        seed=config.SEED,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="int",
    )

    class_names = train_ds.class_names

    # Cache + prefetch for a faster, smoother input pipeline.
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)

    return train_ds, val_ds, class_names
