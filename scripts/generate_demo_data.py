"""Generate a small synthetic leaf-image dataset so the pipeline runs end-to-end.

This is **only** for demonstration/testing. Each class is given a distinct
colour-and-texture signature (a healthy green leaf, a brown-spotted leaf, a
yellowing leaf) so a small CNN can learn to separate them in seconds. To train
on a real problem, delete the `data/` folder and drop in the PlantVillage
dataset instead (see the README).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

# Allow running both as a module and as a plain script.
try:
    from src import config
except ModuleNotFoundError:  # pragma: no cover - path shim for direct execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src import config


# class name -> (base RGB colour, spot RGB colour)
CLASSES = {
    "Healthy": ((60, 160, 60), None),
    "Bacterial_Spot": ((70, 150, 60), (110, 70, 40)),     # brown spots
    "Leaf_Mold": ((180, 190, 70), (150, 150, 50)),        # yellowing
}

IMAGES_PER_CLASS = 120
IMG_DIM = 128


def _make_leaf(base, spot, rng):
    """Create one noisy leaf-like RGB image as a numpy array."""
    img = np.zeros((IMG_DIM, IMG_DIM, 3), dtype=np.float32)
    for c in range(3):
        img[:, :, c] = base[c]
    # texture noise
    img += rng.normal(0, 18, img.shape)

    # elliptical leaf mask centred in the frame
    yy, xx = np.mgrid[0:IMG_DIM, 0:IMG_DIM]
    cx, cy = IMG_DIM / 2, IMG_DIM / 2
    mask = ((xx - cx) / (IMG_DIM * 0.42)) ** 2 + ((yy - cy) / (IMG_DIM * 0.48)) ** 2 <= 1
    img[~mask] = 235  # light background outside the leaf

    # disease spots
    if spot is not None:
        for _ in range(rng.integers(8, 20)):
            sx, sy = rng.integers(20, IMG_DIM - 20, size=2)
            r = rng.integers(3, 8)
            spot_mask = (xx - sx) ** 2 + (yy - sy) ** 2 <= r**2
            spot_mask &= mask
            for c in range(3):
                img[:, :, c][spot_mask] = spot[c]

    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def generate():
    rng = np.random.default_rng(config.SEED)
    print(f"Generating synthetic demo dataset in {config.DATA_DIR} ...")
    for name, (base, spot) in CLASSES.items():
        class_dir = config.DATA_DIR / name
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(IMAGES_PER_CLASS):
            _make_leaf(base, spot, rng).save(class_dir / f"{name.lower()}_{i:03d}.jpg")
        print(f"  {name}: {IMAGES_PER_CLASS} images")
    print("Done. Now run:  python main.py train")


if __name__ == "__main__":
    generate()
