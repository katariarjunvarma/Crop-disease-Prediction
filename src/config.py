"""Central configuration for the Crop Disease Prediction pipeline.

All paths are resolved relative to the project root so the scripts work no
matter which directory you launch them from.
"""
from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"          # image folders, one sub-folder per class
MODEL_DIR = PROJECT_ROOT / "models"       # saved Keras model + class names
REPORTS_DIR = PROJECT_ROOT / "reports"    # plots and evaluation reports

MODEL_PATH = MODEL_DIR / "crop_disease_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

# --- Image / training hyper-parameters -------------------------------------
IMAGE_SIZE = (128, 128)   # leaf images are resized to this before training
BATCH_SIZE = 32
EPOCHS = 15
VALIDATION_SPLIT = 0.2
SEED = 123
LEARNING_RATE = 1e-3

# Ensure the output directories always exist.
for _d in (DATA_DIR, MODEL_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
