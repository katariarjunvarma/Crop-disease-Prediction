"""Streamlit web UI for the Crop Disease Prediction model.

Upload a leaf image and get the predicted disease class with confidence scores.
This is a thin presentation layer on top of the trained Keras model — the CLI
pipeline in `main.py` is still the source of truth for training/evaluation.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from src import config

st.set_page_config(page_title="Crop Disease Prediction", page_icon="🌱", layout="centered")


@st.cache_resource(show_spinner="Loading model...")
def load_model_and_classes():
    # Self-bootstrap: on a fresh checkout / cloud deploy there is no trained
    # model (it is git-ignored), so generate the demo dataset and train once.
    if not config.MODEL_PATH.exists() or not config.CLASS_NAMES_PATH.exists():
        with st.spinner("First run: generating demo data and training the model..."):
            from scripts.generate_demo_data import generate
            from src.train import train

            generate()
            train()
    model = tf.keras.models.load_model(config.MODEL_PATH)
    with open(config.CLASS_NAMES_PATH) as f:
        class_names = json.load(f)
    return model, class_names


def predict(model, pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize(config.IMAGE_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)  # add batch dimension
    return model.predict(arr, verbose=0)[0]


def _pretty(name: str) -> str:
    return name.replace("___", " — ").replace("_", " ")


# --- Header ----------------------------------------------------------------
st.title("🌱 Crop Disease Prediction")
st.caption(
    "Classify plant diseases from a leaf image using a TensorFlow/Keras CNN. "
    "Built during an AI & ML internship (AICTE – Edunet Foundation)."
)

model, class_names = load_model_and_classes()

with st.sidebar:
    st.header("About")
    st.write(
        "This demo ships with a small **synthetic** dataset so it runs instantly. "
        "Swap in the real "
        "[PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) "
        "and retrain for production use."
    )
    st.write("**Classes the model knows:**")
    for c in class_names:
        st.write(f"- {_pretty(c)}")

# --- Main interaction ------------------------------------------------------
uploaded = st.file_uploader(
    "Upload a leaf image", type=["jpg", "jpeg", "png"], accept_multiple_files=False
)

if uploaded is None:
    st.info("⬆️ Upload a leaf image to get a prediction.")
    st.stop()

image = Image.open(uploaded)
probs = predict(model, image)
top = int(np.argmax(probs))

col_img, col_pred = st.columns([1, 1])
with col_img:
    st.image(image, caption="Uploaded leaf", use_container_width=True)
with col_pred:
    st.metric("Prediction", _pretty(class_names[top]))
    st.metric("Confidence", f"{probs[top]:.1%}")

st.subheader("Class probabilities")
for name, p in sorted(zip(class_names, probs), key=lambda x: -x[1]):
    st.write(_pretty(name))
    st.progress(float(p))
