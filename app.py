"""Streamlit web UI for the Crop Disease Prediction model.

Upload (or pick a sample) leaf image and get the predicted disease class with a
confidence score, an agronomy-style description, and management advice. This is
a thin presentation layer on top of the trained Keras model — the CLI pipeline
in `main.py` remains the source of truth for training/evaluation.

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

st.set_page_config(
    page_title="Crop Disease Prediction",
    page_icon="🌱",
    layout="centered",
    menu_items={"About": "Crop Disease Prediction — TensorFlow/Keras + scikit-learn. "
                "Built by Katari Arjun Varma."},
)

# ---------------------------------------------------------------------------
# Domain knowledge: short, human-written notes for each disease. Matching is
# keyword-based so it also works for the real PlantVillage class names
# (e.g. "Tomato___Bacterial_spot", "Potato___Late_blight").
# ---------------------------------------------------------------------------
DISEASE_LIBRARY = [
    {
        "keywords": ["healthy"],
        "label": "Healthy",
        "severity": ("Healthy", "green"),
        "summary": "The leaf looks healthy — uniform colour with no lesions, "
        "spotting, or mould.",
        "symptoms": "Even green pigmentation, intact margins, no discolouration.",
        "management": "No action needed. Keep monitoring and maintain good field "
        "hygiene, balanced nutrition, and proper spacing for airflow.",
    },
    {
        "keywords": ["bacterial", "spot"],
        "label": "Bacterial Spot",
        "severity": ("Moderate–High", "orange"),
        "summary": "A bacterial infection (*Xanthomonas* spp.) that thrives in "
        "warm, wet, humid conditions and spreads via water splash.",
        "symptoms": "Small water-soaked spots that enlarge into dark brown/black "
        "lesions, often with a yellow halo; severe cases cause defoliation.",
        "management": "Use certified disease-free seed, rotate crops, avoid "
        "overhead irrigation, and apply copper-based bactericides preventively.",
    },
    {
        "keywords": ["mold", "mould"],
        "label": "Leaf Mold",
        "severity": ("Moderate", "orange"),
        "summary": "A fungal disease (*Passalora fulva*) common in greenhouses "
        "and humid climates with poor ventilation.",
        "symptoms": "Pale-yellow blotches on the upper surface and olive-green to "
        "grey velvety mould on the underside of the leaf.",
        "management": "Lower humidity, improve ventilation and spacing, remove "
        "infected leaves, grow resistant varieties, and apply fungicide if needed.",
    },
    {
        "keywords": ["blight"],
        "label": "Blight",
        "severity": ("High", "red"),
        "summary": "A fast-spreading fungal disease (early/late blight) that can "
        "destroy foliage and fruit in cool, wet weather.",
        "symptoms": "Concentric brown lesions or large dark water-soaked patches; "
        "rapid leaf collapse in late blight.",
        "management": "Remove infected debris, ensure drainage, rotate crops, and "
        "apply protectant fungicides on a schedule during high-risk weather.",
    },
    {
        "keywords": ["rust"],
        "label": "Rust",
        "severity": ("Moderate", "orange"),
        "summary": "A fungal disease producing rust-coloured pustules, favoured by "
        "moisture on the leaf surface.",
        "symptoms": "Orange-to-brown powdery pustules on the underside of leaves; "
        "yellowing and early leaf drop.",
        "management": "Plant resistant cultivars, avoid leaf wetness, and apply "
        "fungicides when pustules first appear.",
    },
    {
        "keywords": ["mildew"],
        "label": "Powdery Mildew",
        "severity": ("Moderate", "orange"),
        "summary": "A fungal disease that forms a white powdery coating, common in "
        "dry days with humid nights.",
        "symptoms": "White-to-grey powdery patches on leaf surfaces; curling and "
        "yellowing of affected tissue.",
        "management": "Increase airflow, avoid excess nitrogen, and use sulphur or "
        "potassium-bicarbonate sprays.",
    },
    {
        "keywords": ["virus", "mosaic", "curl"],
        "label": "Viral Infection",
        "severity": ("High", "red"),
        "summary": "A viral disease (e.g. mosaic / leaf-curl viruses) usually "
        "spread by insect vectors such as whiteflies and aphids.",
        "symptoms": "Mottled light/dark mosaic patterns, leaf curling, stunting, "
        "and distorted growth.",
        "management": "There is no cure — remove infected plants, control insect "
        "vectors, and use virus-resistant varieties and clean tools.",
    },
]

GENERIC_INFO = {
    "label": "Disease detected",
    "severity": ("Review", "orange"),
    "summary": "The model has matched this leaf to a known disease class.",
    "symptoms": "Refer to a local agronomy guide for class-specific symptoms.",
    "management": "Isolate affected plants and consult an extension officer for "
    "a treatment plan.",
}


def get_disease_info(class_name: str) -> dict:
    name = class_name.lower()
    for entry in DISEASE_LIBRARY:
        if any(k in name for k in entry["keywords"]):
            return entry
    return GENERIC_INFO


# ---------------------------------------------------------------------------
# Model loading + inference
# ---------------------------------------------------------------------------
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
    return name.replace("___", " — ").replace("_", " ").title()


def _sample_image(class_name: str):
    """Return the path of the first demo image for a class, if available."""
    folder = config.DATA_DIR / class_name
    if folder.exists():
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            files = sorted(folder.glob(ext))
            if files:
                return files[0]
    return None


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🌱 Crop Disease Prediction")
st.caption(
    "Detect plant diseases from a single leaf image using a TensorFlow/Keras "
    "convolutional neural network. Built during an AI & ML internship "
    "(AICTE – Edunet Foundation)."
)

model, class_names = load_model_and_classes()

# ---------------------------------------------------------------------------
# Sidebar — about, model details, and tunable parameters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "Upload a leaf image and the model predicts the most likely disease "
        "along with a confidence score and management advice."
    )
    st.write(
        "This demo ships with a small **synthetic** dataset so it runs instantly. "
        "Swap in the real "
        "[PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) "
        "and retrain for real-world accuracy."
    )

    st.divider()
    st.subheader("⚙️ Settings")
    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.50,
        max_value=0.95,
        value=0.60,
        step=0.05,
        help="Predictions below this confidence are flagged as uncertain.",
    )
    show_all_probs = st.checkbox("Show all class probabilities", value=True)

    st.divider()
    with st.expander("🧠 Model details"):
        st.write(f"**Architecture:** CNN (3 conv blocks + dense head)")
        st.write(f"**Input size:** {config.IMAGE_SIZE[0]}×{config.IMAGE_SIZE[1]} RGB")
        st.write(f"**Trainable parameters:** {model.count_params():,}")
        st.write(f"**Classes ({len(class_names)}):**")
        for c in class_names:
            st.write(f"- {_pretty(c)}")

# ---------------------------------------------------------------------------
# Input — upload or pick a sample
# ---------------------------------------------------------------------------
uploaded = st.file_uploader(
    "Upload a leaf image", type=["jpg", "jpeg", "png"], accept_multiple_files=False
)

st.write("…or try a sample leaf:")
sample_cols = st.columns(len(class_names))
for col, c in zip(sample_cols, class_names):
    if col.button(_pretty(c), use_container_width=True):
        path = _sample_image(c)
        if path:
            st.session_state["sample_path"] = str(path)

# Decide the active image: a fresh upload always wins over a stored sample.
image = None
if uploaded is not None:
    image = Image.open(uploaded)
    st.session_state.pop("sample_path", None)
elif st.session_state.get("sample_path"):
    image = Image.open(st.session_state["sample_path"])

if image is None:
    st.info("⬆️ Upload a leaf image or pick a sample above to get a prediction.")
    st.stop()

# ---------------------------------------------------------------------------
# Prediction + results
# ---------------------------------------------------------------------------
probs = predict(model, image)
top = int(np.argmax(probs))
top_conf = float(probs[top])
info = get_disease_info(class_names[top])
sev_text, sev_color = info["severity"]

col_img, col_pred = st.columns([1, 1])
with col_img:
    st.image(image, caption="Analysed leaf", use_container_width=True)
with col_pred:
    st.metric("Prediction", _pretty(class_names[top]))
    st.metric("Confidence", f"{top_conf:.1%}")
    st.markdown(f"**Severity:** :{sev_color}[{sev_text}]")

if top_conf < confidence_threshold:
    st.warning(
        f"⚠️ Low confidence ({top_conf:.1%}). The model is unsure — try a "
        "clearer, well-lit photo of a single leaf for a more reliable result."
    )

# Disease description card
st.subheader(f"📋 {info['label']}")
st.write(info["summary"])
desc_left, desc_right = st.columns(2)
with desc_left:
    st.markdown("**Typical symptoms**")
    st.write(info["symptoms"])
with desc_right:
    st.markdown("**Management**")
    st.write(info["management"])

# Probability breakdown
if show_all_probs:
    st.subheader("Class probabilities")
    for name, p in sorted(zip(class_names, probs), key=lambda x: -x[1]):
        st.write(f"{_pretty(name)} — {p:.1%}")
        st.progress(float(p))

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Built by **Katari Arjun Varma** · "
    "[GitHub](https://github.com/katariarjunvarma) · "
    "[LinkedIn](https://linkedin.com/in/karjunvarma)  \n"
    "Tech stack: Python · TensorFlow / Keras · scikit-learn · Streamlit  \n"
    "_Educational project — not a substitute for professional agronomic diagnosis._"
)
