"""Streamlit web UI for the Crop Disease Prediction model.

Upload (or drag-drop) a leaf image — or pick one of the bundled samples — and the
app predicts the most likely disease with a confidence score, an agronomy-style
description, and management advice. This is a thin presentation layer on top of
the trained Keras model; `main.py` remains the source of truth for training.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import json

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from PIL import Image

from src import config

st.set_page_config(
    page_title="Crop Disease Prediction",
    page_icon="🌱",
    layout="centered",
)

# Predictions below this confidence are flagged as uncertain.
CONFIDENCE_THRESHOLD = 0.60

# Bundled, version-controlled sample leaves so the demo works on a fresh deploy.
# Each category is its own folder holding ~20 example images.
SAMPLES_DIR = config.PROJECT_ROOT / "samples"
SAMPLE_CATEGORIES = {
    "🌿 Healthy": SAMPLES_DIR / "healthy",
    "🍂 Leaf Mold": SAMPLES_DIR / "leaf_mold",
    "🦠 Diseased (Bacterial Spot)": SAMPLES_DIR / "bacterial_spot",
}


def _list_samples(folder):
    files = []
    if folder.exists():
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            files.extend(sorted(folder.glob(ext)))
    return files

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
# Sidebar — about + readable model summary (no settings)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "Upload a leaf image (or try a sample) and the model predicts the most "
        "likely disease, how confident it is, and how to manage it."
    )
    st.write(
        "This demo ships with a small **synthetic** dataset so it runs instantly. "
        "Swap in the real "
        "[PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) "
        "and retrain for real-world accuracy."
    )

    st.divider()
    st.subheader("🧠 How the model works")
    st.write(
        "A convolutional neural network (CNN) scans the leaf image and scores how "
        "closely it matches each disease it has learned. The highest score becomes "
        "the prediction."
    )
    st.markdown(
        f"""
| | |
|---|---|
| **Model** | CNN (3 conv blocks + dense head) |
| **Image input** | {config.IMAGE_SIZE[0]}×{config.IMAGE_SIZE[1]} pixels (RGB) |
| **Learned weights** | {model.count_params():,} |
| **Conditions** | {len(class_names)} ({', '.join(_pretty(c) for c in class_names)}) |
"""
    )

# ---------------------------------------------------------------------------
# Input — upload (drag & drop) or pick a bundled sample
# ---------------------------------------------------------------------------
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

st.subheader("📤 Upload a leaf image")
uploaded = st.file_uploader(
    "Drag & drop an image here, or browse your files",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
    key=f"uploader_{st.session_state.uploader_key}",
)

st.subheader("🍃 Samples")
st.caption("Open a folder and click **Use** under any leaf to analyse it.")
tabs = st.tabs(list(SAMPLE_CATEGORIES.keys()))
for tab, (label, folder) in zip(tabs, SAMPLE_CATEGORIES.items()):
    with tab:
        images = _list_samples(folder)
        if not images:
            st.write("No sample images found in this folder.")
            continue
        st.caption(f"{len(images)} sample images")
        ncols = 4
        for start in range(0, len(images), ncols):
            row_cols = st.columns(ncols)
            for col, path in zip(row_cols, images[start:start + ncols]):
                with col:
                    st.image(str(path), use_container_width=True)
                    if st.button("Use", key=f"use_{folder.name}_{path.name}",
                                 use_container_width=True):
                        st.session_state["sample_path"] = str(path)
                        st.session_state.uploader_key += 1  # clear any upload
                        st.session_state["scroll_to_results"] = True
                        st.rerun()

# Decide the active image: a fresh upload always wins over a stored sample.
image = None
if uploaded is not None:
    image = Image.open(uploaded)
    st.session_state.pop("sample_path", None)
    uid = getattr(uploaded, "file_id", None) or f"{uploaded.name}:{uploaded.size}"
    if st.session_state.get("last_upload") != uid:
        st.session_state["last_upload"] = uid
        st.session_state["scroll_to_results"] = True
elif st.session_state.get("sample_path"):
    image = Image.open(st.session_state["sample_path"])

if image is None:
    st.info("⬆️ Upload a leaf image or pick a sample above to get a prediction.")
    st.stop()

# ---------------------------------------------------------------------------
# Prediction + results
# ---------------------------------------------------------------------------
st.divider()
st.markdown('<div id="results-anchor"></div>', unsafe_allow_html=True)
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

if top_conf < CONFIDENCE_THRESHOLD:
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
st.subheader("Class probabilities")
for name, p in sorted(zip(class_names, probs), key=lambda x: -x[1]):
    st.write(f"{_pretty(name)} — {p:.1%}")
    st.progress(float(p))

# Smooth-scroll to the results when a sample was used or a new image uploaded.
if st.session_state.pop("scroll_to_results", False):
    components.html(
        """
        <script>
          setTimeout(function () {
            const el = window.parent.document.getElementById('results-anchor');
            if (el) { el.scrollIntoView({behavior: 'smooth', block: 'start'}); }
          }, 150);
        </script>
        """,
        height=0,
    )

# Reset / upload-new control
st.divider()
if st.button("🔄 Try another image", use_container_width=True):
    st.session_state.pop("sample_path", None)
    st.session_state.uploader_key += 1
    st.rerun()

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
