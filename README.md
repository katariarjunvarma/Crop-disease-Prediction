# 🌱 Crop Disease Prediction

A deep-learning system that classifies plant diseases from leaf images. Built
with **Python, TensorFlow/Keras, and scikit-learn**, it implements a complete
machine-learning pipeline — data preprocessing, feature extraction with a
convolutional neural network, model training, and performance evaluation.

> Developed during an **AI & ML Internship (AICTE – Edunet Foundation)**.

---

## Features

- **End-to-end ML pipeline** — preprocessing → CNN feature extraction → training → evaluation → inference.
- **TensorFlow/Keras CNN** with on-the-fly data augmentation (flip / rotate / zoom) for robustness.
- **scikit-learn evaluation** — accuracy, per-class precision/recall/F1, and a confusion-matrix plot.
- **Single-image prediction CLI** that outputs the predicted disease and class probabilities.
- **Streamlit web app** — upload a leaf image and see the prediction with confidence bars.
- **Runs out of the box** on a built-in synthetic demo dataset; swap in the real
  [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) dataset for production use.

---

## Project structure

```
Crop-disease-Prediction/
├── main.py                     # CLI entry point (generate-demo / train / evaluate / predict)
├── app.py                      # Streamlit web app (upload a leaf -> prediction)
├── requirements.txt
├── src/
│   ├── config.py               # paths & hyper-parameters
│   ├── data.py                 # data preprocessing -> tf.data pipelines
│   ├── model.py                # CNN architecture (feature extractor + classifier head)
│   ├── train.py                # training pipeline
│   ├── evaluate.py             # scikit-learn performance evaluation
│   └── predict.py              # single-image inference
├── scripts/
│   └── generate_demo_data.py   # synthetic demo dataset generator
├── data/                       # image folders, one per class (git-ignored)
├── models/                     # saved model + class names (git-ignored)
└── reports/                    # training curves, confusion matrix, report (git-ignored)
```

---

## Setup

```bash
# 1. (recommended) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt
```

> **Note:** TensorFlow currently supports Python 3.9–3.12. If you are on
> Python 3.13, create the virtual environment with a 3.12 interpreter.

---

## Quick start (demo dataset)

```bash
python main.py generate-demo     # create a small synthetic dataset under data/
python main.py train             # train the CNN
python main.py evaluate          # print + save the scikit-learn report
python main.py predict data/Healthy/healthy_000.jpg
```

Artifacts are written to `models/` (trained model + class names) and `reports/`
(training-history plot, confusion matrix, classification report).

---

## Web app

Launch the Streamlit interface and classify leaves from your browser:

```bash
streamlit run app.py
```

On first run (or a fresh deploy) the app auto-generates the demo dataset and
trains the model, then lets you upload a leaf image and shows the predicted
disease with per-class confidence bars. Deployable for free on
[Streamlit Community Cloud](https://streamlit.io/cloud).

---

## Using the real PlantVillage dataset

1. Download the dataset from
   [Kaggle – PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset).
2. Delete the demo `data/` folder and replace it with the dataset's class
   sub-folders, e.g.:

   ```
   data/
       Tomato___Healthy/
       Tomato___Bacterial_spot/
       Potato___Early_blight/
       ...
   ```
3. Run `python main.py train` followed by `python main.py evaluate`.

The pipeline auto-detects the class names from the folder structure, so no code
changes are needed.

---

## How it works

1. **Preprocessing** (`src/data.py`) — leaf images are loaded, resized to
   128×128, batched, and split into training/validation sets using
   `tf.keras.utils.image_dataset_from_directory`.
2. **Feature extraction + model** (`src/model.py`) — stacked Conv/MaxPool blocks
   learn hierarchical visual features; a dense head with dropout performs the
   final classification. Pixels are rescaled to `[0, 1]` and augmented at train time.
3. **Training** (`src/train.py`) — Adam optimizer, sparse categorical
   cross-entropy; saves the model and a training-history plot.
4. **Evaluation** (`src/evaluate.py`) — runs the model over the held-out set and
   uses scikit-learn for the accuracy score, classification report, and confusion matrix.
5. **Inference** (`src/predict.py`) — classifies a single leaf image and reports
   the predicted disease with confidence scores.

---

## Tech stack

`Python` · `TensorFlow / Keras` · `scikit-learn` · `NumPy` · `Pillow` · `Matplotlib`
