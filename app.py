import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = Path(__file__).parent / "melanoma_cnn_model.keras"
CLASSES_PATH = Path(__file__).parent / "class_names.json"
IMAGE_SIZE = (128, 128)  # Must match the input size used when training your CNN.

st.set_page_config(
    page_title="Melanoma Skin Lesion Classifier",
    page_icon="🔬",
    layout="centered",
)

@st.cache_resource
def load_cnn(model_path: str):
    return tf.keras.models.load_model(model_path, compile=False)

@st.cache_data
def load_class_names(classes_path: str):
    with open(classes_path, "r", encoding="utf-8") as f:
        names = json.load(f)
    if not isinstance(names, list) or not names:
        raise ValueError("class_names.json must contain a non-empty JSON list.")
    return names

st.title("🔬 Melanoma Skin Lesion Classifier")
st.write(
    "Upload an image to view the prediction from the trained CNN model. "
    "This is an educational demonstration, not a medical diagnostic tool."
)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Add your trained "
        "`melanoma_cnn_model.keras` file to the same folder as app.py."
    )
    st.stop()

if not CLASSES_PATH.exists():
    st.error(
        "class_names.json not found. Create it from your Kaggle notebook "
        "using the original `class_names` list."
    )
    st.stop()

try:
    model = load_cnn(str(MODEL_PATH))
    class_names = load_class_names(str(CLASSES_PATH))
except Exception as exc:
    st.error(f"Could not load the model or class labels: {exc}")
    st.stop()

st.success("CNN model loaded.")

uploaded_file = st.file_uploader(
    "Choose a skin lesion image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Run prediction", type="primary"):
            resized = image.resize(IMAGE_SIZE)
            image_array = np.asarray(resized, dtype=np.float32)
            batch = np.expand_dims(image_array, axis=0)

            probabilities = np.asarray(model.predict(batch, verbose=0))[0].reshape(-1)

            if len(probabilities) != len(class_names):
                st.error(
                    "The number of model outputs does not match the number "
                    "of labels in class_names.json. Check the original class order."
                )
            else:
                best_index = int(np.argmax(probabilities))
                st.subheader(f"Predicted class: {class_names[best_index]}")
                st.metric("Model score", f"{probabilities[best_index] * 100:.2f}%")

                st.write("Scores by class")
                for label, score in zip(class_names, probabilities):
                    st.write(f"**{label}** — {score * 100:.2f}%")
                    st.progress(float(np.clip(score, 0.0, 1.0)))

    except Exception as exc:
        st.error(f"Could not process this image: {exc}")

st.divider()
st.caption(
    "Important: A model prediction is not a diagnosis and must not be used "
    "to rule out or confirm skin cancer. Please consult a qualified "
    "dermatologist for assessment of any concerning lesion."
)
