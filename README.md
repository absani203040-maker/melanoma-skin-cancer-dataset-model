# Melanoma Skin Lesion CNN — Streamlit App

A small Streamlit interface for running an already-trained image classification CNN.

## Files required

Place these files in the repository root:

- `app.py`
- `requirements.txt`
- `class_names.json`
- `melanoma_cnn_model.keras` — your trained model downloaded from Kaggle

**The trained `.keras` model is not included in this starter project.** You must add your own model file.

## 1. Export files from Kaggle

After training (or after restoring your saved model), run this in the Kaggle notebook:

```python
import json

model.save("/kaggle/working/melanoma_cnn_model.keras")

with open("/kaggle/working/class_names.json", "w") as f:
    json.dump(class_names, f)

print("Saved model and class names.")
```

Download both files from Kaggle's Output section. Keep the original order of `class_names`; it must match the model's output neurons.

## 2. GitHub

Create a repository, for example `Melanoma-Skin-Cancer-CNN`, and upload the project files. If the model is larger than GitHub's regular file limit, use Git LFS or a suitable model-hosting service rather than committing a large file normally.

## 3. Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 4. Deploy

On Streamlit Community Cloud, create an app from your GitHub repository and set the main file to `app.py`.

## Important compatibility checks

- This app resizes images to `128 x 128`, matching the CNN example.
- The example CNN includes `Rescaling(1./255)` inside the model. This app therefore passes pixel values in the 0–255 range and does not normalize them a second time.
- Confirm the deployed TensorFlow version is compatible with the version used to save your model.
- The class label list must be the exact original `class_names` from training.

## Medical disclaimer

This is an educational project, not a medical device. Model scores are not a diagnosis. Do not use this app to make medical decisions; consult a qualified dermatologist.
