import os
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

MODEL_PATH = "cat_dog_model.h5"
MODEL_FILE_ID = "1vjQkv8Wplq1JtQyFHEBH-G3tg7mZFxT2"  

# Download model from Google Drive if not present
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model. Please wait...")
    import gdown
    gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", MODEL_PATH, quiet=False)

# Load the model
model = load_model(MODEL_PATH)


# Load your trained model
# model = load_model('cat_dog_model.h5')

st.title("Cats vs Dogs Classifier")
uploaded_file = st.file_uploader("Upload an image of a cat or dog...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((256, 256))
    st.image(img, caption='Uploaded Image', use_column_width=True)

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = "Dog" if prediction > 0.5 else "Cat"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.write(f"Prediction: **{label}** ({confidence * 100:.2f}% confidence)")
