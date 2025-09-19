"""app

Original file is located at
    https://colab.research.google.com/drive/1Ne6UDH2v93DydbVX_rhFdfWSLwGw0Wl0
"""

import streamlit as st
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="auto",
)

# Function to load the model (cached to improve performance)
@st.cache_resource
def load_keras_model():
    model = load_model('brain_tumor_model.h5')
    return model

# Function to preprocess the uploaded image
def preprocess_image(image):
    # Convert the PIL image to a NumPy array
    img_array = np.array(image)
    # Ensure it's a 3-channel image
    if img_array.ndim == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    # Resize the image to the model's expected input size
    resized_img = cv2.resize(img_array, (128, 128))
    # Normalize the pixel values
    normalized_img = resized_img / 255.0
    # Expand dimensions to create a batch of 1
    batched_img = np.expand_dims(normalized_img, axis=0)
    return batched_img

# --- Main App Interface ---

st.title("Brain Tumor MRI Classifier 🧠")
st.write("Upload an MRI scan of a brain, and the model will predict whether a tumor is present.")

# Load the trained model
model = load_keras_model()

# File uploader widget
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Scan.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image and make a prediction
    processed_image = preprocess_image(image)
    prediction_score = model.predict(processed_image)[0][0]

    # Define class names
    class_names = ['No Tumor', 'Tumor Detected']

    # Display the prediction result
    if prediction_score > 0.5:
        st.error(f"**Result: {class_names[1]}** (Confidence: {prediction_score*100:.2f}%)")
    else:
        st.success(f"**Result: {class_names[0]}** (Confidence: {(1-prediction_score)*100:.2f}%)")