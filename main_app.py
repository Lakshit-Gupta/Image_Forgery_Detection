import os
import random
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import backend as K
import base64
import numpy as np
from PIL import ImageOps, Image
import streamlit as st

def set_background(image_file):
    """Sets the background of a Streamlit app to an image."""
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

def load_and_preprocess_image(image):
    img = image.convert('RGB')
    img = img.resize((384, 384))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
# Setting seed for reproducibility
random.seed(69)
np.random.seed(69)
tf.random.set_seed(69)

# Custom Focal Loss Function
def focal_loss(alpha=0.5, gamma=3.0):
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, K.floatx())
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

        bce = K.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_factor = alpha * K.pow(1 - p_t, gamma)

        return K.mean(focal_factor * bce)
    return loss

# Set background image (optional)
set_background('img.webp')

# Set the title and header of the Streamlit app
st.title('Plant Disease Classification')
st.header('Please upload an image to check if your plant has a disease.')

# Upload file (for image)
file = st.file_uploader('Upload an image', type=['jpeg', 'jpg', 'png'])

# Load the model with custom_objects
model = load_model('v4.h5', custom_objects={'loss': focal_loss()})

# Function to predict the class of an image (Hardcoded Binary Classification)
def predict_image_class(model, image):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)

    # Hardcoded binary classification: 0 for AU, 1 for TP
    if predictions[0][0] > 0.5:
        predicted_class_name = "TP"
        confidence_score = predictions[0][0]
    else:
        predicted_class_name = "AU"
        confidence_score = 1 - predictions[0][0]

    return predicted_class_name, confidence_score

# Display the uploaded image and the prediction only after clicking the button
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    if st.button('Predict'):
        predicted_class_name, confidence_score = predict_image_class(model, image)

        st.markdown(f"""
            <div style="background-color: black; padding: 20px; border-radius: 10px;">
                <h2 style="color: white;">Predicted Class: {predicted_class_name}</h2>
                <h3 style="color: white;">Confidence: {confidence_score * 100:.2f}%</h3>
            </div>
        """, unsafe_allow_html=True)