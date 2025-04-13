import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import gdown

# Load models
@st.cache_resource
def load_models():
    # Download from Google Drive if not already present
    if not os.path.exists("resnet_model.h5"):
        gdown.download("https://drive.google.com/uc?id=1oIAAZvRFtkZNu7MB9CGnWkttZDs8iz4U", "resnet_model.h5", quiet=False)
    if not os.path.exists("unet_model.h5"):
        gdown.download("https://drive.google.com/uc?id=1zJym8WV8dIZthcZ_Ds0TQyILx2hzXHX5", "unet_model.h5", quiet=False)

    resnet_model = tf.keras.models.load_model('resnet_model.h5')
    unet_model = tf.keras.models.load_model('unet_model.h5')
    return resnet_model, unet_model

# Decode mask

def decode_mask(mask_idx, class_rgb_values):
    rgb_mask = np.zeros((*mask_idx.shape, 3), dtype=np.uint8)
    for i, color in enumerate(class_rgb_values):
        rgb_mask[mask_idx == i] = color
    return rgb_mask

# Predict and visualize

def predict(image, resnet_model, unet_model, class_names, class_rgb_values):
    image_resized = image.resize((256, 256))
    image_array = np.expand_dims(np.array(image_resized), axis=0)

    # Classification
    pred_class = resnet_model.predict(image_array)
    class_idx = np.argmax(pred_class)
    class_name = class_names[class_idx]

    # Segmentation
    pred_mask = unet_model.predict(image_array)
    pred_mask = np.argmax(pred_mask[0], axis=-1)
    pred_mask_rgb = decode_mask(pred_mask, class_rgb_values)

    return class_name, pred_mask_rgb

# Streamlit App

st.set_page_config(page_title="LULC Segmentation", layout="wide")
st.title("🌍 Land Use & Land Cover Classification and Segmentation")

uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='🛰️ Uploaded Satellite Image', use_column_width=True)

    # Load models and class data
    resnet_model, unet_model = load_models()
    class_df = pd.read_csv('class_dict.csv')
    class_names = class_df['name'].tolist()
    class_rgb_values = class_df[['r', 'g', 'b']].values

    st.write("🔄 Predicting...")
    predicted_class, predicted_mask = predict(image, resnet_model, unet_model, class_names, class_rgb_values)

    col1, col2 = st.columns(2)
    col1.markdown(f"### 🏷️ Predicted Class: {predicted_class}")
    col2.image(predicted_mask, caption='🎨 Predicted Segmentation Mask', use_column_width=True)
