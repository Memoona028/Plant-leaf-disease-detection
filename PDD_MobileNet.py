# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 22:43:56 2024

@author: DELL
"""
#streamlit run "C:\Users\DELL\Downloads\files\PDD_MobileNet.py"

import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf
# Load your trained model
MODEL_PATH = r"C:\Users\DELL\Downloads\plant_models\plant_leaf_disease_detection_MobileNet.h5"
model = load_model(MODEL_PATH, custom_objects={'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D})

def model_predict(img):
    new_arr = cv2.resize(img, (224, 224))  # Resize the image to 100x100
    new_arr = np.array(new_arr / 255)
    new_arr = new_arr.reshape(-1, 224, 224, 3)
    preds = model.predict(new_arr)
    return preds

def main():
    st.title("Plant Disease Classifier")

    # Displaying names below the title
    st.markdown("""
        ### ANN Project by :
        - Memoona Basharat(21-CS-97)
        - Areeba Nazim (21-CS-79)
        - Sara Ahmed (21-CS-01)
        ### Submitted to
        - Sir Munwar Iqbal
        """
    )

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert the file-like object to a NumPy array
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)  # 1 means load color image

        # Resize the image to a smaller size for better visualization
        resized_image = cv2.resize(image, (100, 100))  

        st.image(resized_image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        preds = model_predict(image)
        pred_class = np.argmax(preds)
        
        CATEGORIES = ['Apple___Apple_scab',
         'Apple___Black_rot',
         'Apple___Cedar_apple_rust',
         'Apple___healthy',
         'Blueberry___healthy',
         'Cherry_(including_sour)___Powdery_mildew',
         'Cherry_(including_sour)___healthy',
         'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
         'Corn_(maize)___Common_rust_',
         'Corn_(maize)___Northern_Leaf_Blight',
         'Corn_(maize)___healthy',
         'Grape___Black_rot',
         'Grape___Esca_(Black_Measles)',
         'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
         'Grape___healthy',
         'Orange___Haunglongbing_(Citrus_greening)',
         'Peach___Bacterial_spot',
         'Peach___healthy',
         'Pepper,_bell___Bacterial_spot',
         'Pepper,_bell___healthy',
         'Potato___Early_blight',
         'Potato___Late_blight',
         'Potato___healthy',
         'Raspberry___healthy',
         'Soybean___healthy',
         'Squash___Powdery_mildew',
         'Strawberry___Leaf_scorch',
         'Strawberry___healthy',
         'Tomato___Bacterial_spot',
         'Tomato___Early_blight',
         'Tomato___Late_blight',
         'Tomato___Leaf_Mold',
         'Tomato___Septoria_leaf_spot',
         'Tomato___Spider_mites Two-spotted_spider_mite',
         'Tomato___Target_Spot',
         'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
         'Tomato___Tomato_mosaic_virus',
         'Tomato___healthy']
        
        # Get the percentage of prediction
        percentage = preds[0][pred_class] * 100
        # Highlighted box around the predicted disease with percentage
        st.info(f"Prediction: **{CATEGORIES[pred_class]}** (Confidence: {percentage:.2f}%)")

if __name__ == "__main__":
    main()
def main():
    st.title("Plant Disease Classifier")

    # Displaying names below the title
    st.markdown("""
        ### ANN Project by :
        - Memoona Basharat(21-CS-97)
        - Areeba Nazim (21-CS-79)
        - Sara Ahmed (21-CS-01)
        ### Submitted to
        - Sir Munwar Iqbal
        """
    )

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert the file-like object to a NumPy array
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)  # 1 means load color image

        # Resize the image to a smaller size for better visualization
        resized_image = cv2.resize(image, (200, 200))  

        st.image(resized_image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        preds = model_predict(image)
        pred_class = np.argmax(preds)
        
        CATEGORIES = ['Apple___Apple_scab',
         'Apple___Black_rot',
         'Apple___Cedar_apple_rust',
         'Apple___healthy',
         'Blueberry___healthy',
         'Cherry_(including_sour)___Powdery_mildew',
         'Cherry_(including_sour)___healthy',
         'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
         'Corn_(maize)___Common_rust_',
         'Corn_(maize)___Northern_Leaf_Blight',
         'Corn_(maize)___healthy',
         'Grape___Black_rot',
         'Grape___Esca_(Black_Measles)',
         'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
         'Grape___healthy',
         'Orange___Haunglongbing_(Citrus_greening)',
         'Peach___Bacterial_spot',
         'Peach___healthy',
         'Pepper,_bell___Bacterial_spot',
         'Pepper,_bell___healthy',
         'Potato___Early_blight',
         'Potato___Late_blight',
         'Potato___healthy',
         'Raspberry___healthy',
         'Soybean___healthy',
         'Squash___Powdery_mildew',
         'Strawberry___Leaf_scorch',
         'Strawberry___healthy',
         'Tomato___Bacterial_spot',
         'Tomato___Early_blight',
         'Tomato___Late_blight',
         'Tomato___Leaf_Mold',
         'Tomato___Septoria_leaf_spot',
         'Tomato___Spider_mites Two-spotted_spider_mite',
         'Tomato___Target_Spot',
         'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
         'Tomato___Tomato_mosaic_virus',
         'Tomato___healthy']
        
        # Get the percentage of prediction
        percentage = preds[0][pred_class] * 100
        # Highlighted box around the predicted disease with percentage
        st.info(f"Prediction: **{CATEGORIES[pred_class]}** (Confidence: {percentage:.2f}%)")