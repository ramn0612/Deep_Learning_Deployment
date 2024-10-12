# app/app.py
import streamlit as st
import pickle
import numpy as np
from PIL import Image

# Load your models
with open('../models/dnn_model.pkl', 'rb') as file:
    dnn_model = pickle.load(file)

with open('../models/adaboost_model.pkl', 'rb') as file:
    adaboost_model = pickle.load(file)

with open('../models/cnn_model.pkl', 'rb') as file:
    cnn_model = pickle.load(file)

with open('../models/dnn_image_model.pkl', 'rb') as file:
    dnn_image_model = pickle.load(file)

st.title("Parking Prediction and Image Labeling")

# Section for Parking Prediction
st.header("Predict Parking Availability")
business_info = st.text_input("Enter Business Information")

if st.button("Predict Parking"):
    # Dummy example: Replace with your actual input processing
    prediction = dnn_model.predict([business_info])  # Adjust input format
    st.write("Parking Available:", prediction[0])

# Section for Image Label Prediction
st.header("Predict Image Label")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    if st.button("Predict Label"):
        # Preprocess the image if needed (resize, normalize, etc.)
        image_array = np.array(image)  # Adjust as per your model requirements
        image_prediction = cnn_model.predict(image_array)  # Adjust input format
        st.write("Predicted Label:", image_prediction[0])

# Display model interpretations if required
# st.write("Interpretation of Output")
# ... (Add code for interpretations if applicable)
