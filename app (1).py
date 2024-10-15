import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import joblib

# Load the trained model
model = tf.keras.models.load_model('cnn_model.keras')  # Replace with your model path

# Load your OneHotEncoder
one_hot_encoder = joblib.load('one_hot_encoder.pkl')  # Replace with your one-hot encoder path

def predict_image_class(model, img_path, one_hot_encoder):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the class probabilities
    probabilities = model.predict(img_array)

    # Get the predicted class index (highest probability)
    predicted_class_index = np.argmax(probabilities, axis=1)[0]

    # Convert the predicted index back to the original class label using OneHotEncoder
    predicted_class_one_hot = np.zeros((1, one_hot_encoder.categories_[0].shape[0]))
    predicted_class_one_hot[0][predicted_class_index] = 1

    predicted_class_label = one_hot_encoder.inverse_transform(predicted_class_one_hot)[0]

    # Prepare probabilities for each class label
    class_probs = {one_hot_encoder.categories_[0][i]: prob for i, prob in enumerate(probabilities[0])}

    return predicted_class_label, class_probs

# Streamlit app
st.title("Image Classification with CNN")
st.write("Upload an image to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    img = image.load_img(uploaded_file, target_size=(128, 128))
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")

    # Save the uploaded image temporarily for prediction
    temp_file_path = "temp_image.jpg"  # Save it temporarily
    img.save(temp_file_path)

    # Make prediction
    predicted_label, probabilities = predict_image_class(model, temp_file_path, one_hot_encoder)

    # Show results
    st.write(f"Predicted Label: {predicted_label}")
    st.write("Probabilities:")
    
    for label, prob in probabilities.items():
        st.write(f"{label}: {prob:.4f}")