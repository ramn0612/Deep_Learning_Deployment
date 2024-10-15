# CNN Model Deployment with Streamlit

This project involves building and deploying a Convolutional Neural Network (CNN) using Streamlit. The model makes predictions based on user input, showcasing deep learning in action within a simple and interactive web interface.

#### Features

- **Image Upload**: Users can upload images for model prediction.
- **CNN Model**: Built using TensorFlow/Keras for image classification tasks.
- **Model Deployment**: Streamlit is used for deploying the model in a web-based interface.
- **Real-time Prediction**: Predictions are made and displayed immediately after image upload.
- **User-Friendly UI**: A clean and intuitive interface built with Streamlit.

#### Tech Stack

- **Frontend**: Streamlit for web interface
- **Backend**: CNN model using TensorFlow/Keras
- **Programming Language**: Python

#### Libraries Used

- **TensorFlow/Keras**: For building and training the CNN model
- **Streamlit**: For deploying the app and creating a user interface
- **NumPy**: For numerical operations
- **Pillow**: For image processing
- **Matplotlib**: For visualizing predictions
- **Scikit-learn**: For splitting the dataset and other utilities

#### Step-by-Step Guide

### Step 1: Clone the repository

```bash
git clone https://github.com/ramn0612/Deep_Learning_Deployment.git
cd Deep_Learning_Deployment
```

### Step 2: Set up a Virtual Environment

```bash
conda create -n cnn-venv python=3.11
```

### Step 3: Activate the virtual environment

```bash
conda activate cnn-venv
```

### Step 4: Install the Requirements

```bash
pip install -r requirements.txt
```

### Step 5: Run the Streamlit app

```bash
streamlit run app.py
```

### Step 6: Upload Test Image

Once the app opens in your browser,
Click on the "Browse files" button to upload the file.
(I have also provided a sample Image in the repo folder for testing purposes)

### Step 6: Check The result

As soon as you upload the Image the model will give its predictions.
Please scroll down to see the "Predicted Label:"
In addition to that You can also check the probability scores of all the classes

---

#### Future Enhancements

- **Support for additional image formats**.
- **Integration with cloud storage for large-scale datasets**.
- **Improved model accuracy using more advanced CNN architectures**.

---

#### Contributions

Feel free to contribute by opening issues or submitting pull requests.
