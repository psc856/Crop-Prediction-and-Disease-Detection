import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.layers import *
from keras.models import *
import keras.backend as K

# Define custom loss function if you have any
def custom_loss_function(y_true, y_pred):
    # Define your custom loss function logic here
    pass

# Load the model with custom objects dictionary
try:
    MODEL = tf.keras.models.load_model(r"C:\Users\Dell\Desktop\t\potatoes.h5", custom_objects={"custom_loss_function": custom_loss_function})
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Define input shape expected by the model
INPUT_SHAPE = (256, 256, 3)  # Adjust the shape according to your model's input requirements

# Function to preprocess image
def preprocess_image(image):
    # Resize image to match model's expected sizing
    image = image.resize((INPUT_SHAPE[1], INPUT_SHAPE[0]))
    # Convert PIL image to numpy array
    image = np.array(image)
    # Normalize pixel values to range [0, 1]
    image = image / 255.0
    return image

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Function to predict
def predict(image):
    img = preprocess_image(image)
    img_batch = np.expand_dims(img, axis=0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_class, confidence

# Main function
def main():
    st.title('Plant Disease Classifier')
    st.text('Upload an image of a plant to classify its disease.')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Predict'):
            try:
                predicted_class, confidence = predict(image)
                st.write(f"Prediction: {predicted_class}")
                st.write(f"Confidence: {confidence * 100:.2f}%")
            except Exception as e:
                st.error(f"Error predicting: {e}")

if __name__ == "__main__":
    main()
