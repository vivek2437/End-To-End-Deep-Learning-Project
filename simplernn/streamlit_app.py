import streamlit as st
import tensorflow as tf
import numpy as np
from simplernn import preprocess_text  # your preprocessing function

# Load the trained TensorFlow model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")  # your saved model
    return model

model = load_model()

st.title("End-to-End Deep Learning Project")
st.write("Enter text below for prediction:")

# Input from user
user_input = st.text_area("Your Input Text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        # Preprocess
        processed_input = preprocess_text(user_input)  # must return suitable input for model
        input_array = np.array([processed_input])  # model expects batch dimension

        # Make prediction
        predictions = model.predict(input_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

        st.success(f"Predicted Class: {predicted_class}")
        st.info(f"Confidence: {confidence*100:.2f}%")
