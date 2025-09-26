import streamlit as st
import pickle
import torch
from simplernn import SimpleRNNModel  # import your model class
from preprocess import preprocess_text  # your preprocessing function

# Load the trained model
@st.cache_resource
def load_model():
    model = SimpleRNNModel()
    model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    model.eval()
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
        processed_input = preprocess_text(user_input)
        # Convert to tensor
        input_tensor = torch.tensor([processed_input], dtype=torch.long)
        
        # Model prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        st.success(f"Predicted Class: {predicted_class}")
        st.info(f"Confidence: {confidence*100:.2f}%")
