import streamlit as st
import pickle
import pandas as pd

# Load the saved model and encoders
@st.cache_resource  # Cache the model loading to avoid reloading on every interaction
def load_model_and_encoders():
    try:
        with open('model_penguin_66130701713.pkl', 'rb') as file:
            loaded_data = pickle.load(file)
            # Assuming loaded_data is a tuple/list containing model and encoders:
            model = loaded_data[0]
            species_encoder = loaded_data[1]
            island_encoder = loaded_data[2]
            sex_encoder = loaded_data[3]
        return model, species_encoder, island_encoder, sex_encoder
    except FileNotFoundError:
        st.error("Model file not found. Please check the file path.")
        return None, None, None, None

# Load model and encoders
model, species_encoder, island_encoder, sex_encoder = load_model_and_encoders()

# Check if loading was successful before proceeding
if model is not None:
    st.success("Model and encoders loaded successfully!")
    # Here you can add code to input user data and make predictions
else:
    st.warning("Could not load the model and encoders. Please fix the issue to proceed.")
