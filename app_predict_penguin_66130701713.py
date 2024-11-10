import streamlit as st
import pickle
import pandas as pd

# Load the saved model and encoders
@st.cache_resource
def load_model_and_encoders():
    try:
        with open('model_penguin_66130701713.pkl', 'rb') as file:
            loaded_data = pickle.load(file)
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
    
    # User input fields
    island = st.selectbox("Island", island_encoder.classes_)
    culmen_length_mm = st.number_input("Culmen Length (mm)", min_value=0.0)
    culmen_depth_mm = st.number_input("Culmen Depth (mm)", min_value=0.0)
    flipper_length_mm = st.number_input("Flipper Length (mm)", min_value=0)
    body_mass_g = st.number_input("Body Mass (g)", min_value=0)
    sex = st.selectbox("Sex", sex_encoder.classes_)

    # Prediction function
    def predict_penguin_species(island, culmen_length_mm, culmen_depth_mm, flipper_length_mm, body_mass_g, sex):
        # Create a DataFrame with the correct feature names as in training
        input_data = pd.DataFrame({
            'island': [island],
            'culmen_length_mm': [culmen_length_mm],
            'culmen_depth_mm': [culmen_depth_mm],
            'flipper_length_mm': [flipper_length_mm],
            'body_mass_g': [body_mass_g],
            'sex': [sex]
        })
        
        # Encode categorical features
        input_data['island'] = island_encoder.transform(input_data['island'])
        input_data['sex'] = sex_encoder.transform(input_data['sex'])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        predicted_species = species_encoder.inverse_transform([prediction])[0]
        return predicted_species

    # Add a prediction button
    if st.button("Predict Species"):
        result = predict_penguin_species(island, culmen_length_mm, culmen_depth_mm, flipper_length_mm, body_mass_g, sex)
        st.write(f"Predicted Penguin Species: {result}")

else:
    st.warning("Could not load the model and encoders. Please fix the issue to proceed.")
