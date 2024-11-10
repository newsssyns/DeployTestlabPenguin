
import streamlit as st
import pickle
import pandas as pd

# Load the saved model and encoders
with open('model_penguin_66130701713.pkl', 'rb') as file:
    model, species_encoder, island_encoder, sex_encoder = pickle.load(file)

# Define the prediction function
def predict_penguin_species(island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex):
    # Create a DataFrame for the input features
    input_data = pd.DataFrame({
        'island': [island],
        'bill_length_mm': [bill_length_mm],
        'bill_depth_mm': [bill_depth_mm],
        'flipper_length_mm': [flipper_length_mm],
        'body_mass_g': [body_mass_g],
        'sex': [sex]
    })

    # Encode categorical features
    input_data['island'] = island_encoder.transform(input_data['island'])
    input_data['sex'] = sex_encoder.transform(input_data['sex'])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Decode prediction to species name
    predicted_species = species_encoder.inverse_transform([prediction])[0]

    return predicted_species

# Streamlit app
st.title("Penguin Species Prediction")

# Input fields
island = st.selectbox("Island", island_encoder.classes_)
bill_length_mm = st.number_input("Bill Length (mm)")
bill_depth_mm = st.number_input("Bill Depth (mm)")
flipper_length_mm = st.number_input("Flipper Length (mm)")
body_mass_g = st.number_input("Body Mass (g)")
sex = st.selectbox("Sex", sex_encoder.classes_)

# Prediction button
if st.button("Predict"):
    predicted_species = predict_penguin_species(island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex)
    st.success(f"The predicted species is: **{predicted_species}**")

