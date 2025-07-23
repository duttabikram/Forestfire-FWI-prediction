import streamlit as st
import numpy as np
import pickle

# Load the model and scaler
ridge_model = pickle.load(open("models/ridge.pkl", "rb"))
standard_scaler = pickle.load(open("models/scaler.pkl", "rb"))

st.title("🔥 Forest Fire Risk Prediction")
st.markdown("Enter the environmental conditions below to predict fire risk.")

# Input fields
Temperature = st.number_input("Temperature (°C)", value=25.0)
RH = st.number_input("Relative Humidity (%)", value=40.0)
Ws = st.number_input("Wind Speed (km/h)", value=10.0)
Rain = st.number_input("Rainfall (mm)", value=0.0)
FFMC = st.number_input("FFMC Index", value=85.0)
DMC = st.number_input("DMC Index", value=50.0)
ISI = st.number_input("ISI Index", value=10.0)
Classes = st.number_input("Classes (0 or 1)", min_value=0, max_value=1)
Region = st.number_input("Region (0 to 7)", min_value=0, max_value=7)

# Collect the inputs into a feature array
input_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
scaled_data = standard_scaler.transform(input_data)

# Predict on button click
if st.button("Predict Fire Risk"):
    prediction = ridge_model.predict(scaled_data)
    st.success(f"Predicted Risk Score: {prediction[0]:.2f}")
