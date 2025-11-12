import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download, HfApi
import joblib
import os

# Initialize HF API with token from environment
api = HfApi(token=os.getenv("HF_TOKEN"))

# Download and load the model from Hugging Face
model_path = hf_hub_download(
    repo_id="adi333/engine-failure-prediction",
    repo_type="model"
    filename="best_engine_failure_prediction_model_v1.joblib"
)
model = joblib.load(model_path)

# Streamlit UI
st.title("Engine Failure Prediction")
st.write("""
This application predicts whether the automobile's engine condition is good or it needs maintenance. 
Please enter the engine sensor readings  below to get a prediction.
""")

# --- User inputs ---
engineRPM = st.number_input("Engine rpm", value=30)
lubOilPressure = st.number_input("Lub oil pressure", value=30 )
fuelPressure = st.number_input("Fuel pressure", value=30)
coolantPressure = st.number_input("Coolant pressure",value=30)
lubOilTemp = st.number_input("lub oil temp(in °C)",value=30)
coolantTemp = st.number_input("Coolant temp (in °C)",value=30)

# --- Assemble input ---
input_data = pd.DataFrame([{
    'Engine rpm' : engineRPM,
    'Lub oil pressure': lubOilPressure,
    'Fuel pressure': fuelPressure,
    'Coolant pressure': coolantPressure,
    'Lub oil temp': lubOilTemp,
    'Coolant temp': coolantTemp
}])

# --- Prediction ---
if st.button("Predict Engine Condition"):
    prediction = model.predict(input_data)[0]
    result = "Engine condition is good" if prediction==0 else "Engine condition is faulty"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
