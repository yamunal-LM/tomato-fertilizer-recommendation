import streamlit as st
import pandas as pd
import joblib

# Load saved files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("🍅 Tomato Fertilizer Recommendation System")

soil_ph = st.number_input("Soil pH", value=6.5)
nitrogen = st.number_input("Nitrogen (mg/kg)", value=45)
phosphorus = st.number_input("Phosphorus (mg/kg)", value=30)
potassium = st.number_input("Potassium (mg/kg)", value=180)
moisture = st.number_input("Moisture (%)", value=65)
application_rate = st.number_input("Application Rate (kg/acre)", value=50)

growth_stage = st.selectbox(
    "Growth Stage",
    ["Seedling", "Vegetative", "Flowering", "Fruiting"]
)

if st.button("Predict Fertilizer"):

    data = pd.DataFrame({
        "Sample_ID": [1],
        "Soil_pH": [soil_ph],
        "Nitrogen_mgkg": [nitrogen],
        "Phosphorus_mgkg": [phosphorus],
        "Potassium_mgkg": [potassium],
        "Moisture_%": [moisture],
        "Growth_Stage": [growth_stage],
        "Application_Rate_kg_per_acre": [application_rate]
    })

    data = pd.get_dummies(data)
    data = data.reindex(columns=feature_columns, fill_value=0)
    data = scaler.transform(data)

    prediction = model.predict(data)

    st.success(f"Recommended Fertilizer: {prediction[0]}")
