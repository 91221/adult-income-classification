import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================
# Load Saved Files
# ==============================

model = joblib.load("model/xgboost.pkl")   # Best model
scaler = joblib.load("model/scaler.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")

st.set_page_config(page_title="Adult Income Predictor", layout="centered")

st.title("💼 Adult Income Prediction")
st.markdown("Predict whether income exceeds $50K/year")

# ==============================
# User Inputs
# ==============================

age = st.slider("Age", 18, 90, 30)
education_num = st.slider("Education Number", 1, 16, 10)
hours_per_week = st.slider("Hours per week", 1, 100, 40)
capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
capital_loss = st.number_input("Capital Loss", 0, 10000, 0)

# Categorical Dropdowns (decoded properly)

def selectbox_from_encoder(column):
    options = label_encoders[column].classes_
    return st.selectbox(column.replace("-", " ").title(), options)

workclass = selectbox_from_encoder("workclass")
marital_status = selectbox_from_encoder("marital-status")
occupation = selectbox_from_encoder("occupation")
relationship = selectbox_from_encoder("relationship")
race = selectbox_from_encoder("race")
sex = selectbox_from_encoder("sex")
native_country = selectbox_from_encoder("native-country")

# ==============================
# Prediction
# ==============================

if st.button("Predict Income"):

    # Encode categorical inputs
    workclass = label_encoders["workclass"].transform([workclass])[0]
    marital_status = label_encoders["marital-status"].transform([marital_status])[0]
    occupation = label_encoders["occupation"].transform([occupation])[0]
    relationship = label_encoders["relationship"].transform([relationship])[0]
    race = label_encoders["race"].transform([race])[0]
    sex = label_encoders["sex"].transform([sex])[0]
    native_country = label_encoders["native-country"].transform([native_country])[0]

    input_data = np.array([[age, workclass, 0, 0, education_num,
                            marital_status, occupation, relationship,
                            race, sex, capital_gain, capital_loss,
                            hours_per_week, native_country]])

    # Scale numeric features
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.success(f"Prediction: Income >50K (Confidence: {probability:.2f})")
    else:
        st.warning(f"Prediction: Income <=50K (Confidence: {1-probability:.2f})")

