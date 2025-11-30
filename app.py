import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import math

# Load model & scaler
model = joblib.load("rf_diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Diabetes Prediction",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("🩺 Diabetes Prediction System")
st.write("Enter your details, and I'll analyze your risk index.")

# Initialize BMI calculator storage if not present
if "bmi_calc_value" not in st.session_state:
    st.session_state.bmi_calc_value = 25.0

# ----- Inputs (except BMI slider removed) -----
gender = st.selectbox("Gender", ["Male", "Female"], key="gender_select_main")
age = st.slider("Age (years)", 10, 80, 30, key="age_slider_main")
hypertension = st.selectbox("Hypertension (0=none, 1=yes)", [0, 1], key="input_htn_main")
heart_disease = st.selectbox("Heart Disease (0=none, 1=yes)", [0, 1], key="input_hd_main")
hba1c = st.slider("HbA1c (%)", 4.0, 9.0, 5.5, step=0.1, key="input_hba1c_main")
glucose = st.slider("Blood Glucose (mg/dL)", 70, 300, 100, key="input_glucose_main")

# ----- BMI Calculator (always used as input now) -----
with st.expander("📏 Compute BMI Here"):
    weight = st.number_input("Weight (kg)", 1.0, 300.0, 70.0, key="bmi_calc_w_main")
    height = st.number_input("Height (cm)", 30.0, 250.0, 170.0, key="bmi_calc_h_main")
    if st.button("Compute BMI", key="btn_bmi_calc_main"):
        bmi_val = weight / ((height / 100) ** 2)
        st.metric("Calculated BMI", f"{bmi_val:.2f}")
        st.session_state.bmi_calc_value = round(bmi_val, 2)

# Assign BMI input from calculator storage (no slider fallback anymore)
bmi = st.session_state.bmi_calc_value

# ----- Risk Likelihood Scoring -----
def risk_likelihood(a, g, h, b):
    s = 0
    s += 1.5 if (a > 45) else 0
    s += 2   if (a > 60) else 0
    s += 2.5 if (g > 140) else 0
    s += 3   if (g > 200) else 0
    s += 2   if (h > 5.7) else 0
    s += 3   if (h > 6.5) else 0
    s += 1.5 if (b > 27) else 0
    s += 2.5 if (b > 30) else 0
    return min(s / 12, 1.0)

# ----- Prepare model input -----
gender_encoded = 1 if (gender == "Male") else 0
X = np.array([[gender_encoded, age, hypertension, heart_disease, bmi, hba1c, glucose]])
X_scaled = scaler.transform(X)

# ----- Prediction Trigger (Next button press) -----
if st.button("Predict Diabetes", key="btn_predict_main"):
    prediction = model.predict(X_scaled)[0]

    if prediction == 1:
        st.error("⚠️ Prediction: Positive")
    else:
        st.success("✅ Prediction: Negative")
        st.balloons()

    # ----- Risk Insight Chart -----
    if prediction == 0:
        st.subheader("📊 Philippine Diabetes Risk Insight")

        r = risk_likelihood(age, glucose, hba1c, bmi)
        vals = [age/80*100, bmi/40*100, glucose/300*100, hba1c/9*100]
        labs = ["Age", "BMI", "Glucose", "HbA1c"]

        fig, ax = plt.subplots()
        ax.bar(labs, vals)
        ax.set_ylabel("Relative Scale (%)")
        ax.set_ylim(0, 105)

        for i, v in enumerate(vals):
            ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=11)

        st.pyplot(fig)

        st.write("---")
        st.metric("Diabetes Risk Score (PH Pattern)", f"{r*100:.1f}%")
        st.progress(r)

        if r < 0.2:
            st.write("🏅 Health Guardian Badge Unlocked")
        elif r < 0.5:
            st.write("🎖️ Risk Aware Badge Earned")
        else:
            st.write("🔥 Survivor Mode — medical guidance advised")

st.write("---")
