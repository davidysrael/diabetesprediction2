import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import joblib
import math
import matplotlib.pyplot as plt
import base64

# Load model & scaler
model = joblib.load("rf_diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
  page_title="BloodBeaconPH",
  layout="centered",
  initial_sidebar_state="collapsed",
)

# Background image loader
def load_bg(path):
  with open(path, "rb") as f:
    return base64.b64encode(f.read()).decode()

BG_PATH = r"main/Welcome to BloodBeaconPH.png"
bg_base64 = load_bg(BG_PATH)

# Inject background and remove stepper controls
st.markdown(
  f"""
  <style>
    .stApp {{
      background: url("data:image/png;base64,{bg_base64}");
      background-size: cover;
      background-position: center;
      background-repeat: no-repeat;
      background-attachment: fixed;
    }}
    .block-container {{
      background-color: rgba(0,0,0,0.12);
      border-radius: 2rem;
      padding: 2rem;
      backdrop-filter: blur(6px);
    }}

    /* REMOVE numeric steppers completely at input level */
    input[type="number"] {{
      appearance: textfield !important;
      pointer-events: auto;
    }}
  </style>
  """,
  unsafe_allow_html=True,
)

# Risk heuristic function
def risk_likelihood(a, g, h, b):
  score = 0
  score += 1.5 if (a > 45) else 0
  score += 2 if (a > 60) else 0
  score += 2.5 if (g > 140) else 0
  score += 3 if (g > 200) else 0
  score += 2 if (h > 5.7) else 0
  score += 3 if (h > 6.5) else 0
  score += 1.5 if (b > 27) else 0
  score += 2.5 if (b > 30) else 0
  return min(score / 12, 1.0)

# Sidebar physician profile
with st.sidebar:
  st.subheader("👨‍⚕️ Physician Console")
  st.write("Dr. Gary Glucose A.I")
  st.caption("Endocrinologist • PH Biomarker Specialist")
  st.write("""
  Specialized in glucose pattern recognition and chronic illness risk assessment.  
  Passionate about diagnostics and preventive healthcare.
  """)

# Header
st.title("🩸 BloodBeaconPH")
st.write("Hi, Dr. Gary Glucose at your service. I am a Machine Learning powered diabetes risk scanner configured for PH clinical trends.")

# PH glossary
with st.expander("🧾 PH Medical Glossary"):
  st.write("""
  HbA1c — measures average blood sugar in the last 2–3 months.  
  Glucose mg/dL — current blood sugar concentration.  
  BMI — body mass index based on height and weight.  
  Hypertension — high blood pressure, a diabetes risk factor.
  """)

# BMI Calculator
st.subheader("📏 BMI Calculator")

if ("bmi_calc_value" not in st.session_state):
  st.session_state.bmi_calc_value = None

weight = st.number_input("Weight (kg)", min_value=(1.0), max_value=(300.0), value=(70.0), format=("%.2f"), key=("bmi_w"))
height = st.number_input("Height (cm)", min_value=(30.0), max_value=(250.0), value=(170.0), format=("%.2f"), key=("bmi_h"))

if (st.button("Compute BMI", key=("btn_bmi"))):
  bmi_temp = weight / ((height / 100) ** 2)
  st.session_state.bmi_calc_value = round(bmi_temp, 2)
  st.metric("✅ Calculated BMI", f"{bmi_temp:.2f}")

bmi = st.session_state.bmi_calc_value

# Patient Inputs
st.subheader("🧬 Patient Inputs")

gender = st.selectbox("Gender", ["Male","Female"], key=("gender_select_main"))
age = st.number_input("Age (years)", min_value=(10.0), max_value=(80.0), value=(30.0), format=("%.2f"), key=("age_input_main"))
hypertension = st.selectbox("Hypertension [0=none, 1=yes]", [0, 1], key=("input_htn_main"))
heart_disease = st.selectbox("Heart Disease [0=none, 1=yes]", [0, 1], key=("input_hd_main"))
hba1c = st.number_input("HbA1c (%)", min_value=(4.0), max_value=(9.0), value=(5.5), format=("%.2f"), key=("input_hba1c_main"))
glucose = st.number_input("Blood Glucose (mg/dL)", min_value=(70.0), max_value=(300.0), value=(100.0), format=("%.2f"), key=("input_glucose_main"))

# Round all numeric inputs to 2 decimals if typed beyond limit
age = round(age, 2)
hba1c = round(hba1c, 2)
glucose = round(glucose, 2)

# Metrics Panel
c1, c2, c3, c4 = st.columns(4)
c1.metric("Age", age)
c2.metric("BMI", ("--" if (bmi is None) else bmi))
c3.metric("Glucose", glucose)
c4.metric("HbA1c", hba1c)

# Validation interlock
scan_ready = False
if (bmi is None):
  st.warning("🔐 Scan lock: compute BMI first to proceed.")
else:
  scan_ready = True

# Build model input matrix
gender_encoded = 1 if (gender == "Male") else 0
X = np.array([[gender_encoded, age, hypertension, heart_disease, bmi, hba1c, glucose]])
console = st.empty()

# Beacon Scan Prediction
if (st.button("🔍 Initiate Beacon Scan", key=("btn_predict"), disabled=(not scan_ready))):
  st.subheader("📊 Biomarker Breakdown")

  values = [age/80 * 100, bmi/40 * 100, glucose/300 * 100, hba1c/9 * 100]
  labels = ["Age","BMI","Glucose","HbA1c"]

  fig, ax = plt.subplots()
  ax.bar(labels, values)

  ax.set_title("PH Clinical Biomarker Levels", fontsize=(14))
  ax.set_ylabel("Risk Contribution (%)", fontsize=(12))
  ax.set_ylim(0, 110)
  ax.grid(axis=("y"), alpha=(0.2))

  for i, v in enumerate(values):
    ax.text(i, v + 2, f"{v:.1f}%", ha=("center"), fontsize=(12), weight=("bold"))

  st.pyplot(fig)

  r_live = risk_likelihood(age, glucose, hba1c, bmi)
  st.subheader("📡 Live Risk Radar")
  st.progress(r_live)
  st.caption(f"Current system threat index: {r_live * 100:.1f}%")

  console.write("Calibrating hematology sensors...")
  console.write("Reading glucose and HbA1c matrix...")
  console.write("Firing predictive core...")

  X_scaled = scaler.transform(X)
  result = model.predict(X_scaled)[0]

  if (result == 1):
    st.error("🚨 High risk detected.")
    console.write("A probability of insulin resistance alert.")
  else:
    st.success("✅ No high risk detected.")
    st.balloons()
    console.write("All vitals optimal, sir.")

# Footer
st.write("---")
st.caption("Diagnostics completed by Dr. Gary Glucose from BloodBeaconPH system core.")
