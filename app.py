import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import joblib
import math
import matplotlib.pyplot as plt
import base64

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

# Inject background + FORCE REMOVE numeric input type so browser NEVER spawns spinner
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

    /* Completely kill number input spinner UI */
    input[type="number"] {{
      all: unset !important;
      display: block;
      width: 100%;
      background-color: white;
      color: black;
      padding: 0.4rem;
      border-radius: 0.6rem;
      border: 1px solid #ccc;
      font-size: 1rem;
    }}
  </style>
  """,
  unsafe_allow_html=True,
)

# Load model + scaler
model = joblib.load("rf_diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Risk heuristic
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

# Sidebar profile
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
st.write("Dr. Gary Glucose online. Hematology sensors primed and awaiting input.")

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

# Replacing number_input button UI by taking clean decimals manually
weight = st.text_input("Weight (kg)", value=("70.00"))
height = st.text_input("Height (cm)", value=("170.00"))

# Convert to float safely, enforce 2 decimals
try:
  weight = round(float(weight), 2)
except:
  weight = 70.00

try:
  height = round(float(height), 2)
except:
  height = 170.00

if (st.button("Compute BMI", key=("btn_bmi"))):
  bmi_temp = weight / ((height / 100) ** 2)
  st.session_state.bmi_calc_value = round(bmi_temp, 2)
  st.metric("✅ Calculated BMI", f"{bmi_temp:.2f}")

bmi = st.session_state.bmi_calc_value

# Inputs
st.subheader("🧬 Patient Inputs")

gender = st.selectbox("Gender", ["Male","Female"], key=("gender_select_main"))
age = st.text_input("Age (years)", value=("30.00"))
hba1c = st.text_input("HbA1c (%)", value=("5.50"))
glucose = st.text_input("Blood Glucose (mg/dL)", value=("100.00"))

hypertension = st.selectbox("Hypertension [0=none, 1=yes]", [0, 1], key=("input_htn_main"))
heart_disease = st.selectbox("Heart Disease [0=none, 1=yes]", [0, 1], key=("input_hd_main"))

# Convert text decimals to float and enforce 2 decimals
try:
  age = round(float(age), 2)
except:
  age = 30.00

try:
  hba1c = round(float(hba1c), 2)
except:
  hba1c = 5.50

try:
  glucose = round(float(glucose), 2)
except:
  glucose = 100.00

# Metrics panel
c1, c2, c3, c4 = st.columns(4)
c1.metric("Age", f"{age:.2f}")
c2.metric("BMI", ("--" if (bmi is None) else f"{bmi:.2f}"))
c3.metric("Glucose", f"{glucose:.2f}")
c4.metric("HbA1c", f"{hba1c:.2f}")

# Validation interlock
scan_ready = False
if (bmi is None):
  st.warning("🔐 Scan lock: BMI must be computed first.")
else:
  scan_ready = True

# Build ML input matrix
gender_encoded = 1 if (gender == "Male") else 0
X = np.array([[gender_encoded, age, hypertension, heart_disease, bmi, hba1c, glucose]])
console = st.empty()

# Prediction
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
    ax.text(i, v + 2, f"{round(v,2):.2f}%", ha=("center"), fontsize=(12), weight=("bold"))

  st.pyplot(fig)

  r_live = risk_likelihood(age, glucose, hba1c, bmi)
  st.subheader("📡 Live Risk Radar")
  st.progress(r_live)
  st.caption(f"Current system threat index: {r_live * 100:.2f}%")

  console.write("Calibrating sensors...")
  console.write("Reading biomarker matrix...")
  console.write("Running predictive core...")

  X_scaled = scaler.transform(X)
  result = model.predict(X_scaled)[0]

  if (result == 1):
    st.error("🚨 High diabetes risk detected.")
    console.write("Insulin resistance warning likely.")
  else:
    st.success("✅ No high diabetes risk detected.")
    st.balloons()
    console.write("Vitals are within optimal range, sir.")

# Footer
st.write("---")
st.caption("Diagnostics completed by Dr. Gary Glucose from BloodBeaconPH system core.")
