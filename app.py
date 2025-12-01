import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import base64
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.subplots as sp
import json

# Main goal: load ML assets + dataset, fix repo paths, embed analytics HTML

# --- PATH FIX (repo-relative) ---
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "diabetes_prediction_dataset.csv")

# Load dataset
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# ML assets
model = joblib.load(os.path.join(BASE_DIR, "rf_diabetes_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

# Background loader
def load_bg(path):
  with open(path, "rb") as f:
    return base64.b64encode(f.read()).decode()

BG_PATH = os.path.join(BASE_DIR, "main", "Welcome to BloodBeaconPH.png")
bg_base64 = load_bg(BG_PATH)

# Page config
st.set_page_config(
  page_title="BloodBeaconPH",
  layout="centered",
  initial_sidebar_state="collapsed",
)

# Apply background
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
  </style>
  """,
  unsafe_allow_html=True,
)

# Risk scoring
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

# Sidebar console
with st.sidebar:
  st.subheader("🧬 System Physician Console")
  st.write("**AI Core Physician — BloodBeacon**")

# Header
st.title("🩸 BloodBeaconPH Analytics")
st.caption("Feature intelligence refreshed on every rerun.")

# Glossary with Heart Disease added
with st.expander("🧾 PH Medical Glossary"):
  st.write("""
**HbA1c** — measures average blood sugar in the last 2–3 months.  
**Glucose mg/dL** — current blood sugar concentration.  
**BMI** — body mass index based on height and weight.  
**Hypertension** — high blood pressure, a diabetes risk factor.  
**Heart Disease** — cardiovascular condition that increases diabetes risk.  
  """)

# --- DASHBOARD BUILD (Plotly HTML export) ---
RANGES = {
  "age": (0.08, 80.0),
  "bmi": (10.01, 95.69),
  "HbA1c_level": (3.5, 9.0),
  "blood_glucose_level": (80, 300),
  "hypertension": (0, 1),
  "heart_disease": (0, 1),
  "diabetes": (0, 1),
}

df_stats = df.copy()

dashboard = sp.make_subplots(
  rows=4, cols=2,
  specs=[
    [{"type":"bar"}, {"type":"bar"}],
    [{"type":"bar"}, {"type":"bar"}],
    [{"type":"bar"}, {"type":"bar"}],
    [{"type":"pie"}, None]
  ],
  subplot_titles=[
    "Age","BMI",
    "Current Blood Glucose","HbA1c Level",
    "Hypertension","Heart Disease",
    "Diabetes","Gender"
  ]
)

def build_hist(feature, bins):
  mn, mx = RANGES[feature]
  hist, edges = np.histogram(df_stats[feature], bins=bins, range=(mn, mx))
  labels = [f"{edges[i]:.2f} – {edges[i+1]:.2f}" for i in range(len(hist))]
  return labels, hist.tolist()

age_labels, age_vals = build_hist("age", 20)
bmi_labels, bmi_vals = build_hist("bmi", 18)
glucose_labels, glucose_vals = build_hist("blood_glucose_level", 15)
hba1c_labels, hba1c_vals = build_hist("HbA1c_level", 15)

dashboard.add_trace(go.Bar(x=age_labels, y=age_vals, hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>"), row=1,col=1)
dashboard.add_trace(go.Bar(x=bmi_labels, y=bmi_vals, hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>"), row=1,col=2)
dashboard.add_trace(go.Bar(x=glucose_labels, y=glucose_vals, hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>"), row=2,col=1)
dashboard.add_trace(go.Bar(x=hba1c_labels, y=hba1c_vals, hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>"), row=2,col=2)

def build_flag(feature):
  counts = df_stats[feature].value_counts().sort_index()
  return counts.index.astype(str).tolist(), counts.values.tolist()

htn_l, htn_v = build_flag("hypertension")
hd_l, hd_v = build_flag("heart_disease")
dm_l, dm_v = build_flag("diabetes")

dashboard.add_trace(go.Bar(x=htn_l, y=htn_v, hovertemplate="Value: %{x}<br>Count: %{y}<extra></extra>"), row=3,col=1)
dashboard.add_trace(go.Bar(x=hd_l, y=hd_v, hovertemplate="Value: %{x}<br>Count: %{y}<extra></extra>"), row=3,col=2)
dashboard.add_trace(go.Bar(x=dm_l, y=dm_v, hovertemplate="Value: %{x}<br>Count: %{y}<extra></extra>"), row=4,col=1)

gender_counts = df_stats["gender"].value_counts()
dashboard.add_trace(go.Pie(
  labels=gender_counts.index.tolist(),
  values=gender_counts.values.tolist(),
  hole=0.35,
  hovertemplate="Category: %{label}<br>Count: %{value}<br>Percent: %{percent}<extra></extra>"
), row=4,col=1)

dashboard.update_layout(height=900, showlegend=False, margin=dict(l=20,r=20,t=40,b=20))

# Write analytics HTML and JSON
dashboard.write_html(os.path.join(BASE_DIR, "dashboard.html"))

bundle = {
  f: {"min":v[0], "max":v[1]} for f,v in RANGES.items()
}
bundle["charts"] = {
  "age": age_vals,
  "bmi": bmi_vals,
  "blood_glucose": glucose_vals,
  "HbA1c": hba1c_vals,
  "hypertension": htn_v,
  "heart_disease": hd_v,
  "diabetes": dm_v,
  "gender": {"labels": gender_counts.index.tolist(), "values": gender_counts.values.tolist()}
}

with open(os.path.join(BASE_DIR, "charts.json"), "w") as jf:
  json.dump(bundle, jf, indent=2)

# --- EMBED ANALYTICS PANEL UNDER APP ---
with st.expander("📊 Live Analytics Panel", expanded=True):
  html = open(os.path.join(BASE_DIR, "dashboard.html"), "r").read()
  st.components.v1.html(html, height=920)

# Footer
st.write("---")
st.caption("Dashboard rendered by BloodBeaconPH AI Core.")
