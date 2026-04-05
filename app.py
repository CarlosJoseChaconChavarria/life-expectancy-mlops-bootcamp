import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Life Expectancy Predictor", page_icon="🌍", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Share+Tech+Mono&display=swap');

    html, body, [class*="css"] {
        font-family: 'Share Tech Mono', monospace;
        background-color: #0a0e1a;
        color: #00f5ff;
    }

    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a0e1a 100%);
    }

    h1 {
        font-family: 'Orbitron', monospace !important;
        color: #00f5ff !important;
        text-shadow: 0 0 20px #00f5ff, 0 0 40px #00f5ff88;
        letter-spacing: 3px;
        text-align: center;
    }

    p, .stMarkdown {
        color: #7fdbff;
    }

    div[data-testid="stNumberInput"] label {
        color: #00f5ff !important;
        font-family: 'Share Tech Mono', monospace !important;
        letter-spacing: 1px;
        text-transform: uppercase;
        font-size: 0.85rem;
    }

    div[data-testid="stNumberInput"] input {
        background-color: #0d1b2a !important;
        color: #00f5ff !important;
        border: 1px solid #00f5ff55 !important;
        border-radius: 4px !important;
        font-family: 'Share Tech Mono', monospace !important;
    }

    div[data-testid="stNumberInput"] input:focus {
        border: 1px solid #00f5ff !important;
        box-shadow: 0 0 10px #00f5ff55 !important;
    }

    .stButton > button {
        background: transparent !important;
        color: #00f5ff !important;
        border: 1px solid #00f5ff !important;
        border-radius: 4px !important;
        font-family: 'Orbitron', monospace !important;
        letter-spacing: 2px;
        text-transform: uppercase;
        width: 100%;
        padding: 0.75rem;
        transition: all 0.3s ease;
        box-shadow: 0 0 10px #00f5ff33;
    }

    .stButton > button:hover {
        background: #00f5ff22 !important;
        box-shadow: 0 0 20px #00f5ff88, inset 0 0 20px #00f5ff11 !important;
    }

    .stSuccess {
        background-color: #001a1a !important;
        border: 1px solid #00f5ff !important;
        border-radius: 4px !important;
        color: #00f5ff !important;
        box-shadow: 0 0 20px #00f5ff44;
    }

    .stSuccess p {
        color: #00f5ff !important;
        font-family: 'Orbitron', monospace !important;
        font-size: 1.1rem;
        text-shadow: 0 0 10px #00f5ff;
    }

    hr {
        border-color: #00f5ff33;
    }
</style>
""", unsafe_allow_html=True)

# 1. LOAD THE SAVED BRAIN
model = joblib.load('life_expectancy_model.pkl')
scaler = joblib.load('scaler.pkl')

# 2. DESIGN THE WEBSITE UI
st.title("LIFE EXPECTANCY PREDICTOR")
st.markdown("<p style='text-align:center; letter-spacing:2px;'>Enter parameters to compute life expectancy forecast</p>", unsafe_allow_html=True)
st.markdown("---")

# 3. CREATE INPUT BOXES FOR THE USER
col1, col2 = st.columns(2)

with col1:
    mortality = st.number_input("Adult Mortality (per 1000)", min_value=0, max_value=1000, value=200)
    gdp = st.number_input("GDP of the Country", min_value=1, max_value=150000, value=5000)

with col2:
    alcohol = st.number_input("Alcohol Consumption (liters/capita)", min_value=0.0, max_value=20.0, value=5.0)
    schooling = st.number_input("Average Years of Schooling", min_value=0.0, max_value=20.0, value=10.0)

st.markdown("---")

# 4. THE "PREDICT" BUTTON
if st.button("[ COMPUTE FORECAST ]"):
    gdp_log = np.log1p(gdp)
    inputs = pd.DataFrame([[mortality, alcohol, gdp_log, schooling]],
                          columns=['Adult Mortality', 'Alcohol', 'GDP_transformed', 'Schooling'])
    inputs_scaled = scaler.transform(inputs)
    prediction = model.predict(inputs_scaled)
    st.success(f"PREDICTED LIFE EXPECTANCY: {prediction[0]:.2f} YEARS")
