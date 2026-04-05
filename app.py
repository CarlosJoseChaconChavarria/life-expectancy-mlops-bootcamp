import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Life Expectancy Predictor", page_icon="🌍", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #f5f5f5;
        color: #1a1a1a;
    }

    .stApp {
        background-color: #f5f5f5;
    }

    h1 {
        color: #1a1a1a !important;
        font-weight: 700 !important;
        text-align: center;
        letter-spacing: 1px;
    }

    p, .stMarkdown p {
        color: #555555;
        text-align: center;
    }

    div[data-testid="stNumberInput"] label {
        color: #333333 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    div[data-testid="stNumberInput"] input {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        border: 1.5px solid #cccccc !important;
        border-radius: 8px !important;
    }

    div[data-testid="stNumberInput"] input:focus {
        border-color: #888888 !important;
        box-shadow: 0 0 0 3px #88888822 !important;
    }

    .stButton > button {
        background-color: #333333 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.65rem 2.5rem !important;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background-color: #111111 !important;
        box-shadow: 0 4px 12px #00000033 !important;
    }

    .stSuccess {
        background-color: #eeeeee !important;
        border: 1.5px solid #aaaaaa !important;
        border-radius: 8px !important;
    }

    .stSuccess p {
        color: #1a1a1a !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        text-align: center;
    }

    hr {
        border-color: #dddddd;
    }
</style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load('life_expectancy_model.pkl')
scaler = joblib.load('scaler.pkl')

# Header
st.title("🌍 Life Expectancy Predictor")
st.markdown("Enter the details below to predict the life expectancy of a country.")
st.markdown("---")

# Two-column input layout
col1, col2 = st.columns(2, gap="medium")

with col1:
    mortality = st.number_input("Adult Mortality (per 1000)", min_value=0, max_value=1000, value=200)
    alcohol = st.number_input("Alcohol Consumption (liters per capita)", min_value=0.0, max_value=20.0, value=5.0)

with col2:
    gdp = st.number_input("GDP of the country", min_value=1, max_value=150000, value=5000)
    schooling = st.number_input("Average Years of Schooling", min_value=0.0, max_value=20.0, value=10.0)

st.markdown("---")

# Centered predict button
col_left, col_center, col_right = st.columns([1, 2, 1])
with col_center:
    predict = st.button("Predict Life Expectancy")

if predict:
    gdp_log = np.log1p(gdp)
    inputs = pd.DataFrame([[mortality, alcohol, gdp_log, schooling]],
                          columns=['Adult Mortality', 'Alcohol', 'GDP_transformed', 'Schooling'])
    inputs_scaled = scaler.transform(inputs)
    prediction = model.predict(inputs_scaled)
    st.success(f"The predicted Life Expectancy is: {prediction[0]:.2f} years")
