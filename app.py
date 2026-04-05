import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Life Expectancy Predictor", page_icon="🌍", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Nunito', sans-serif;
        background-color: #fdf6f0;
        color: #4a4a4a;
    }

    .stApp {
        background-color: #fdf6f0;
    }

    h1 {
        color: #7b5ea7 !important;
        font-weight: 700 !important;
        text-align: center;
        letter-spacing: 1px;
    }

    p, .stMarkdown p {
        color: #7a7a8c;
        text-align: center;
    }

    div[data-testid="stNumberInput"] label {
        color: #6b6b8a !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    div[data-testid="stNumberInput"] input {
        background-color: #ffffff !important;
        color: #4a4a4a !important;
        border: 1.5px solid #d5c8f0 !important;
        border-radius: 10px !important;
    }

    div[data-testid="stNumberInput"] input:focus {
        border-color: #a89fd8 !important;
        box-shadow: 0 0 0 3px #d5c8f044 !important;
    }

    .stButton > button {
        background-color: #a89fd8 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        font-family: 'Nunito', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 0.65rem 2.5rem !important;
        transition: all 0.2s ease;
        display: block;
        margin: 0 auto;
    }

    .stButton > button:hover {
        background-color: #9081cc !important;
        box-shadow: 0 4px 15px #a89fd855 !important;
        transform: translateY(-1px);
    }

    .stSuccess {
        background-color: #e8f5e9 !important;
        border: 1.5px solid #a5d6a7 !important;
        border-radius: 12px !important;
    }

    .stSuccess p {
        color: #388e3c !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        text-align: center;
    }

    hr {
        border-color: #e8dff5;
    }

    div[data-testid="column"] {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 1.2rem;
        border: 1.5px solid #ede8f8;
        box-shadow: 0 2px 10px #c5b8e822;
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
