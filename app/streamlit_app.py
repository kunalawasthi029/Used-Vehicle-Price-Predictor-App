"""
Vehicle Price Prediction - Streamlit Web Application

An interactive app that lets users estimate used car prices based on
vehicle features using a trained XGBoost model. Dropdown options
cascade-filter based on manufacturer using curated real-world mappings.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")

# Add project root to path so src modules are importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import load_artifact, PLOTS_DIR
from src.data_preprocessing import CYLINDER_MAP, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from src.manufacturer_mapping import get_manufacturer_options


# -----------------------------------------------
# US State Abbreviation -> Full Name Mapping
# -----------------------------------------------
STATE_NAMES = {
    "ak": "Alaska", "al": "Alabama", "ar": "Arkansas", "az": "Arizona",
    "ca": "California", "co": "Colorado", "ct": "Connecticut", "dc": "Washington DC",
    "de": "Delaware", "fl": "Florida", "ga": "Georgia", "hi": "Hawaii",
    "ia": "Iowa", "id": "Idaho", "il": "Illinois", "in": "Indiana",
    "ks": "Kansas", "ky": "Kentucky", "la": "Louisiana", "ma": "Massachusetts",
    "md": "Maryland", "me": "Maine", "mi": "Michigan", "mn": "Minnesota",
    "mo": "Missouri", "ms": "Mississippi", "mt": "Montana", "nc": "North Carolina",
    "nd": "North Dakota", "ne": "Nebraska", "nh": "New Hampshire", "nj": "New Jersey",
    "nm": "New Mexico", "nv": "Nevada", "ny": "New York", "oh": "Ohio",
    "ok": "Oklahoma", "or": "Oregon", "pa": "Pennsylvania", "ri": "Rhode Island",
    "sc": "South Carolina", "sd": "South Dakota", "tn": "Tennessee", "tx": "Texas",
    "ut": "Utah", "va": "Virginia", "vt": "Vermont", "wa": "Washington",
    "wi": "Wisconsin", "wv": "West Virginia", "wy": "Wyoming",
}
STATE_ABBREV = {v: k for k, v in STATE_NAMES.items()}


# -----------------------------------------------
# Page Configuration
# -----------------------------------------------
st.set_page_config(
    page_title="Vehicle Price Predictor",
    page_icon="https://cdn-icons-png.flaticon.com/512/741/741407.png",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------
# Custom CSS
# -----------------------------------------------
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.3rem; }
    .main-header p  { font-size: 1.05rem; opacity: 0.9; }

    .prediction-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem; border-radius: 16px;
        text-align: center; color: white; margin: 1.5rem 0;
    }
    .prediction-card .price { font-size: 3rem; font-weight: 800; margin: 0.5rem 0; }
    .prediction-card .label { font-size: 1.1rem; opacity: 0.9; }

    .waiting-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 16px;
        text-align: center; color: white; margin: 1.5rem 0;
    }
    .waiting-card .message { font-size: 1.3rem; font-weight: 600; margin: 0.5rem 0; }
    .waiting-card .sub     { font-size: 1rem; opacity: 0.85; }

    .section-header {
        font-size: 1.3rem; font-weight: 600; color: #333;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem; margin: 1.5rem 0 1rem 0;
    }
    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white; font-size: 1.1rem; font-weight: 700;
        padding: 0.7rem 1.5rem; border: none; border-radius: 10px;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #0e8174 0%, #2ecc6e 100%);
        color: white;
    }
    #MainMenu {visibility: hidden;}
    footer    {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------
# Load Model & Artifacts
# -----------------------------------------------
@st.cache_resource
def load_model_artifacts():
    """Load the trained model, label encoders, feature list, and metrics."""
    model = load_artifact("xgboost_best_model.pkl")
    encoders = load_artifact("label_encoders.pkl")
    feature_cols = load_artifact("feature_columns.pkl")
    metrics = load_artifact("model_metrics.pkl")
    return model, encoders, feature_cols, metrics


try:
    model, encoders, feature_cols, metrics = load_model_artifacts()
except FileNotFoundError:
    st.error(
        "Model artifacts not found. "
        "Run the training pipeline first:  python -m src.model_training"
    )
    st.stop()


# -----------------------------------------------
# Helpers
# -----------------------------------------------
def safe_index(options, preferred):
    """Return index of preferred value in options, or 0 if not found."""
    return options.index(preferred) if preferred in options else 0


def display_name(value):
    """Capitalize a feature value for display (title case)."""
    if value in ("SUV", "4wd", "fwd", "rwd"):
        return value.upper()
    return value.replace("-", " ").title()


# -----------------------------------------------
# Header
# -----------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>Vehicle Price Predictor</h1>
    <p>Estimate used car prices using Machine Learning &mdash; powered by XGBoost</p>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------
# Sidebar - User Inputs with Curated Filters
# -----------------------------------------------
st.sidebar.markdown("## Vehicle Features")
st.sidebar.markdown(
    "Select vehicle details below. Options automatically adjust based on "
    "the manufacturer to show only realistic combinations."
)
st.sidebar.markdown("---")

# --- Manufacturer (primary selector) ---
all_manufacturers = sorted(encoders["manufacturer"].classes_)
# Remove 'Unknown' from the UI
mfr_display = [m for m in all_manufacturers if m != "Unknown"]
manufacturer = st.sidebar.selectbox(
    "Manufacturer",
    mfr_display,
    index=safe_index(mfr_display, "toyota"),
    format_func=display_name,
)

st.sidebar.markdown("---")

# --- Year ---
year = st.sidebar.slider("Year", min_value=2015, max_value=2022, value=2019, step=1)

# --- Vehicle Type (curated by manufacturer) ---
valid_types = get_manufacturer_options(manufacturer, "type")
vehicle_type = st.sidebar.selectbox(
    "Vehicle Type", valid_types,
    index=safe_index(valid_types, "sedan"),
    format_func=display_name,
)

# --- Fuel Type (curated by manufacturer) ---
valid_fuels = get_manufacturer_options(manufacturer, "fuel")
fuel = st.sidebar.selectbox(
    "Fuel Type", valid_fuels,
    index=safe_index(valid_fuels, "gas"),
    format_func=display_name,
)

# --- Cylinders (curated by manufacturer) ---
valid_cyls = get_manufacturer_options(manufacturer, "cylinders")
cylinders_str = st.sidebar.selectbox(
    "Cylinders", valid_cyls,
    index=safe_index(valid_cyls, "6 cylinders") if "6 cylinders" in valid_cyls
          else safe_index(valid_cyls, "4 cylinders"),
    format_func=display_name,
)
cylinders_num = CYLINDER_MAP.get(cylinders_str, 4)

# --- Transmission (curated by manufacturer) ---
valid_trans = get_manufacturer_options(manufacturer, "transmission")
transmission = st.sidebar.selectbox(
    "Transmission", valid_trans,
    index=safe_index(valid_trans, "automatic"),
    format_func=display_name,
)

# --- Drive (curated by manufacturer) ---
valid_drives = get_manufacturer_options(manufacturer, "drive")
drive = st.sidebar.selectbox(
    "Drive", valid_drives,
    index=safe_index(valid_drives, "fwd") if "fwd" in valid_drives
          else safe_index(valid_drives, "4wd"),
    format_func=display_name,
)

st.sidebar.markdown("---")

# --- Condition ---
condition_options = [c for c in encoders["condition"].classes_ if c != "Unknown"]
condition = st.sidebar.selectbox(
    "Condition", condition_options,
    index=safe_index(condition_options, "good"),
    format_func=display_name,
)

# --- Title Status ---
title_options = [t for t in encoders["title_status"].classes_ if t != "Unknown"]
title_status = st.sidebar.selectbox(
    "Title Status", title_options,
    index=safe_index(title_options, "clean"),
    format_func=display_name,
)

# --- Odometer ---
odometer = st.sidebar.slider(
    "Odometer (miles)", min_value=0, max_value=300000, value=40000, step=1000,
)

# --- Paint Color ---
color_options = [c for c in encoders["paint_color"].classes_ if c != "Unknown"]
paint_color = st.sidebar.selectbox(
    "Paint Color", color_options,
    index=safe_index(color_options, "white"),
    format_func=display_name,
)

# --- State (full names) ---
state_full_names = sorted(STATE_NAMES.values())
state_full = st.sidebar.selectbox(
    "State", state_full_names,
    index=safe_index(state_full_names, "California"),
)
state_abbrev = STATE_ABBREV.get(state_full, "ca")

st.sidebar.markdown("---")

# --- PREDICT BUTTON ---
predict_clicked = st.sidebar.button("Predict Price")


# -----------------------------------------------
# Build Input Feature Vector
# -----------------------------------------------
car_age = 2025 - year

input_data = {
    "year": year,
    "odometer": odometer,
    "car_age": car_age,
    "cylinders_num": cylinders_num,
    "manufacturer": encoders["manufacturer"].transform([manufacturer])[0],
    "condition": encoders["condition"].transform([condition])[0],
    "fuel": encoders["fuel"].transform([fuel])[0],
    "title_status": encoders["title_status"].transform([title_status])[0],
    "transmission": encoders["transmission"].transform([transmission])[0],
    "drive": encoders["drive"].transform([drive])[0],
    "type": encoders["type"].transform([vehicle_type])[0],
    "paint_color": encoders["paint_color"].transform([paint_color])[0],
    "state": encoders["state"].transform([state_abbrev])[0],
}

input_df = pd.DataFrame([input_data])[feature_cols]


# -----------------------------------------------
# Prediction Display
# -----------------------------------------------
col1, col2, col3 = st.columns([1, 2, 1])

if predict_clicked:
    prediction = model.predict(input_df)[0]
    prediction = max(prediction, 0)
    st.session_state["last_prediction"] = prediction
    st.session_state["last_desc"] = (
        f"{year} {display_name(manufacturer)} | "
        f"{display_name(vehicle_type)} | "
        f"{odometer:,} miles | {display_name(condition)}"
    )

if "last_prediction" in st.session_state:
    with col2:
        st.markdown(f"""
        <div class="prediction-card">
            <div class="label">Estimated Vehicle Price</div>
            <div class="price">${st.session_state['last_prediction']:,.0f}</div>
            <div class="label">{st.session_state['last_desc']}</div>
        </div>
        """, unsafe_allow_html=True)
else:
    with col2:
        st.markdown("""
        <div class="waiting-card">
            <div class="message">Configure vehicle features in the sidebar</div>
            <div class="sub">Then click <strong>Predict Price</strong> to get an estimate</div>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------
# Vehicle Summary
# -----------------------------------------------
st.markdown('<div class="section-header">Vehicle Summary</div>', unsafe_allow_html=True)

col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    st.metric("Year", str(year))
    st.metric("Manufacturer", display_name(manufacturer))
    st.metric("Condition", display_name(condition))
with col_b:
    st.metric("Odometer", f"{odometer:,} mi")
    st.metric("Cylinders", display_name(cylinders_str))
    st.metric("Fuel Type", display_name(fuel))
with col_c:
    st.metric("Transmission", display_name(transmission))
    st.metric("Drive", display_name(drive))
    st.metric("Vehicle Type", display_name(vehicle_type))
with col_d:
    st.metric("Paint Color", display_name(paint_color))
    st.metric("State", state_full)
    st.metric("Car Age", f"{car_age} years")


# -----------------------------------------------
# Model Performance
# -----------------------------------------------
st.markdown(
    '<div class="section-header">Model Performance Comparison</div>',
    unsafe_allow_html=True,
)

if metrics:
    metric_cols = st.columns(len(metrics))
    for i, m in enumerate(metrics):
        with metric_cols[i]:
            is_best = m["model"] == "XGBoost"
            border_color = "#11998e" if is_best else "#e9ecef"
            bg_color = "#f0fdf4" if is_best else "#f8f9fa"
            badge = " (Best)" if is_best else ""
            st.markdown(f"""
            <div style="background: {bg_color}; border: 2px solid {border_color};
                        border-radius: 12px; padding: 1.2rem; text-align: center;">
                <div style="font-size: 1.1rem; font-weight: 700; color: #333;
                            margin-bottom: 0.8rem;">
                    {m['model']}{badge}
                </div>
                <div style="margin-bottom: 0.5rem;">
                    <span style="font-size: 0.8rem; color: #666;">MAE</span><br>
                    <span style="font-size: 1.1rem; font-weight: 600;">
                        ${m['mae']:,.0f}</span>
                </div>
                <div style="margin-bottom: 0.5rem;">
                    <span style="font-size: 0.8rem; color: #666;">RMSE</span><br>
                    <span style="font-size: 1.1rem; font-weight: 600;">
                        ${m['rmse']:,.0f}</span>
                </div>
                <div>
                    <span style="font-size: 0.8rem; color: #666;">R2 Score</span><br>
                    <span style="font-size: 1.2rem; font-weight: 700;
                           color: {'#11998e' if is_best else '#333'};">
                        {m['r2']:.4f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


# -----------------------------------------------
# Visualizations Gallery
# -----------------------------------------------
st.markdown(
    '<div class="section-header">Data Insights & Visualizations</div>',
    unsafe_allow_html=True,
)

plot_files = {
    "Model Comparison": "model_comparison.png",
    "Feature Importance": "feature_importance.png",
    "Predicted vs Actual": "predicted_vs_actual.png",
    "Price Distribution": "price_distribution.png",
    "Price by Manufacturer": "price_by_manufacturer.png",
    "Price by Vehicle Type": "price_by_type.png",
    "Price vs Odometer": "price_vs_odometer.png",
    "Price by Fuel Type": "price_by_fuel.png",
    "Correlation Heatmap": "correlation_heatmap.png",
}

available_plots = {
    k: v for k, v in plot_files.items()
    if os.path.exists(os.path.join(PLOTS_DIR, v))
}

if available_plots:
    tabs = st.tabs(list(available_plots.keys()))
    for tab, (name, filename) in zip(tabs, available_plots.items()):
        with tab:
            st.image(os.path.join(PLOTS_DIR, filename), use_container_width=True)
else:
    st.info("No plots found. Run the training pipeline to generate them.")


# -----------------------------------------------
# Footer
# -----------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #999; font-size: 0.85rem; padding: 1rem 0;">
        <strong>Vehicle Price Prediction</strong> &middot;
        Built with Python, Scikit-learn, XGBoost &amp; Streamlit<br>
        Data: 156,789 used car listings &middot; Model: XGBoost Regressor
    </div>
    """,
    unsafe_allow_html=True,
)
