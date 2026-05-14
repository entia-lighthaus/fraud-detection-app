# ─────────────────────────────────────────────────────────────
#  FRAUD DETECTION APP
#  Built with Streamlit + LightGBM + SHAP
#  Author: Innocentia Duru
#  This app allows users to input transaction details and get a fraud prediction,
#  along with an explanation of the prediction using SHAP values.
# ─────────────────────────────────────────────────────────────

from sklearn import impute
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from utils.feature_engineering import build_features

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title = "Fraud Detection System",
    page_icon  = "🛡️",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── Load artifacts (cached so they load once) ─────────────────
# We load the trained model, imputer, isolation forest, feature columns, and threshold from disk.
# Using @st.cache_resource ensures that these artifacts are loaded only once and cached for future use, improving performance.
@st.cache_resource
def load_artifacts():
    import json
    import lightgbm as lgb

    # LightGBM native loader — fully version independent
    booster   = lgb.Booster(model_file="artifacts/lgbm_model.txt")

    # Isolation Forest — protocol=2 pkl
    iso_forest = joblib.load("artifacts/iso_forest.pkl")

    # Plain JSON files — no version dependency at all
    with open("artifacts/feature_cols.json") as f:
        features = json.load(f)

    with open("artifacts/threshold.json") as f:
        threshold = float(json.load(f))

    with open("artifacts/imputer_medians.json") as f:
        medians = json.load(f)

    return booster, iso_forest, features, threshold, medians

booster, iso_forest, feature_cols, threshold, medians = load_artifacts()

def apply_medians(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN values using training medians. Version-independent."""
    df = df.copy()
    for col in df.columns:
        if col in medians:
            df[col] = df[col].fillna(medians[col])
    return df


# ── Sidebar navigation ────────────────────────────────────────
# The sidebar contains the app title, navigation options for different pages, and a summary of the model's key parameters.
st.sidebar.image(
    "https://img.icons8.com/color/96/shield.png", width=60
)
st.sidebar.title("Fraud Detection")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Transaction Checker",
     "Model Performance",
     "Batch Scanner"],
    index = 0
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Model threshold:** `{threshold:.4f}`\n\n"
    f"**Features:** `{len(feature_cols)}`\n\n"
    f"**Algorithm:** LightGBM + Isolation Forest"
)

