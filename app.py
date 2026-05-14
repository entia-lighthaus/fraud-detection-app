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


# ─────────────────────────────────────────────────────────────
#  PAGE 1 — TRANSACTION CHECKER
#  This page allows users to input details of a single transaction, which the model will analyze to predict whether it's fraudulent or legitimate.
#  The page is organized into three sections for input: Transaction Details, Identity & Geography, and Behavioural History.
# ─────────────────────────────────────────────────────────────

if page == "Transaction Checker":

    st.title("Fraud Transaction Checker")
    st.markdown(
        "Enter transaction details below. The model will score it, "
        "flag it as fraud or legitimate, and explain exactly why."
    )
    st.markdown("---")


    # ── Input form ────────────────────────────────────────────
    # We use Streamlit's column layout to organize the input fields into three sections: Transaction Details, Identity & Geography, and Behavioural History.
    col1, col2, col3 = st.columns(3)

    
    # Transaction Details inputs to capture necessary information about the transaction such as: amount, merchant category, whether the card was present, hour of transaction, and day of week.
    with col1:
        st.subheader("Transaction Details")
        amount      = st.number_input("Amount (₦)", min_value=0.5,
                                       max_value=500_000.0, value=1500.0, step=50.0)
        
        merchant_category = st.selectbox(
            "Merchant Category",
            ["online", "atm", "grocery", "dining",
             "travel", "gas", "retail"]
        )
        
        card_present = st.toggle("Card Present (physical card used)", value=False)
        hour         = st.slider("Hour of Transaction", 0, 23, 14)
        dayofweek    = st.selectbox(
            "Day of Week",
            [0,1,2,3,4,5,6],
            format_func=lambda x: ["Mon","Tue","Wed",
                                    "Thu","Fri","Sat","Sun"][x]
        )

    
    # Identity & Geography inputs to capture information about the user's location and device, which are important factors in fraud detection. 
    # This includes whether the transaction is cross-border, if the IP matches the user's home country, and details about the device used for the transaction.
    with col2:
        st.subheader("Identity & Geography")
        cross_border     = st.toggle("Cross-border Transaction", value=True)
        ip_country_match = st.toggle("IP Matches Home Country", value=False)
        ip_user_count    = st.number_input(
            "Users Sharing This IP", min_value=1,
            max_value=10_000, value=4000
        )

        st.subheader("Device")
        is_new_device           = st.toggle("New Device", value=True)
        is_emulator             = st.toggle("Emulator Detected", value=False)
        is_unknown_browser      = st.toggle("Unknown Browser", value=True)
        device_shared_users     = st.number_input(
            "Users Sharing This Device", min_value=1,
            max_value=500, value=3
        )
       
        user_device_count       = st.number_input(
            "Devices Used by This User", min_value=1,
            max_value=20, value=2
        )
        
        device_fraud_neighbourhood = st.slider(
            "Device Fraud Rate (historical)", 0.0, 1.0, 0.12
        )


    # This section contains Behavioral History inputs to capture the user's recent transaction patterns, which can provide important context for fraud detection. 
    # This includes the number of transactions in the last hour and 24 hours, total spend in the last 24 hours, average transaction amount in the last 7 days,
    # time since last transaction, percentage change in amount compared to the last transaction, transaction density at this merchant in the last 24 hours, and the diversity of merchant categories used by the user.
    with col3:
        st.subheader("Behavioural History")
        txn_count_1h  = st.number_input(
            "Transactions in Last 1 Hour", min_value=0, max_value=50, value=3
        )
       
        txn_count_24h = st.number_input(
            "Transactions in Last 24 Hours", min_value=0, max_value=200, value=5
        )
        
        sum_24h       = st.number_input(
            "Total Spend Last 24h (₦)", min_value=0.0,
            max_value=1_000_000.0, value=8000.0
        )
        
        avg_amount_7d = st.number_input(
            "Avg Transaction Amount Last 7 Days (₦)",
            min_value=0.0, max_value=500_000.0, value=2000.0
        )
        
        time_since_last_txn = st.number_input(
            "Seconds Since Last Transaction",
            min_value=0, max_value=604_800, value=180
        )
       
        amount_change_pct = st.number_input(
            "Amount Change % vs Last Transaction",
            min_value=-100.0, max_value=10_000.0, value=250.0
        )
        
        merchant_txn_density_24h = st.number_input(
            "Transactions at This Merchant (Last 24h)",
            min_value=0, max_value=10_000, value=3500
        )
        
        user_merchant_diversity = st.number_input(
            "Merchant Categories Used by User",
            min_value=1, max_value=7, value=4
        )

    st.markdown("---")
    run = st.button(" Analyse Transaction", use_container_width=True, type="primary")


    if run:
        # ── Build features ────────────────────────────────────
        txn = {
            "amount":                    amount,
            "hour":                      hour,
            "dayofweek":                 dayofweek,
            "cross_border":              cross_border,
            "ip_country_match":          ip_country_match,
            "card_present":              card_present,
            "is_emulator":               is_emulator,
            "is_unknown_browser":        is_unknown_browser,
            "is_new_device":             is_new_device,
            "device_shared_users":       device_shared_users,
            "user_device_count":         user_device_count,
            "txn_count_1h":              txn_count_1h,
            "txn_count_24h":             txn_count_24h,
            "sum_24h":                   sum_24h,
            "avg_amount_7d":             avg_amount_7d,
            "time_since_last_txn":       time_since_last_txn,
            "amount_change_pct":         amount_change_pct,
            "is_rapid_succession":       int(time_since_last_txn < 300),
            "is_first_txn":              0,
            "merchant_category":         merchant_category,
            "ip_user_count":             ip_user_count,
            "merchant_txn_density_24h":  merchant_txn_density_24h,
            "user_merchant_diversity":   user_merchant_diversity,
            "device_fraud_neighbourhood":device_fraud_neighbourhood,
        }

        X_raw          = build_features(txn)
        X_imputed      = apply_medians(X_raw)
        
        anomaly_score  = -iso_forest.decision_function(X_imputed)[0]
        X_imputed["anomaly_score"] = anomaly_score
        
        feature_cols_final = feature_cols + ["anomaly_score"]
        X_final        = X_imputed[feature_cols_final]
       
        fraud_prob     = float(booster.predict(X_final)[0])
        is_fraud       = fraud_prob >= threshold

        # ── Result banner ─────────────────────────────────────
        st.markdown("## Result")
        r1, r2, r3 = st.columns(3)

        with r1:
            if is_fraud:
                st.error("###  FRAUD DETECTED")
            else:
                st.success("###  LEGITIMATE")

        with r2:
            fig_gauge = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = round(fraud_prob * 100, 1),
                title = {"text": "Fraud Probability (%)"},
                gauge = {
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#F44336" if is_fraud else "#4CAF50"},
                    "steps": [
                        {"range": [0,  30], "color": "#E8F5E9"},
                        {"range": [30, 60], "color": "#FFF9C4"},
                        {"range": [60, 100],"color": "#FFEBEE"},
                    ],
                    "threshold": {
                        "line":  {"color": "black", "width": 3},
                        "thickness": 0.75,
                        "value": threshold * 100
                    }
                }
            ))
            fig_gauge.update_layout(height=220, margin=dict(t=30,b=0,l=20,r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with r3:
            st.metric("Fraud Score",    f"{fraud_prob:.4f}")
            st.metric("Threshold",      f"{threshold:.4f}")
            st.metric("Anomaly Score",  f"{anomaly_score:.4f}")
            st.metric("Decision",       " BLOCK" if is_fraud else " PASS")


        # ── SHAP explanation ──────────────────────────────────
        st.markdown("---")
        st.markdown("### Why did the model make this decision?")

        explainer   = shap.TreeExplainer(booster)
        shap_vals   = explainer.shap_values(X_final)
        sv          = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
        sv_row      = sv[0]

        shap_df = pd.DataFrame({
            "Feature": feature_cols_final,
            "Value":   X_final.iloc[0].values,
            "SHAP":    sv_row
        }).reindex(pd.Series(sv_row).abs().sort_values(ascending=False).index)

        top10 = shap_df.head(10)

        s1, s2 = st.columns(2)

        with s1:
            st.markdown("**Top 10 factors driving this decision:**")
            for _, row in top10.iterrows():
                direction = "🔴 toward fraud" if row["SHAP"] > 0 else "🔵 toward legit"
                st.markdown(
                    f"- **{row['Feature']}** = `{row['Value']:.3f}` "
                    f"→ `{row['SHAP']:+.4f}` {direction}"
                )

        with s2:
            colors = ["#F44336" if v > 0 else "#2196F3"
                      for v in top10["SHAP"]]
            fig_shap = go.Figure(go.Bar(
                x          = top10["SHAP"],
                y          = top10["Feature"],
                orientation= "h",
                marker_color = colors,
            ))
            fig_shap.update_layout(
                title  = "SHAP Values — Feature Contributions",
                xaxis_title = "SHAP Value (+ = fraud, - = legit)",
                height = 380,
                margin = dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_shap, use_container_width=True)



# ─────────────────────────────────────────────────────────────
#  PAGE 2 — MODEL PERFORMANCE
#  This page displays key performance metrics from the model evaluation, including PR-AUC, ROC-AUC, fraud recall, and fraud precision. 
#  It also shows a bar chart of the top 20 feature importances derived from the trained LightGBM model, and provides a summary of the model's configuration and components used in the pipeline.
# ─────────────────────────────────────────────────────────────

elif page == " Model Performance":

    st.title(" Model Performance")
    st.markdown(
        "Key metrics from the test set evaluation. "
        "All charts reflect the model trained in Google Colab."
    )
    st.markdown("---")


    # ── Metric cards ──────────────────────────────────────────
    # We use Streamlit's metric component to display the key performance metrics in a visually appealing way. 
    # Each metric card shows the value of the metric along with a brief description or comparison to a baseline (e.g., random chance).
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PR-AUC",          "0.9856", "↑ vs random 0.012")
    m2.metric("ROC-AUC",         "0.9998")
    m3.metric("Fraud Recall",    "94.1%",  "at chosen threshold")
    m4.metric("Fraud Precision", "93.2%",  "at chosen threshold")

    st.markdown("---")


    # ── Feature importance ────────────────────────────────────
    # This section visualizes the top 20 feature importances from the trained LightGBM model using a horizontal bar chart.
    # The importance values are derived from the model's built-in feature importance attribute, which indicates how much each feature contributed to the model's predictions.
    st.subheader("Feature Importance (from trained model)")

    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        "Feature":    feature_cols,
        "Importance": importances
    }).sort_values("Importance", ascending=True).tail(20)

    fig_imp = px.bar(
        imp_df, x="Importance", y="Feature",
        orientation = "h",
        color       = "Importance",
        color_continuous_scale = "reds",
        title       = "Top 20 Feature Importances"
    )

    fig_imp.update_layout(height=550, showlegend=False,
                          coloraxis_showscale=False)
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")
    st.subheader("Model Configuration")
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("""
        | Parameter | Value |
        |---|---|
        | Algorithm | LightGBM |
        | n_estimators | 1000 (early stopping) |
        | learning_rate | 0.05 |
        | num_leaves | 63 |
        | max_depth | 7 |
        | feature_fraction | 0.8 |
        """)

    with c2:
        st.markdown("""
        | Component | Purpose |
        |---|---|
        | SMOTE | Oversample fraud to 10% |
        | SimpleImputer | Fill NaN with median |
        | Isolation Forest | Anomaly score feature |
        | Threshold tuning | Best F1 decision boundary |
        """)