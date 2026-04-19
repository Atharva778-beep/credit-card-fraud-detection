import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Fraud Detector", layout="wide", page_icon="🔍")

@st.cache_resource
def load_artifacts():
    model = joblib.load("output/fraud_model.pkl")
    feature_names = joblib.load("output/feature_names.pkl")
    return model, feature_names

model, feature_names = load_artifacts()

st.title("🔍 Credit Card Fraud Detector")
st.markdown("Random Forest fraud detection app using the same features as the trained notebook model.")

st.sidebar.header("📊 Transaction Details")
amount = st.sidebar.number_input("💰 Transaction Amount", min_value=0.0, value=100.0, step=10.0, format="%.2f")
hour = st.sidebar.slider("🕐 Hour of Day (0-23)", 0, 23, 14)
v14 = st.sidebar.number_input("🔑 V14 (Top Fraud Signal)", value=-5.0, step=0.1, format="%.2f")
v10 = st.sidebar.number_input("V10", value=-2.0, step=0.1, format="%.2f")
v12 = st.sidebar.number_input("V12", value=0.0, step=0.1, format="%.2f")
v17 = st.sidebar.number_input("V17", value=-1.0, step=0.1, format="%.2f")

if st.sidebar.button("🚨 ANALYZE FRAUD RISK", type="primary", use_container_width=True):
    time_seconds = hour * 3600
    is_high_amount = 1 if amount > 200 else 0
    log_amount = np.log1p(amount)

    row = {
        'Time': time_seconds,
        'Amount': amount,
        'Hour': hour,
        'Is_high_amount': is_high_amount,
        'Log_amount': log_amount,
        'V1': 0.0, 'V2': 1.0, 'V3': -2.0, 'V4': -1.0, 'V5': 0.0,
        'V6': 0.0, 'V7': 0.2, 'V8': 0.0, 'V9': 0.0, 'V10': v10,
        'V11': 0.5, 'V12': v12, 'V13': 0.0, 'V14': v14, 'V15': 0.0,
        'V16': -0.5, 'V17': v17, 'V18': 0.0, 'V19': 0.0, 'V20': 0.0,
        'V21': 0.0, 'V22': 0.0, 'V23': 0.0, 'V24': 0.0, 'V25': 0.0,
        'V26': 0.0, 'V27': 0.0, 'V28': 0.0
    }

    input_data = pd.DataFrame([row])
    input_data = input_data[feature_names]

    fraud_proba = model.predict_proba(input_data)[0, 1]
    prediction = "FRAUD ⚠️" if fraud_proba > 0.5 else "LEGIT ✅"

    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Fraud Probability", f"{fraud_proba:.1%}")
    with col2:
        if fraud_proba > 0.7:
            st.error("🚨 HIGH RISK - BLOCK IMMEDIATELY")
        elif fraud_proba > 0.3:
            st.warning("⚠️ MEDIUM RISK - MANUAL REVIEW")
        else:
            st.success("✅ LOW RISK - APPROVE")

    st.markdown(f"**Final Verdict: {prediction}**")

    st.subheader("📈 Key Fraud Signals")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("V14", f"{v14:.2f}")
    c2.metric("V10", f"{v10:.2f}")
    c3.metric("Amount", f"₹{amount:,.2f}")
    c4.metric("Hour", f"{hour}:00")

with st.sidebar.expander("📊 Model Stats"):
    st.metric("ROC-AUC", "0.946")
    st.metric("Precision", "97.1%")
    st.metric("Recall", "71.6%")
    st.metric("F1-Score", "82.4%")

st.sidebar.markdown("---")
st.sidebar.caption("Built from the notebook-trained fraud detection model.")