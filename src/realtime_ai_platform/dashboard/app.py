from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from src.realtime_ai_platform.config import settings

st.set_page_config(page_title="AI Prediction Monitor", layout="wide")
st.title("Real-Time AI Prediction Monitor")

metadata_path = Path(settings.model_dir) / "metadata.json"
drift_path = Path(settings.reports_dir) / "drift_report.json"

left, right = st.columns(2)

with left:
    st.subheader("Model")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        st.write("Created at", metadata.get("created_at"))
        st.dataframe(pd.DataFrame([metadata.get("metrics", {})]))
    else:
        st.warning("No trained model artifact found.")

with right:
    st.subheader("Drift")
    if drift_path.exists():
        drift = json.loads(drift_path.read_text(encoding="utf-8"))
        st.metric("Max drift score", f"{drift.get('max_score', 0):.4f}")
        st.write("Drift detected", drift.get("drift_detected"))
        st.bar_chart(pd.Series(drift.get("feature_scores", {})))
    else:
        st.info("No drift report generated yet.")

st.subheader("Live Prediction Probe")
with st.form("prediction"):
    transaction_type = st.selectbox("Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"])
    amount = st.number_input("Amount", min_value=0.0, value=181.0)
    oldbalance_org = st.number_input("Old origin balance", min_value=0.0, value=181.0)
    newbalance_orig = st.number_input("New origin balance", min_value=0.0, value=0.0)
    oldbalance_dest = st.number_input("Old destination balance", min_value=0.0, value=0.0)
    newbalance_dest = st.number_input("New destination balance", min_value=0.0, value=0.0)
    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "step": 1,
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalance_org,
        "newbalanceOrig": newbalance_orig,
        "oldbalanceDest": oldbalance_dest,
        "newbalanceDest": newbalance_dest,
    }
    response = requests.post("http://api:8000/predict", json=payload, timeout=10)
    st.json(response.json())
