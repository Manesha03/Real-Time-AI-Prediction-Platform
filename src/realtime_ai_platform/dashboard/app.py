from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from src.realtime_ai_platform.config import settings

st.set_page_config(page_title="AI Prediction Monitor", layout="wide")
st.title("Fraud Prediction Control Center")

metadata_path = Path(settings.model_dir) / "metadata.json"
drift_path = Path(settings.reports_dir) / "drift_report.json"
api_url = "http://api:8000"


@st.cache_data(show_spinner=False)
def load_sample_transactions(data_path: str) -> pd.DataFrame:
    if not data_path or not Path(data_path).exists():
        return pd.DataFrame()
    columns = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
    ]
    return pd.read_csv(data_path, usecols=columns, nrows=5000)


def risk_label(probability: float) -> tuple[str, str]:
    if probability >= 0.8:
        return "High fraud risk", "error"
    if probability >= 0.5:
        return "Possible fraud", "warning"
    if probability >= 0.2:
        return "Needs review", "info"
    return "Likely safe", "success"


def call_prediction_api(payload: dict) -> dict:
    try:
        response = requests.post(f"{api_url}/predict", json=payload, timeout=10)
    except requests.RequestException:
        response = requests.post("http://localhost:8000/predict", json=payload, timeout=10)
    response.raise_for_status()
    return response.json()

left, right = st.columns(2)

with left:
    st.subheader("Model Health")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metrics = metadata.get("metrics", {})
        st.metric("Model status", "Ready")
        st.write("Last trained", metadata.get("created_at"))
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Fraud recall": metrics.get("recall"),
                        "Precision": metrics.get("precision"),
                        "F1 score": metrics.get("f1"),
                        "ROC AUC": metrics.get("roc_auc"),
                    }
                ]
            ),
            hide_index=True,
        )
    else:
        st.warning("Model is not ready. Train the model before making predictions.")

with right:
    st.subheader("Data Drift")
    if drift_path.exists():
        drift = json.loads(drift_path.read_text(encoding="utf-8"))
        st.metric("Max drift score", f"{drift.get('max_score', 0):.4f}")
        if drift.get("drift_detected"):
            st.error("Incoming data has changed enough to trigger retraining.")
        else:
            st.success("Incoming data is close to the training pattern.")
        st.bar_chart(pd.Series(drift.get("feature_scores", {})))
    else:
        st.info("No drift report generated yet.")

st.subheader("Fraud Prediction")
samples = load_sample_transactions(settings.data_path)

if samples.empty:
    st.info("Dataset examples are not available. Enter a transaction manually.")
    selected = {}
else:
    fraud_count = int(samples["isFraud"].sum())
    total_count = len(samples)
    metric_cols = st.columns(3)
    metric_cols[0].metric("Dataset examples loaded", f"{total_count:,}")
    metric_cols[1].metric("Known fraud examples", f"{fraud_count:,}")
    metric_cols[2].metric("Known safe examples", f"{total_count - fraud_count:,}")

    example_mode = st.radio(
        "Transaction source",
        ["Use a dataset example", "Enter manually"],
        horizontal=True,
    )
    if example_mode == "Use a dataset example":
        row_number = st.slider("Dataset row", 0, len(samples) - 1, 0)
        selected = samples.iloc[row_number].to_dict()
        actual = "Fraud" if int(selected.get("isFraud", 0)) == 1 else "Not fraud"
        st.write("Actual dataset label", actual)
    else:
        selected = {}

with st.form("prediction"):
    transaction_type = st.selectbox(
        "Transaction type",
        ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"],
        index=["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"].index(selected.get("type", "TRANSFER")),
    )
    amount = st.number_input("Amount", min_value=0.0, value=float(selected.get("amount", 181.0)))
    oldbalance_org = st.number_input("Sender balance before", min_value=0.0, value=float(selected.get("oldbalanceOrg", 181.0)))
    newbalance_orig = st.number_input("Sender balance after", min_value=0.0, value=float(selected.get("newbalanceOrig", 0.0)))
    oldbalance_dest = st.number_input(
        "Receiver balance before",
        min_value=0.0,
        value=float(selected.get("oldbalanceDest", 0.0)),
    )
    newbalance_dest = st.number_input(
        "Receiver balance after",
        min_value=0.0,
        value=float(selected.get("newbalanceDest", 0.0)),
    )
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
    try:
        prediction = call_prediction_api(payload)
        probability = float(prediction["fraud_probability"])
        label, level = risk_label(probability)
        if level == "error":
            st.error(label)
        elif level == "warning":
            st.warning(label)
        elif level == "info":
            st.info(label)
        else:
            st.success(label)
        st.metric("Fraud probability", f"{probability:.1%}")
        st.write("Final decision", "Fraud" if prediction["is_fraud"] else "Not fraud")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
