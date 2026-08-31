from __future__ import annotations

import pandas as pd

TARGET = "isFraud"
RAW_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    missing = [column for column in RAW_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    data = data[RAW_COLUMNS]
    data["orig_balance_delta"] = data["oldbalanceOrg"] - data["newbalanceOrig"]
    data["dest_balance_delta"] = data["newbalanceDest"] - data["oldbalanceDest"]
    data["orig_error"] = data["orig_balance_delta"] - data["amount"]
    data["dest_error"] = data["dest_balance_delta"] - data["amount"]
    data["amount_to_old_orig_ratio"] = data["amount"] / data["oldbalanceOrg"].replace(0, 1)
    return data


def split_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET not in df.columns:
        raise ValueError(f"Training data must contain target column {TARGET!r}")
    return build_features(df), df[TARGET].astype(int)
