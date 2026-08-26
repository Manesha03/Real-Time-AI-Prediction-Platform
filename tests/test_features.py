import pandas as pd

from src.realtime_ai_platform.pipeline.features import build_features, split_target


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "step": 1,
                "type": "TRANSFER",
                "amount": 181.0,
                "oldbalanceOrg": 181.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
                "isFraud": 1,
            }
        ]
    )


def test_build_features_adds_engineered_columns():
    features = build_features(sample_frame())

    assert "orig_error" in features.columns
    assert "dest_error" in features.columns
    assert features.loc[0, "orig_balance_delta"] == 181.0


def test_split_target_returns_binary_target():
    features, target = split_target(sample_frame())

    assert len(features) == 1
    assert target.iloc[0] == 1
