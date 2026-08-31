from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.realtime_ai_platform.config import settings
from src.realtime_ai_platform.pipeline.features import RAW_COLUMNS, build_features, split_target


def load_training_data(data_path: str, sample_rows: int | None) -> pd.DataFrame:
    if not data_path:
        raise ValueError("DATA_PATH or --data-path must point to the transaction CSV")

    df = pd.read_csv(data_path)
    if sample_rows and len(df) > sample_rows:
        fraud = df[df["isFraud"] == 1]
        normal_sample_size = max(sample_rows - len(fraud), 1)
        normal = df[df["isFraud"] == 0].sample(n=normal_sample_size, random_state=42)
        df = pd.concat([fraud, normal], ignore_index=True).sample(frac=1, random_state=42)
    return df


def reference_profile(features: pd.DataFrame) -> dict:
    profile: dict[str, dict] = {}
    for column in features.columns:
        if column == "type":
            profile[column] = {
                "kind": "categorical",
                "distribution": features[column].value_counts(normalize=True).to_dict(),
            }
        else:
            profile[column] = {
                "kind": "numeric",
                "sample": features[column].dropna().sample(
                    n=min(10000, features[column].notna().sum()),
                    random_state=42,
                ).tolist(),
                "mean": float(features[column].mean()),
                "std": float(features[column].std() or 0.0),
            }
    return profile


def train_model(data_path: str, model_dir: Path, sample_rows: int | None = None) -> dict:
    raw = load_training_data(data_path, sample_rows)
    x, y = split_target(raw)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    numeric_features = [column for column in x.columns if column != "type"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("type", OneHotEncoder(handle_unknown="ignore"), ["type"]),
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=80,
                    max_depth=14,
                    min_samples_leaf=4,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name=f"fraud-rf-{datetime.now(timezone.utc).isoformat()}"):
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        metrics = {
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "average_precision": float(average_precision_score(y_test, probabilities)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
        }

        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        mlflow.log_param("rows", len(raw))
        mlflow.log_param("features", ",".join(x.columns))

        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_dir / "model.joblib")
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "raw_columns": RAW_COLUMNS,
            "feature_columns": list(build_features(raw).columns),
            "metrics": metrics,
            "reference_profile": reference_profile(x_train),
        }
        (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_dict(metadata, "metadata.json")
        return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=settings.data_path)
    parser.add_argument("--model-dir", default=str(settings.model_dir))
    parser.add_argument("--sample-rows", type=int, default=settings.train_sample_rows)
    args = parser.parse_args()

    metadata = train_model(args.data_path, Path(args.model_dir), args.sample_rows)
    print(json.dumps({"status": "trained", "metrics": metadata["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
