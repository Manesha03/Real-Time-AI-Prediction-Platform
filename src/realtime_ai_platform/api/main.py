from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from src.realtime_ai_platform.config import settings
from src.realtime_ai_platform.pipeline.features import build_features

PREDICTIONS = Counter("fraud_predictions_total", "Total prediction requests", ["label"])
LATENCY = Histogram("fraud_prediction_latency_seconds", "Prediction request latency")


class Transaction(BaseModel):
    step: int = Field(ge=0)
    type: str
    amount: float = Field(ge=0)
    oldbalanceOrg: float = Field(ge=0)
    newbalanceOrig: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)
    newbalanceDest: float = Field(ge=0)


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    model_created_at: str | None = None


def load_model() -> tuple[object, dict]:
    model_path = settings.model_dir / "model.joblib"
    metadata_path = settings.model_dir / "metadata.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return joblib.load(model_path), metadata


app = FastAPI(title="Real-Time AI Prediction Platform", version="1.0.0")


@app.get("/")
def root() -> dict:
    return {
        "service": "Real-Time AI Prediction Platform API",
        "status_endpoint": "/health",
        "api_docs": "/docs",
        "prediction_endpoint": "/predict",
        "metrics_endpoint": "/metrics",
    }


@app.get("/health")
def health() -> dict:
    model_exists = (Path(settings.model_dir) / "model.joblib").exists()
    return {"status": "ok" if model_exists else "model_missing", "model_exists": model_exists}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction) -> PredictionResponse:
    started = perf_counter()
    try:
        model, metadata = load_model()
        features = build_features(pd.DataFrame([transaction.model_dump()]))
        probability = float(model.predict_proba(features)[0][1])
        is_fraud = probability >= 0.5
        PREDICTIONS.labels(label=str(is_fraud).lower()).inc()
        return PredictionResponse(
            is_fraud=is_fraud,
            fraud_probability=probability,
            model_created_at=metadata.get("created_at"),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        LATENCY.observe(perf_counter() - started)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
