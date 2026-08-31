from __future__ import annotations

import json

from src.realtime_ai_platform.config import settings
from src.realtime_ai_platform.pipeline.drift import detect_drift
from src.realtime_ai_platform.pipeline.train import train_model


def retrain_if_needed() -> dict:
    drift = detect_drift(
        settings.data_path,
        settings.model_dir,
        settings.reports_dir,
        settings.drift_threshold,
    )
    if settings.force_retrain or drift["drift_detected"]:
        metadata = train_model(settings.data_path, settings.model_dir, settings.train_sample_rows)
        return {"status": "retrained", "drift": drift, "metrics": metadata["metrics"]}
    return {"status": "skipped", "drift": drift}


def main() -> None:
    print(json.dumps(retrain_if_needed(), indent=2))


if __name__ == "__main__":
    main()
