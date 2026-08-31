from pathlib import Path
import os

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class Settings(BaseModel):
    data_path: str = os.getenv("DATA_PATH", "")
    model_dir: Path = Path(os.getenv("MODEL_DIR", "models/current"))
    reports_dir: Path = Path(os.getenv("REPORTS_DIR", "reports"))
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "file:mlruns")
    mlflow_experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "real-time-fraud-platform")
    train_sample_rows: int = int(os.getenv("TRAIN_SAMPLE_ROWS", "250000"))
    retrain_interval_seconds: int = int(os.getenv("RETRAIN_INTERVAL_SECONDS", "3600"))
    drift_threshold: float = float(os.getenv("DRIFT_THRESHOLD", "0.2"))
    force_retrain: bool = os.getenv("FORCE_RETRAIN", "false").lower() == "true"


settings = Settings()
