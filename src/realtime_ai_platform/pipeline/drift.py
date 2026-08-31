from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.realtime_ai_platform.config import settings
from src.realtime_ai_platform.pipeline.features import build_features


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) <= 2:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)
    expected_percents = np.maximum(expected_counts / max(expected_counts.sum(), 1), 1e-6)
    actual_percents = np.maximum(actual_counts / max(actual_counts.sum(), 1), 1e-6)
    return float(np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents)))


def categorical_shift(expected_distribution: dict[str, float], actual: pd.Series) -> float:
    actual_distribution = actual.value_counts(normalize=True).to_dict()
    categories = set(expected_distribution) | set(actual_distribution)
    return float(
        sum(abs(expected_distribution.get(category, 0.0) - actual_distribution.get(category, 0.0)) for category in categories)
        / 2
    )


def detect_drift(data_path: str, model_dir: Path, reports_dir: Path, threshold: float) -> dict:
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found at {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    current = build_features(pd.read_csv(data_path))
    profile = metadata["reference_profile"]

    feature_scores = {}
    for column, reference in profile.items():
        if reference["kind"] == "categorical":
            feature_scores[column] = categorical_shift(reference["distribution"], current[column])
            continue

        expected = np.array(reference.get("sample", []), dtype=float)
        actual = current[column].dropna().to_numpy(dtype=float)
        feature_scores[column] = population_stability_index(expected, actual)

    max_score = max(feature_scores.values()) if feature_scores else 0.0
    result = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "threshold": threshold,
        "max_score": max_score,
        "drift_detected": max_score >= threshold,
        "feature_scores": feature_scores,
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "drift_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    html_rows = "\n".join(
        f"<tr><td>{feature}</td><td>{score:.4f}</td></tr>" for feature, score in sorted(feature_scores.items())
    )
    (reports_dir / "drift_report.html").write_text(
        f"<html><body><h1>Data Drift Report</h1><p>Drift detected: {result['drift_detected']}</p>"
        f"<table><tr><th>Feature</th><th>Score</th></tr>{html_rows}</table></body></html>",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=settings.data_path)
    parser.add_argument("--model-dir", default=str(settings.model_dir))
    parser.add_argument("--reports-dir", default=str(settings.reports_dir))
    parser.add_argument("--threshold", type=float, default=settings.drift_threshold)
    args = parser.parse_args()

    result = detect_drift(args.data_path, Path(args.model_dir), Path(args.reports_dir), args.threshold)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
