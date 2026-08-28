# Operations Runbook

## Train

```bash
python -m src.realtime_ai_platform.pipeline.train --data-path /path/to/PS_20174392719_1491204439457_log.csv
```

## Drift Check

```bash
python -m src.realtime_ai_platform.pipeline.drift --data-path /path/to/new_transactions.csv
```

## Retrain Once

```bash
python -m src.realtime_ai_platform.pipeline.retrain
```

## Deploy

1. Push the repository to GitHub.
2. Add cloud secrets such as `RENDER_DEPLOY_HOOK_URL`.
3. Build and run the Docker image in the target cloud environment.
4. Mount or publish the latest `models/current` artifact.

## Service Level Checks

- `/health` returns model availability.
- `/metrics` exposes Prometheus metrics.
- MLflow contains experiment history and model metrics.
- `reports/drift_report.json` records the latest drift status.
