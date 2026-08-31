from __future__ import annotations

import logging
import time

from src.realtime_ai_platform.config import settings
from src.realtime_ai_platform.pipeline.retrain import retrain_if_needed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    while True:
        try:
            logging.info("running retraining check")
            logging.info("result=%s", retrain_if_needed())
        except Exception:
            logging.exception("retraining check failed")
        time.sleep(settings.retrain_interval_seconds)


if __name__ == "__main__":
    main()
