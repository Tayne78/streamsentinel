"""Detector service: Kafka ``features`` -> score every window -> Kafka ``scores``.

Loads the champion model from the MLflow registry and keeps polling for a newer champion, so a
retrained model goes live without restarting the service or losing the per-symbol baselines.

    python -m streamsentinel.services.detector_service
"""

from __future__ import annotations

import logging
import signal
import time

from streamsentinel.config import Settings
from streamsentinel.detector import DetectorConfig, StreamingDetector
from streamsentinel.kafka_io import encode, iter_json, make_consumer, make_producer, wait_for_kafka
from streamsentinel.registry import load_champion

log = logging.getLogger("streamsentinel.detector")

MODEL_REFRESH_S = 60


class ModelWatcher:
    """Polls the registry and hot-swaps the detector's model when the champion changes."""

    def __init__(self, settings: Settings, detector: StreamingDetector) -> None:
        self.settings = settings
        self.detector = detector
        self._last_check = 0.0

    def maybe_refresh(self, force: bool = False) -> None:
        if not force and time.monotonic() - self._last_check < MODEL_REFRESH_S:
            return
        self._last_check = time.monotonic()
        loaded = load_champion(
            self.settings.mlflow_tracking_uri, self.settings.model_name, self.settings.model_alias
        )
        if loaded is None:
            return
        config, forest = loaded
        config = config.with_overrides(self.settings.z_threshold_override)
        if config.model_version != self.detector.config.model_version:
            self.detector.swap_model(config, forest)
            log.info(
                "Using model v%s: mode=%s z_threshold=%.2f if_threshold=%.4f",
                config.model_version,
                config.mode,
                config.z_threshold,
                config.if_threshold,
            )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = Settings.from_env()
    detector = StreamingDetector(
        DetectorConfig().with_overrides(settings.z_threshold_override), iforest=None
    )
    watcher = ModelWatcher(settings, detector)
    watcher.maybe_refresh(force=True)
    if detector.config.model_version is None:
        log.info("No registered model yet: running z-score only until training finishes")

    wait_for_kafka(settings.kafka_bootstrap)
    consumer = make_consumer(settings.kafka_bootstrap, "detector", [settings.features_topic])
    producer = make_producer(settings.kafka_bootstrap, "detector")

    running = True

    def _stop(*_: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    for window in iter_json(consumer):
        if not running:
            break
        watcher.maybe_refresh()
        if window is None:
            continue
        score = detector.process(window)
        producer.produce(settings.scores_topic, key=score.symbol, value=encode(score.to_dict()))
        producer.poll(0)
        if score.is_anomaly:
            log.warning(
                "ANOMALY %s %s severity=%.2f: %s",
                score.symbol,
                score.anomaly_type,
                score.severity,
                score.explanation,
            )
    consumer.close()
    producer.flush(10)


if __name__ == "__main__":
    main()
