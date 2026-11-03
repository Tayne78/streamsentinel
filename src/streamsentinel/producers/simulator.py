"""Replays the market simulator in real time into Kafka, with labelled anomalies.

    python -m streamsentinel.producers.simulator

Useful offline, for demos, and because the labels let the dashboard show live precision/recall.
"""

from __future__ import annotations

import logging
import os
import signal
import time

from streamsentinel.config import Settings
from streamsentinel.kafka_io import encode, make_producer, wait_for_kafka
from streamsentinel.simulator import MarketSimulator

log = logging.getLogger("streamsentinel.simulator")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = Settings.from_env()
    rate = float(os.getenv("SIM_ANOMALY_RATE", "0.03"))
    sim = MarketSimulator(
        window_seconds=settings.window_seconds,
        anomaly_rate=rate,
        seed=int(os.getenv("SIM_SEED", str(int(time.time())))),
    )
    wait_for_kafka(settings.kafka_bootstrap)
    producer = make_producer(settings.kafka_bootstrap, "simulator-producer")

    running = True

    def _stop(*_: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    width = settings.window_seconds * 1000
    window = int(time.time() * 1000) // width * width
    log.info("Simulating %s, anomaly rate %.1f%%", ", ".join(sim.symbols), rate * 100)
    while running:
        trades = sim.generate_window(window)
        injected = trades.dropna(subset=["label"]).groupby("symbol")["label"].first()
        for symbol, label in injected.items():
            log.info("Injecting %s into %s", label, symbol)
        for trade in trades.to_dict("records"):
            if not running:
                break
            delay = trade["ts"] / 1000 - time.time()
            if delay > 0:
                time.sleep(delay)  # emit at event time, like a real feed
            trade["label"] = trade["label"] if isinstance(trade["label"], str) else None
            producer.produce(settings.trades_topic, key=trade["symbol"], value=encode(trade))
            producer.poll(0)
        window += width
    producer.flush(10)


if __name__ == "__main__":
    main()
