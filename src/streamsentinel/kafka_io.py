"""Thin wrappers around confluent-kafka so every service talks to Kafka the same way."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Iterator
from typing import Any

from confluent_kafka import Consumer, KafkaError, KafkaException, Producer

log = logging.getLogger(__name__)


def _default(obj: Any) -> Any:
    # numpy scalars and similar
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def encode(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, default=_default, separators=(",", ":")).encode()


def make_producer(bootstrap: str, client_id: str) -> Producer:
    return Producer(
        {
            "bootstrap.servers": bootstrap,
            "client.id": client_id,
            "linger.ms": 20,
            "compression.type": "lz4",
            "enable.idempotence": True,
        }
    )


def make_consumer(bootstrap: str, group_id: str, topics: Iterable[str]) -> Consumer:
    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": group_id,
            "auto.offset.reset": "latest",
            "enable.auto.commit": True,
        }
    )
    consumer.subscribe(list(topics))
    return consumer


def wait_for_kafka(bootstrap: str, timeout_s: float = 120.0) -> None:
    """Block until the broker answers metadata requests (containers start in any order)."""
    deadline = time.monotonic() + timeout_s
    probe = Producer({"bootstrap.servers": bootstrap})
    while True:
        try:
            probe.list_topics(timeout=5)
            return
        except KafkaException as exc:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Kafka at {bootstrap} not reachable") from exc
            log.info("Waiting for Kafka at %s ...", bootstrap)
            time.sleep(2)


def iter_json(consumer: Consumer, poll_timeout: float = 1.0) -> Iterator[dict[str, Any] | None]:
    """Yield decoded JSON messages; yields None on idle polls so callers can do housekeeping."""
    while True:
        msg = consumer.poll(poll_timeout)
        if msg is None:
            yield None
            continue
        if msg.error():
            if msg.error().code() != KafkaError._PARTITION_EOF:
                log.warning("Kafka error: %s", msg.error())
            continue
        try:
            yield json.loads(msg.value())
        except (ValueError, TypeError):
            log.warning("Skipping undecodable message on %s", msg.topic())
