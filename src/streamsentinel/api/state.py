"""In-memory live state for the API, plus SQLite persistence for anomalies."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class AnomalyStore:
    """Anomalies survive API restarts; the window history does not need to."""

    def __init__(self, path: str) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS anomalies ("
                " symbol TEXT NOT NULL, window_start INTEGER NOT NULL, payload TEXT NOT NULL,"
                " PRIMARY KEY (symbol, window_start))"
            )
            self._conn.commit()

    def add(self, score: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO anomalies VALUES (?, ?, ?)",
                (score["symbol"], score["window_start"], json.dumps(score)),
            )
            self._conn.commit()

    def recent(self, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT payload FROM anomalies ORDER BY window_start DESC LIMIT ?", (limit,)
            ).fetchall()
        return [json.loads(r[0]) for r in reversed(rows)]


class Metrics:
    """Prometheus metrics: the pipeline is observable like any other production service."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.windows = Counter(
            "streamsentinel_windows_scored_total",
            "Feature windows scored",
            ["symbol"],
            registry=self.registry,
        )
        self.anomalies = Counter(
            "streamsentinel_anomalies_total",
            "Anomalies detected",
            ["symbol", "type"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "streamsentinel_detection_latency_seconds",
            "Window close to detection",
            buckets=(1, 2, 3, 5, 8, 13, 21, 34),
            registry=self.registry,
        )
        self.last_window = Gauge(
            "streamsentinel_last_window_end_timestamp_seconds",
            "End of the newest scored window",
            ["symbol"],
            registry=self.registry,
        )


class LiveState:
    def __init__(
        self,
        store: AnomalyStore,
        max_windows: int = 360,
        max_anomalies: int = 500,
        metrics: Metrics | None = None,
    ) -> None:
        self.store = store
        self.metrics = metrics or Metrics()
        self.max_windows = max_windows
        self._windows: dict[str, deque[dict[str, Any]]] = {}
        self._anomalies: deque[dict[str, Any]] = deque(store.recent(max_anomalies), max_anomalies)
        self._latencies: deque[float] = deque(maxlen=500)
        self._counts = {"windows": 0, "anomalies": 0, "tp": 0, "fp": 0, "fn": 0, "labelled": 0}
        self._model = {"mode": None, "version": None}
        self._lock = threading.Lock()
        self._subscribers: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    # -- live updates -------------------------------------------------------------------------
    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    def _broadcast(self, message: dict[str, Any]) -> None:
        for queue in list(self._subscribers):
            if queue.full():
                continue  # slow client: drop rather than block the pipeline
            queue.put_nowait(message)

    # -- ingestion (called from the Kafka consumer thread) ------------------------------------
    def ingest(self, score: dict[str, Any]) -> None:
        symbol = score["symbol"]
        with self._lock:
            self._windows.setdefault(symbol, deque(maxlen=self.max_windows)).append(score)
            self._counts["windows"] += 1
            self._model = {
                "mode": score.get("detector_mode"),
                "version": score.get("model_version"),
            }
            latency = (score.get("detected_at", 0) - score["window_end"]) / 1000
            if latency >= 0:
                self._latencies.append(latency)
                self.metrics.latency.observe(latency)
            if score.get("source") == "simulator" and not score.get("warmup"):
                self._update_ground_truth(score)
            if score.get("is_anomaly"):
                self._counts["anomalies"] += 1
                self._anomalies.append(score)

        self.metrics.windows.labels(symbol).inc()
        self.metrics.last_window.labels(symbol).set(score["window_end"] / 1000)
        if score.get("is_anomaly"):
            self.metrics.anomalies.labels(symbol, score.get("anomaly_type") or "unknown").inc()
            self.store.add(score)
        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._broadcast, {"type": "score", "data": score})

    def _update_ground_truth(self, score: dict[str, Any]) -> None:
        injected, flagged = score.get("label") is not None, bool(score.get("is_anomaly"))
        self._counts["labelled"] += 1
        if injected and flagged:
            self._counts["tp"] += 1
        elif flagged:
            self._counts["fp"] += 1
        elif injected:
            self._counts["fn"] += 1

    # -- queries ------------------------------------------------------------------------------
    def symbols(self) -> list[str]:
        with self._lock:
            return sorted(self._windows)

    def windows(self, symbol: str, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._windows.get(symbol, ()))[-limit:]

    def anomalies(self, limit: int, symbol: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            items = [a for a in self._anomalies if symbol is None or a["symbol"] == symbol]
        return items[-limit:]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            counts = dict(self._counts)
            latencies = list(self._latencies)
            model = dict(self._model)
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        ground_truth = None
        if counts["labelled"]:
            precision = tp / (tp + fp) if tp + fp else None
            recall = tp / (tp + fn) if tp + fn else None
            ground_truth = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}
        return {
            "windows_scored": counts["windows"],
            "anomalies": counts["anomalies"],
            "latency_p50_s": float(np.percentile(latencies, 50)) if latencies else None,
            "latency_p95_s": float(np.percentile(latencies, 95)) if latencies else None,
            "model": model,
            "ground_truth": ground_truth,
        }
