"""FastAPI service: REST + WebSocket access to live scores, Prometheus metrics, dashboard.

uvicorn streamsentinel.api.main:app --port 8000
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from streamsentinel.api.state import AnomalyStore, LiveState
from streamsentinel.config import Settings

log = logging.getLogger("streamsentinel.api")
STATIC = Path(__file__).parent / "static"


def _consume(state: LiveState, settings: Settings, stop: threading.Event) -> None:
    from streamsentinel.kafka_io import iter_json, make_consumer, wait_for_kafka

    wait_for_kafka(settings.kafka_bootstrap, timeout_s=600)
    consumer = make_consumer(settings.kafka_bootstrap, "api", [settings.scores_topic])
    log.info("Consuming %s", settings.scores_topic)
    try:
        for score in iter_json(consumer):
            if stop.is_set():
                break
            if score is not None:
                state.ingest(score)
    finally:
        consumer.close()


def create_app(state: LiveState | None = None, consume_kafka: bool | None = None) -> FastAPI:
    settings = Settings.from_env()
    state = state or LiveState(AnomalyStore(settings.anomaly_db_path))
    if consume_kafka is None:
        consume_kafka = os.getenv("STREAMSENTINEL_NO_KAFKA") != "1"

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state.bind_loop(asyncio.get_running_loop())
        stop = threading.Event()
        if consume_kafka:
            threading.Thread(
                target=_consume, args=(state, settings, stop), daemon=True, name="kafka"
            ).start()
        yield
        stop.set()

    app = FastAPI(title="StreamSentinel", version="0.1.0", lifespan=lifespan)
    app.state.live = state

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "windows_scored": state.stats()["windows_scored"]}

    @app.get("/api/symbols")
    def symbols() -> list[str]:
        return state.symbols()

    @app.get("/api/windows")
    def windows(
        symbol: str = Query(..., min_length=1),
        limit: int = Query(180, ge=1, le=360),
    ) -> list[dict[str, Any]]:
        return state.windows(symbol.upper(), limit)

    @app.get("/api/anomalies")
    def anomalies(
        limit: int = Query(50, ge=1, le=500), symbol: str | None = None
    ) -> list[dict[str, Any]]:
        return state.anomalies(limit, symbol.upper() if symbol else None)

    @app.get("/api/stats")
    def stats() -> dict[str, Any]:
        return state.stats()

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(state.metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.websocket("/ws/live")
    async def live(ws: WebSocket) -> None:
        await ws.accept()
        queue = state.subscribe()
        try:
            while True:
                await ws.send_json(await queue.get())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            state.unsubscribe(queue)
            with contextlib.suppress(RuntimeError):
                await ws.close()

    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
app = create_app()
