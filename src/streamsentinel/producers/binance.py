"""Live trades from Binance's public WebSocket (no API key needed) into Kafka.

    python -m streamsentinel.producers.binance

If stream.binance.com is blocked where you are, set
BINANCE_WS_URL=wss://data-stream.binance.vision/stream (Binance's market-data-only endpoint).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
from typing import Any

import websockets

from streamsentinel.config import Settings
from streamsentinel.kafka_io import encode, make_producer, wait_for_kafka

log = logging.getLogger("streamsentinel.binance")

DEFAULT_WS_URL = "wss://stream.binance.com:9443/stream"


def parse_trade(message: dict[str, Any]) -> dict[str, Any] | None:
    """Binance combined-stream trade event -> internal trade record."""
    data = message.get("data", message)
    if data.get("e") != "trade":
        return None
    return {
        "symbol": data["s"],
        "price": float(data["p"]),
        "qty": float(data["q"]),
        "ts": int(data["T"]),
        "is_buyer_maker": bool(data["m"]),
        "trade_id": int(data["t"]),
        "source": "binance",
        "label": None,
    }


def stream_url(base: str, symbols: tuple[str, ...]) -> str:
    streams = "/".join(f"{s.lower()}@trade" for s in symbols)
    return f"{base}?streams={streams}"


async def run(settings: Settings, base_url: str) -> None:
    wait_for_kafka(settings.kafka_bootstrap)
    producer = make_producer(settings.kafka_bootstrap, "binance-producer")
    url = stream_url(base_url, settings.symbols)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    backoff, sent, last_report = 1.0, 0, time.monotonic()
    while not stop.is_set():
        try:
            async with websockets.connect(url, ping_interval=20, max_size=2**20) as ws:
                log.info("Connected to %s", url)
                backoff = 1.0
                while not stop.is_set():
                    raw = await asyncio.wait_for(ws.recv(), timeout=30)
                    trade = parse_trade(json.loads(raw))
                    if trade is None:
                        continue
                    producer.produce(
                        settings.trades_topic, key=trade["symbol"], value=encode(trade)
                    )
                    producer.poll(0)
                    sent += 1
                    if time.monotonic() - last_report > 30:
                        log.info("Forwarded %d trades", sent)
                        last_report = time.monotonic()
        except (TimeoutError, OSError, websockets.WebSocketException) as exc:
            # Binance drops connections every 24h and on network hiccups: reconnect with backoff.
            log.warning("WebSocket error (%s), reconnecting in %.0fs", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)
    producer.flush(10)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    asyncio.run(run(Settings.from_env(), os.getenv("BINANCE_WS_URL", DEFAULT_WS_URL)))


if __name__ == "__main__":
    main()
