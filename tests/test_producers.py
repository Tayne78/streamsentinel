import json

from streamsentinel.kafka_io import encode
from streamsentinel.producers.binance import parse_trade, stream_url


def test_parse_binance_combined_stream_trade():
    message = {
        "stream": "btcusdt@trade",
        "data": {
            "e": "trade",
            "E": 1700000000123,
            "s": "BTCUSDT",
            "t": 4242,
            "p": "68123.45",
            "q": "0.0123",
            "T": 1700000000120,
            "m": True,
            "M": True,
        },
    }
    trade = parse_trade(message)
    assert trade == {
        "symbol": "BTCUSDT",
        "price": 68123.45,
        "qty": 0.0123,
        "ts": 1700000000120,
        "is_buyer_maker": True,
        "trade_id": 4242,
        "source": "binance",
        "label": None,
    }


def test_non_trade_events_are_ignored():
    assert parse_trade({"data": {"e": "kline"}}) is None


def test_stream_url_lists_all_symbols():
    url = stream_url("wss://x/stream", ("BTCUSDT", "ETHUSDT"))
    assert url == "wss://x/stream?streams=btcusdt@trade/ethusdt@trade"


def test_encode_handles_numpy_scalars():
    import numpy as np

    payload = json.loads(encode({"a": np.float64(1.5), "b": np.int64(2), "c": np.bool_(True)}))
    assert payload == {"a": 1.5, "b": 2, "c": True}
