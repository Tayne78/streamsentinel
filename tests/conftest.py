import os

# The API module builds an app at import time; keep tests away from Kafka and the real DB.
os.environ.setdefault("STREAMSENTINEL_NO_KAFKA", "1")
os.environ.setdefault("ANOMALY_DB_PATH", ":memory:")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from streamsentinel.features import compute_window_features
from streamsentinel.simulator import MarketSimulator

START_MS = 1_700_000_000_000


@pytest.fixture(scope="session")
def sim_trades() -> pd.DataFrame:
    """30 minutes of simulated trades for three symbols."""
    return MarketSimulator(seed=11, anomaly_rate=0.03).generate(180, START_MS)


@pytest.fixture(scope="session")
def sim_features(sim_trades: pd.DataFrame) -> pd.DataFrame:
    return compute_window_features(sim_trades, 10)


def normal_window(symbol: str, start: int, volume: float = 1.0, **overrides) -> dict:
    window = {
        "symbol": symbol,
        "window_start": start,
        "window_end": start + 10_000,
        "n_trades": 50,
        "volume": volume,
        "notional": volume * 100.0,
        "vwap": 100.0,
        "open": 100.0,
        "high": 100.05,
        "low": 99.95,
        "close": 100.0,
        "taker_buy_volume": volume / 2,
        "taker_buy_ratio": 0.5,
        "range_pct": 0.001,
        "source": "test",
        "label": None,
    }
    window.update(overrides)
    return window
