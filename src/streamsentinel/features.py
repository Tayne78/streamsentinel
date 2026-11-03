"""Window features.

The Spark job (``spark/stream_features.py``) computes these in production. This module is the
pandas reference implementation of exactly the same definitions: it is used for offline training
and evaluation, and a parity test checks that Spark and pandas agree.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

# Inputs to the detectors, in a fixed order. All are scale-free or log-scaled so that the
# per-symbol baseline can turn them into comparable z-scores.
MODEL_FEATURES: tuple[str, ...] = (
    "log_return",
    "range_pct",
    "log_volume",
    "log_n_trades",
    "taker_buy_ratio",
)

FEATURE_LABELS: dict[str, str] = {
    "log_return": "return",
    "range_pct": "price range",
    "log_volume": "volume",
    "log_n_trades": "trade count",
    "taker_buy_ratio": "taker-buy share",
}

TRADE_COLUMNS = ("symbol", "price", "qty", "ts", "is_buyer_maker", "trade_id", "source", "label")


def window_start_ms(ts_ms: int, window_seconds: int) -> int:
    """Start of the tumbling window (epoch-aligned) that contains ``ts_ms``."""
    width = window_seconds * 1000
    return ts_ms - ts_ms % width


def compute_window_features(trades: pd.DataFrame, window_seconds: int) -> pd.DataFrame:
    """Aggregate trades into one row per (symbol, tumbling window).

    Expects the columns in ``TRADE_COLUMNS``; ``source`` and ``label`` are optional.
    Open/close are the prices of the first/last trade by (ts, trade_id), matching the
    ``min_by``/``max_by`` ordering used in Spark.
    """
    if trades.empty:
        return pd.DataFrame()

    width = window_seconds * 1000
    df = trades.copy()
    for col in ("source", "label"):
        if col not in df.columns:
            df[col] = None
    df["window_start"] = df["ts"] - df["ts"] % width
    df = df.sort_values(["symbol", "window_start", "ts", "trade_id"], kind="mergesort")
    df["notional"] = df["price"] * df["qty"]
    df["taker_buy_qty"] = df["qty"].where(~df["is_buyer_maker"].astype(bool), 0.0)

    out = (
        df.groupby(["symbol", "window_start"], sort=True)
        .agg(
            n_trades=("price", "size"),
            volume=("qty", "sum"),
            notional=("notional", "sum"),
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            taker_buy_volume=("taker_buy_qty", "sum"),
            source=("source", "first"),
            label=("label", "first"),
        )
        .reset_index()
    )
    out["window_end"] = out["window_start"] + width
    out["vwap"] = out["notional"] / out["volume"]
    out["taker_buy_ratio"] = out["taker_buy_volume"] / out["volume"]
    out["range_pct"] = (out["high"] - out["low"]) / out["vwap"]
    out["label"] = out["label"].astype(object).where(out["label"].notna(), None)
    out["source"] = out["source"].astype(object).where(out["source"].notna(), None)
    return out[
        [
            "symbol",
            "window_start",
            "window_end",
            "n_trades",
            "volume",
            "notional",
            "vwap",
            "open",
            "high",
            "low",
            "close",
            "taker_buy_volume",
            "taker_buy_ratio",
            "range_pct",
            "source",
            "label",
        ]
    ]


def model_inputs(window: Mapping) -> np.ndarray:
    """Turn one feature window (dict or pandas row) into the detector input vector."""
    open_, close = float(window["open"]), float(window["close"])
    return np.array(
        [
            math.log(close / open_) if open_ > 0 and close > 0 else 0.0,
            float(window["range_pct"]),
            math.log1p(float(window["volume"])),
            math.log1p(float(window["n_trades"])),
            float(window["taker_buy_ratio"]),
        ],
        dtype=float,
    )
