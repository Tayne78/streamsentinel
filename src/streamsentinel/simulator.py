"""Synthetic trade stream with labelled anomalies.

Live exchange data has no ground truth, so detection quality can't be measured on it. The
simulator produces a realistic-looking trade stream (Poisson arrivals, heavy-tailed trade sizes,
slowly drifting activity and volatility regimes) and injects five anomaly types with
randomised intensity, some of them deliberately subtle. Every trade in an injected window
carries the anomaly type as ``label``, which flows through Kafka and Spark into the scores, so
precision and recall can be measured offline *and* live on the dashboard.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ANOMALY_TYPES: tuple[str, ...] = (
    "flash_crash",
    "pump",
    "volume_surge",
    "whale_trade",
    "volatility_burst",
)


@dataclass(frozen=True)
class SymbolSpec:
    symbol: str
    start_price: float
    trades_per_sec: float
    trade_vol: float  # std of per-trade log return
    qty_median: float
    qty_sigma: float = 1.1
    price_decimals: int = 2
    qty_decimals: int = 5


DEFAULT_SPECS: tuple[SymbolSpec, ...] = (
    SymbolSpec("BTCUSDT", 68_000.0, 9.0, 0.00006, 0.004, price_decimals=2, qty_decimals=5),
    SymbolSpec("ETHUSDT", 3_500.0, 6.0, 0.00008, 0.06, price_decimals=2, qty_decimals=4),
    SymbolSpec("SOLUSDT", 160.0, 4.0, 0.00012, 1.5, price_decimals=3, qty_decimals=2),
)


@dataclass
class _SymbolState:
    price: float
    log_rate: float = 0.0  # activity regime, AR(1) in log space
    log_vol: float = 0.0  # volatility regime, AR(1) in log space
    next_trade_id: int = 1


@dataclass
class MarketSimulator:
    specs: tuple[SymbolSpec, ...] = DEFAULT_SPECS
    window_seconds: int = 10
    anomaly_rate: float = 0.02
    seed: int = 7
    source: str = "simulator"
    _rng: np.random.Generator = field(init=False, repr=False)
    _state: dict[str, _SymbolState] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._state = {s.symbol: _SymbolState(price=s.start_price) for s in self.specs}

    @property
    def symbols(self) -> list[str]:
        return [s.symbol for s in self.specs]

    def generate(self, n_windows: int, start_ms: int) -> pd.DataFrame:
        """Generate ``n_windows`` consecutive windows for all symbols."""
        width = self.window_seconds * 1000
        start_ms -= start_ms % width
        frames = [self.generate_window(start_ms + i * width) for i in range(n_windows)]
        return pd.concat(frames, ignore_index=True)

    def generate_window(self, window_start: int, force: str | None = None) -> pd.DataFrame:
        """All trades of one window for every symbol, sorted by timestamp.

        ``force`` injects the given anomaly type into every symbol (useful for demos/tests).
        """
        frames = []
        for spec in self.specs:
            if force is not None:
                anomaly = force
            elif self._rng.random() < self.anomaly_rate:
                anomaly = ANOMALY_TYPES[self._rng.integers(len(ANOMALY_TYPES))]
            else:
                anomaly = None
            frames.append(self._symbol_window(spec, window_start, anomaly))
        return pd.concat(frames, ignore_index=True).sort_values(["ts", "symbol"], kind="mergesort")

    def _symbol_window(self, spec: SymbolSpec, start: int, anomaly: str | None) -> pd.DataFrame:
        rng, st = self._rng, self._state[spec.symbol]
        width = self.window_seconds * 1000

        # Slowly drifting regimes so the baseline has to adapt instead of seeing a flat line.
        st.log_rate = 0.97 * st.log_rate + rng.normal(0, 0.06)
        st.log_vol = 0.97 * st.log_vol + rng.normal(0, 0.05)

        rate = spec.trades_per_sec * self.window_seconds * math.exp(st.log_rate)
        vol = spec.trade_vol * math.exp(st.log_vol)
        p_taker_buy = rng.beta(20, 20)
        qty_scale = 1.0
        drift = 0.0  # total log-return pushed through the window

        # Intensities are drawn wide on purpose: the low end is hard to tell from noise.
        if anomaly == "volume_surge":
            rate *= rng.uniform(1.8, 5.0)
            qty_scale *= rng.uniform(1.3, 2.5)
        elif anomaly == "pump":
            rate *= rng.uniform(1.5, 3.0)
            p_taker_buy = rng.uniform(0.72, 0.9)
            drift = rng.uniform(0.002, 0.012)
        elif anomaly == "flash_crash":
            rate *= rng.uniform(1.5, 4.0)
            p_taker_buy = rng.uniform(0.1, 0.3)
            drift = -rng.uniform(0.004, 0.025)
        elif anomaly == "volatility_burst":
            vol *= rng.uniform(3.0, 8.0)

        n = max(1, int(rng.poisson(rate)))
        ts = np.sort(start + rng.integers(0, width, size=n))
        rets = rng.normal(drift / n, vol, size=n)
        prices = st.price * np.exp(np.cumsum(rets))
        qty = spec.qty_median * np.exp(rng.normal(0, spec.qty_sigma, size=n)) * qty_scale
        is_buyer_maker = rng.random(n) >= p_taker_buy  # taker bought -> buyer is not the maker

        if anomaly == "whale_trade":
            # One block trade far outside the size distribution, with a small price impact.
            i = int(rng.integers(n))
            qty[i] = spec.qty_median * rng.uniform(150, 600)
            side = 1.0 if not is_buyer_maker[i] else -1.0
            prices[i:] *= math.exp(side * rng.uniform(0.0005, 0.003))

        prices = np.round(prices, spec.price_decimals)
        qty = np.maximum(np.round(qty, spec.qty_decimals), 10.0**-spec.qty_decimals)
        st.price = float(prices[-1])

        ids = np.arange(st.next_trade_id, st.next_trade_id + n)
        st.next_trade_id += n
        return pd.DataFrame(
            {
                "symbol": spec.symbol,
                "price": prices,
                "qty": qty,
                "ts": ts.astype(np.int64),
                "is_buyer_maker": is_buyer_maker,
                "trade_id": ids.astype(np.int64),
                "source": self.source,
                "label": anomaly,
            }
        )
