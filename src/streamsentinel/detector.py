"""Streaming anomaly detectors.

Two complementary signals:

* **Robust z-scores** per symbol: every feature is compared with a rolling baseline of the
  same symbol using median and MAD, so the baseline is not dragged around by the outliers it
  is supposed to catch. This makes BTC and SOL comparable even though their volumes differ
  by orders of magnitude, and it explains *why* a window is anomalous.
* **Isolation Forest** on the z-score vector: catches combinations of moderately unusual
  features (e.g. high volume *and* one-sided order flow) that no single z-score flags.

``StreamingDetector`` combines both according to a mode chosen during training.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

from streamsentinel.features import FEATURE_LABELS, MODEL_FEATURES, model_inputs

Mode = Literal["zscore", "iforest", "ensemble"]

_TYPE_BY_FEATURE = {
    "log_return": ("price_spike", "price_crash"),
    "range_pct": ("volatility_burst", "volatility_drought"),
    "log_volume": ("volume_surge", "volume_drought"),
    "log_n_trades": ("activity_surge", "activity_drought"),
    "taker_buy_ratio": ("buy_pressure", "sell_pressure"),
}

Z_CLIP = 1_000.0

# Smallest scale (in feature units) a z-score may be divided by. Price-derived features are
# quantised by the exchange tick: SOLUSDT moves in 0.01 steps, about 1 bp of its price, so a
# baseline built from windows that all span exactly one tick has a MAD and std of almost zero.
# Without a floor, a window spanning zero or four ticks then scores hundreds of sigmas. One
# basis point is the coarsest tick among the traded symbols; the other features stay unfloored.
MIN_SCALE_BY_FEATURE = {"log_return": 1e-4, "range_pct": 1e-4}
_MIN_SCALE = np.array([MIN_SCALE_BY_FEATURE.get(f, 0.0) for f in MODEL_FEATURES])


class RollingBaseline:
    """Per-symbol rolling median/MAD baseline over the last ``history`` windows."""

    def __init__(self, history: int = 180, min_history: int = 30) -> None:
        if min_history > history:
            raise ValueError("min_history must be <= history")
        self.history = history
        self.min_history = min_history
        self._windows: dict[str, deque[np.ndarray]] = {}

    def size(self, symbol: str) -> int:
        return len(self._windows.get(symbol, ()))

    def zscores(self, symbol: str, x: np.ndarray) -> np.ndarray | None:
        """Robust z-scores of ``x`` against the current baseline, or None during warm-up."""
        past = self._windows.get(symbol)
        if past is None or len(past) < self.min_history:
            return None
        hist = np.vstack(past)
        median = np.median(hist, axis=0)
        mad = np.median(np.abs(hist - median), axis=0)
        scale = 1.4826 * mad
        # A feature that was constant so far (MAD = 0) falls back to std, then to a tiny floor.
        fallback = hist.std(axis=0)
        scale = np.where(scale > 1e-12, scale, np.where(fallback > 1e-12, fallback, 1e-6))
        scale = np.maximum(scale, _MIN_SCALE)
        return np.clip((x - median) / scale, -Z_CLIP, Z_CLIP)

    def update(self, symbol: str, x: np.ndarray) -> None:
        self._windows.setdefault(symbol, deque(maxlen=self.history)).append(x)


@dataclass(frozen=True)
class DetectorConfig:
    mode: Mode = "zscore"
    z_threshold: float = 6.0
    if_threshold: float = 0.0
    history: int = 180
    min_history: int = 30
    model_version: str | None = None

    def with_overrides(self, z_threshold: float | None = None) -> DetectorConfig:
        if z_threshold is None:
            return self
        return DetectorConfig(
            mode=self.mode,
            z_threshold=z_threshold,
            if_threshold=self.if_threshold,
            history=self.history,
            min_history=self.min_history,
            model_version=self.model_version,
        )


@dataclass
class Score:
    symbol: str
    window_start: int
    window_end: int
    vwap: float
    close: float
    volume: float
    n_trades: int
    taker_buy_ratio: float
    range_pct: float
    warmup: bool
    is_anomaly: bool = False
    zmax: float | None = None
    z: dict[str, float] = field(default_factory=dict)
    if_score: float | None = None
    anomaly_type: str | None = None
    severity: float = 0.0
    explanation: str = ""
    detector_mode: str = "zscore"
    model_version: str | None = None
    label: str | None = None
    source: str | None = None
    detected_at: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def anomaly_type(z: np.ndarray) -> str:
    """Name the anomaly after the feature that deviates most, and its direction.

    A strong directional price move wins over the range/volume it drags along: a crash always
    widens the range too, but "price crash" is the more useful name.
    """
    abs_z = np.abs(z)
    ret = MODEL_FEATURES.index("log_return")
    if abs_z[ret] >= max(4.0, 0.5 * abs_z.max()):
        return "price_spike" if z[ret] >= 0 else "price_crash"
    idx = int(np.argmax(abs_z))
    up, down = _TYPE_BY_FEATURE[MODEL_FEATURES[idx]]
    return up if z[idx] >= 0 else down


def explain(z: np.ndarray, min_abs: float = 3.0, top: int = 2) -> str:
    """Human-readable reason, e.g. 'volume 9.1σ above baseline, price range 4.2σ above'."""
    order = np.argsort(-np.abs(z))[:top]
    parts = []
    for i in order:
        if abs(z[i]) < min_abs and parts:
            break
        direction = "above" if z[i] >= 0 else "below"
        parts.append(f"{FEATURE_LABELS[MODEL_FEATURES[i]]} {abs(z[i]):.1f}σ {direction} baseline")
    return ", ".join(parts)


class StreamingDetector:
    """Scores one feature window at a time and keeps per-symbol state."""

    def __init__(self, config: DetectorConfig | None = None, iforest: Any | None = None) -> None:
        self.config = config or DetectorConfig()
        self.iforest = iforest
        self.baseline = RollingBaseline(self.config.history, self.config.min_history)

    @property
    def effective_mode(self) -> Mode:
        if self.config.mode != "zscore" and self.iforest is None:
            return "zscore"
        return self.config.mode

    def swap_model(self, config: DetectorConfig, iforest: Any | None) -> None:
        """Hot-swap thresholds and model while keeping the learned baselines."""
        self.config = config
        self.iforest = iforest

    def iforest_score(self, z: np.ndarray) -> float:
        # decision_function < 0 means "anomalous" in sklearn; flip it so higher = more anomalous.
        return float(-self.iforest.decision_function(z.reshape(1, -1))[0])

    def process(self, window: dict[str, Any]) -> Score:
        symbol = window["symbol"]
        x = model_inputs(window)
        z = self.baseline.zscores(symbol, x)
        self.baseline.update(symbol, x)

        score = Score(
            symbol=symbol,
            window_start=int(window["window_start"]),
            window_end=int(window["window_end"]),
            vwap=float(window["vwap"]),
            close=float(window["close"]),
            volume=float(window["volume"]),
            n_trades=int(window["n_trades"]),
            taker_buy_ratio=float(window["taker_buy_ratio"]),
            range_pct=float(window["range_pct"]),
            warmup=z is None,
            detector_mode=self.effective_mode,
            model_version=self.config.model_version,
            label=window.get("label"),
            source=window.get("source"),
            detected_at=int(time.time() * 1000),
        )
        if z is None:
            return score

        zmax = float(np.max(np.abs(z)))
        z_flag = zmax >= self.config.z_threshold
        if_flag = False
        if self.effective_mode != "zscore":
            score.if_score = self.iforest_score(z)
            if_flag = score.if_score > self.config.if_threshold

        mode = self.effective_mode
        score.is_anomaly = (
            z_flag if mode == "zscore" else if_flag if mode == "iforest" else (z_flag or if_flag)
        )
        score.zmax = zmax
        score.z = {name: round(float(v), 3) for name, v in zip(MODEL_FEATURES, z, strict=True)}
        # In pure iforest mode the z threshold is disabled (huge), so scale severity to 6σ.
        reference = self.config.z_threshold if self.config.z_threshold < 1e8 else 6.0
        score.severity = round(min(zmax / reference, 10.0), 3)
        if score.is_anomaly:
            score.anomaly_type = anomaly_type(z)
            score.explanation = explain(z)
        return score


def severity_level(severity: float) -> str:
    if severity >= 2.0:
        return "high"
    if severity >= 1.3:
        return "medium"
    return "low"
