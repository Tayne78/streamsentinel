"""Runtime settings, read from environment variables so every service is configured the same way."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _optional_float(name: str) -> float | None:
    value = os.getenv(name)
    return float(value) if value not in (None, "") else None


@dataclass(frozen=True)
class Settings:
    kafka_bootstrap: str
    trades_topic: str
    features_topic: str
    scores_topic: str
    symbols: tuple[str, ...]
    window_seconds: int
    mlflow_tracking_uri: str
    model_name: str
    model_alias: str
    z_threshold_override: float | None
    anomaly_db_path: str

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            kafka_bootstrap=_env("KAFKA_BOOTSTRAP", "localhost:9094"),
            trades_topic=_env("TRADES_TOPIC", "trades"),
            features_topic=_env("FEATURES_TOPIC", "features"),
            scores_topic=_env("SCORES_TOPIC", "scores"),
            symbols=tuple(
                s.strip().upper()
                for s in _env("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",")
                if s.strip()
            ),
            window_seconds=int(_env("WINDOW_SECONDS", "10")),
            mlflow_tracking_uri=_env("MLFLOW_TRACKING_URI", "http://localhost:5000"),
            model_name=_env("MODEL_NAME", "streamsentinel-detector"),
            model_alias=_env("MODEL_ALIAS", "champion"),
            z_threshold_override=_optional_float("Z_THRESHOLD"),
            anomaly_db_path=_env("ANOMALY_DB_PATH", "data/anomalies.db"),
        )
