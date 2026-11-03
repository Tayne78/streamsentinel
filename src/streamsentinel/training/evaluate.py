"""Offline datasets, metrics and threshold selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from streamsentinel.detector import RollingBaseline
from streamsentinel.features import compute_window_features, model_inputs
from streamsentinel.simulator import MarketSimulator

START_MS = 1_700_000_000_000


@dataclass
class Dataset:
    """Z-score vectors of all post-warm-up windows, with their ground-truth labels."""

    z: np.ndarray
    labels: np.ndarray  # anomaly type or None, dtype=object
    hours: float

    @property
    def y(self) -> np.ndarray:
        return np.array([label is not None for label in self.labels])


def build_dataset(
    hours: float,
    seed: int,
    window_seconds: int = 10,
    anomaly_rate: float = 0.02,
    history: int = 180,
    min_history: int = 30,
) -> Dataset:
    """Simulate trades, aggregate them exactly like Spark does, and replay the baseline."""
    n_windows = int(hours * 3600 / window_seconds)
    sim = MarketSimulator(window_seconds=window_seconds, anomaly_rate=anomaly_rate, seed=seed)
    features = compute_window_features(sim.generate(n_windows, START_MS), window_seconds)
    features = features.sort_values(["window_start", "symbol"], kind="mergesort")

    baseline = RollingBaseline(history, min_history)
    zs, labels = [], []
    for row in features.to_dict("records"):
        x = model_inputs(row)
        z = baseline.zscores(row["symbol"], x)
        baseline.update(row["symbol"], x)
        if z is not None:
            zs.append(z)
            labels.append(row["label"])
    return Dataset(np.vstack(zs), np.array(labels, dtype=object), hours)


def classification_metrics(pred: np.ndarray, y: np.ndarray, hours: float) -> dict[str, float]:
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarms_per_hour": fp / hours,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def recall_by_type(pred: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    out = {}
    for kind in sorted({label for label in labels if label is not None}):
        mask = labels == kind
        out[kind] = float(np.mean(pred[mask]))
    return out


def predict(mode: str, zmax: np.ndarray, if_scores: np.ndarray, z_thr: float, if_thr: float):
    z_flag = zmax >= z_thr
    if_flag = if_scores > if_thr
    if mode == "zscore":
        return z_flag
    if mode == "iforest":
        return if_flag
    return z_flag | if_flag


def select_thresholds(data: Dataset, if_scores: np.ndarray) -> dict[str, dict[str, Any]]:
    """Best F1 configuration per mode on validation data (ties -> fewer false alarms)."""
    zmax = np.abs(data.z).max(axis=1)
    y = data.y
    z_grid = np.round(np.arange(3.0, 15.01, 0.25), 2)
    if_grid = np.quantile(if_scores, np.linspace(0.90, 0.999, 60))

    candidates: dict[str, list[tuple[float, float]]] = {
        "zscore": [(z, np.inf) for z in z_grid],
        "iforest": [(np.inf, t) for t in if_grid],
        "ensemble": [(z, t) for z in z_grid for t in if_grid],
    }
    best: dict[str, dict[str, Any]] = {}
    for mode, grid in candidates.items():
        for z_thr, if_thr in grid:
            m = classification_metrics(predict(mode, zmax, if_scores, z_thr, if_thr), y, data.hours)
            key = (m["f1"], -m["fp"])
            if mode not in best or key > best[mode]["_key"]:
                best[mode] = {"z_threshold": z_thr, "if_threshold": if_thr, "val": m, "_key": key}
    for entry in best.values():
        entry.pop("_key")
    return best
