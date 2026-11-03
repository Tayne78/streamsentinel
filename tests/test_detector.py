import numpy as np
import pytest
from conftest import normal_window
from sklearn.ensemble import IsolationForest

from streamsentinel.detector import (
    DetectorConfig,
    RollingBaseline,
    StreamingDetector,
    anomaly_type,
    explain,
    severity_level,
)
from streamsentinel.features import MODEL_FEATURES


def _warm_up(detector: StreamingDetector, symbol: str = "BTCUSDT", n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    for i in range(n):
        vol = float(np.exp(rng.normal(0, 0.2)))
        ret = float(rng.normal(0, 0.0005))
        detector.process(
            normal_window(
                symbol,
                i * 10_000,
                volume=vol,
                close=100.0 * np.exp(ret),
                n_trades=int(50 * np.exp(rng.normal(0, 0.1))),
                taker_buy_ratio=float(rng.uniform(0.45, 0.55)),
                range_pct=float(abs(rng.normal(0.001, 0.0002))),
            )
        )
    return n * 10_000


def test_baseline_needs_min_history():
    baseline = RollingBaseline(history=10, min_history=3)
    x = np.ones(len(MODEL_FEATURES))
    for _ in range(3):
        assert baseline.zscores("X", x) is None
        baseline.update("X", x)
    assert baseline.zscores("X", x) is not None


def test_baseline_is_robust_and_per_symbol():
    baseline = RollingBaseline(history=50, min_history=5)
    rng = np.random.default_rng(1)
    for _ in range(40):
        baseline.update("A", rng.normal(0, 1, len(MODEL_FEATURES)))
        baseline.update("B", rng.normal(1000, 100, len(MODEL_FEATURES)))
    # A big outlier in the history barely moves median/MAD.
    baseline.update("A", np.full(len(MODEL_FEATURES), 1e6))
    z_a = baseline.zscores("A", np.zeros(len(MODEL_FEATURES)))
    z_b = baseline.zscores("B", np.full(len(MODEL_FEATURES), 1000.0))
    assert np.all(np.abs(z_a) < 1.5)
    assert np.all(np.abs(z_b) < 1.5)


def test_tick_quantised_history_does_not_saturate_the_zscore():
    # SOLUSDT-like: price moves in ~1 bp ticks, so almost every window spans exactly one tick.
    ri = MODEL_FEATURES.index("range_pct")
    tick = 1.08e-4
    baseline = RollingBaseline(history=60, min_history=30)
    for i in range(40):
        x = np.zeros(len(MODEL_FEATURES))
        x[ri] = tick * (2 if i == 7 else 1)
        baseline.update("SOL", x)
    for ticks in (0, 4):
        x = np.zeros(len(MODEL_FEATURES))
        x[ri] = tick * ticks
        assert abs(baseline.zscores("SOL", x)[ri]) < 6
    burst = np.zeros(len(MODEL_FEATURES))
    burst[ri] = 0.01  # 100 bp: a real volatility burst must still stand out
    assert baseline.zscores("SOL", burst)[ri] > 6


def test_warmup_windows_are_never_flagged():
    detector = StreamingDetector(DetectorConfig(min_history=30))
    score = detector.process(normal_window("BTCUSDT", 0, volume=1e9))
    assert score.warmup and not score.is_anomaly


def test_volume_spike_is_detected_and_explained():
    detector = StreamingDetector(DetectorConfig(z_threshold=6.0))
    t = _warm_up(detector)
    score = detector.process(normal_window("BTCUSDT", t, volume=40.0, n_trades=52))
    assert score.is_anomaly
    assert score.anomaly_type == "volume_surge"
    assert "volume" in score.explanation and "above baseline" in score.explanation
    assert score.severity >= 1.0


def test_normal_window_after_warmup_is_not_flagged():
    detector = StreamingDetector(DetectorConfig(z_threshold=6.0))
    t = _warm_up(detector)
    score = detector.process(normal_window("BTCUSDT", t, volume=1.0))
    assert not score.warmup and not score.is_anomaly
    assert set(score.z) == set(MODEL_FEATURES)


def test_crash_is_classified_by_direction():
    z = np.zeros(len(MODEL_FEATURES))
    z[MODEL_FEATURES.index("log_return")] = -12.0
    assert anomaly_type(z) == "price_crash"
    z[MODEL_FEATURES.index("log_return")] = 12.0
    assert anomaly_type(z) == "price_spike"
    assert explain(z).startswith("return 12.0σ above baseline")


def test_strong_price_move_beats_the_range_it_drags_along():
    z = np.zeros(len(MODEL_FEATURES))
    z[MODEL_FEATURES.index("range_pct")] = 22.0
    z[MODEL_FEATURES.index("log_return")] = -13.0
    assert anomaly_type(z) == "price_crash"
    z[MODEL_FEATURES.index("log_return")] = -2.0  # small move: it really is volatility
    assert anomaly_type(z) == "volatility_burst"


def test_iforest_mode_falls_back_to_zscore_without_model():
    detector = StreamingDetector(DetectorConfig(mode="ensemble"), iforest=None)
    assert detector.effective_mode == "zscore"


def test_ensemble_uses_isolation_forest_scores():
    rng = np.random.default_rng(3)
    forest = IsolationForest(n_estimators=50, random_state=0).fit(
        rng.normal(0, 1, (500, len(MODEL_FEATURES)))
    )
    config = DetectorConfig(mode="ensemble", z_threshold=1e9, if_threshold=0.0)
    detector = StreamingDetector(config, iforest=forest)
    t = _warm_up(detector)
    score = detector.process(normal_window("BTCUSDT", t, volume=40.0))
    assert score.if_score is not None and score.if_score > 0
    assert score.is_anomaly  # flagged by the forest although the z threshold is unreachable


def test_swap_model_keeps_baseline():
    detector = StreamingDetector()
    _warm_up(detector, n=40)
    before = detector.baseline.size("BTCUSDT")
    detector.swap_model(DetectorConfig(z_threshold=4.0), None)
    assert detector.baseline.size("BTCUSDT") == before
    assert detector.config.z_threshold == 4.0


@pytest.mark.parametrize(("severity", "level"), [(0.5, "low"), (1.5, "medium"), (3.0, "high")])
def test_severity_levels(severity, level):
    assert severity_level(severity) == level
