import numpy as np
import pandas as pd

from streamsentinel.features import compute_window_features
from streamsentinel.simulator import ANOMALY_TYPES, MarketSimulator

START = 1_700_000_000_000


def test_same_seed_gives_identical_stream():
    a = MarketSimulator(seed=5).generate(20, START)
    b = MarketSimulator(seed=5).generate(20, START)
    pd.testing.assert_frame_equal(a, b)


def test_trades_are_sorted_and_inside_their_window():
    trades = MarketSimulator(seed=2).generate(30, START)
    assert trades["ts"].is_monotonic_increasing
    assert trades["ts"].min() >= START
    assert trades["ts"].max() < START + 30 * 10_000
    assert (trades["price"] > 0).all() and (trades["qty"] > 0).all()


def test_labels_cover_whole_window_and_only_known_types(sim_trades):
    labelled = sim_trades.dropna(subset=["label"])
    assert set(labelled["label"]) <= set(ANOMALY_TYPES)
    # A window is either fully labelled or not at all.
    labelled_windows = labelled.assign(w=labelled.ts // 10_000).groupby(["symbol", "w"]).size()
    all_windows = sim_trades.assign(w=sim_trades.ts // 10_000).groupby(["symbol", "w"]).size()
    pd.testing.assert_series_equal(labelled_windows, all_windows.loc[labelled_windows.index])


def test_forced_anomalies_have_their_signature():
    normal = compute_window_features(
        MarketSimulator(seed=3, anomaly_rate=0).generate(60, START), 10
    )
    sim = MarketSimulator(seed=3)
    surge = compute_window_features(sim.generate_window(START, force="volume_surge"), 10)
    crash = compute_window_features(sim.generate_window(START + 10_000, force="flash_crash"), 10)

    for symbol in sim.symbols:
        base = normal[normal.symbol == symbol]
        s = surge[surge.symbol == symbol].iloc[0]
        c = crash[crash.symbol == symbol].iloc[0]
        assert s.volume > base.volume.median()
        assert np.log(c.close / c.open) < np.log(base.close / base.open).min()
        assert c.taker_buy_ratio < base.taker_buy_ratio.median()
