import math

import pandas as pd
import pytest

from streamsentinel.features import (
    MODEL_FEATURES,
    compute_window_features,
    model_inputs,
    window_start_ms,
)


def _trades(rows):
    return pd.DataFrame(
        rows, columns=["symbol", "price", "qty", "ts", "is_buyer_maker", "trade_id", "label"]
    )


def test_window_alignment_is_epoch_based():
    assert window_start_ms(1_700_000_012_345, 10) == 1_700_000_010_000
    assert window_start_ms(1_700_000_010_000, 10) == 1_700_000_010_000


def test_aggregates_match_hand_computed_values():
    trades = _trades(
        [
            ("BTCUSDT", 100.0, 1.0, 1_000, False, 1, None),  # taker buy
            ("BTCUSDT", 102.0, 3.0, 4_000, True, 2, None),  # taker sell
            ("BTCUSDT", 101.0, 1.0, 9_999, False, 3, None),  # taker buy, last in window
            ("BTCUSDT", 105.0, 2.0, 10_000, False, 4, "pump"),  # next window
        ]
    )
    out = compute_window_features(trades, 10)
    first, second = out.iloc[0], out.iloc[1]

    assert first.window_start == 0 and first.window_end == 10_000
    assert first.n_trades == 3
    assert first.volume == pytest.approx(5.0)
    assert first.vwap == pytest.approx((100 + 306 + 101) / 5)
    assert (first.open, first.high, first.low, first.close) == (100.0, 102.0, 100.0, 101.0)
    assert first.taker_buy_ratio == pytest.approx(2 / 5)
    assert first.range_pct == pytest.approx(2.0 / first.vwap)
    assert first.label is None
    assert second.label == "pump" and second.n_trades == 1


def test_open_close_use_ts_then_trade_id_ordering():
    trades = _trades(
        [
            ("ETHUSDT", 11.0, 1.0, 500, False, 2, None),
            ("ETHUSDT", 10.0, 1.0, 500, False, 1, None),
            ("ETHUSDT", 12.0, 1.0, 900, False, 3, None),
        ]
    )
    row = compute_window_features(trades, 10).iloc[0]
    assert row.open == 10.0 and row.close == 12.0


def test_symbols_are_aggregated_separately(sim_features):
    assert set(sim_features.symbol) == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
    assert not sim_features.duplicated(["symbol", "window_start"]).any()


def test_model_inputs_order_and_values():
    window = {
        "open": 100.0,
        "close": 110.0,
        "range_pct": 0.02,
        "volume": 9.0,
        "n_trades": 99,
        "taker_buy_ratio": 0.7,
    }
    x = model_inputs(window)
    assert len(x) == len(MODEL_FEATURES)
    assert x[0] == pytest.approx(math.log(1.1))
    assert x[2] == pytest.approx(math.log(10))
    assert x[3] == pytest.approx(math.log(100))
    assert x[4] == pytest.approx(0.7)
