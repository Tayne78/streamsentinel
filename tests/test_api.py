import pytest
from fastapi.testclient import TestClient

from streamsentinel.api.main import create_app
from streamsentinel.api.state import AnomalyStore, LiveState


def _score(symbol="BTCUSDT", start=0, anomaly=False, label=None, source="simulator", **kw):
    score = {
        "symbol": symbol,
        "window_start": start,
        "window_end": start + 10_000,
        "vwap": 100.0,
        "close": 100.0,
        "volume": 5.0,
        "n_trades": 40,
        "taker_buy_ratio": 0.5,
        "range_pct": 0.001,
        "warmup": False,
        "is_anomaly": anomaly,
        "zmax": 8.0 if anomaly else 1.0,
        "z": {},
        "if_score": None,
        "anomaly_type": "volume_surge" if anomaly else None,
        "severity": 1.4 if anomaly else 0.2,
        "explanation": "volume 8.0σ above baseline" if anomaly else "",
        "detector_mode": "zscore",
        "model_version": "3",
        "label": label,
        "source": source,
        "detected_at": start + 12_000,
    }
    score.update(kw)
    return score


@pytest.fixture
def state():
    return LiveState(AnomalyStore(":memory:"))


@pytest.fixture
def client(state):
    with TestClient(create_app(state=state, consume_kafka=False)) as c:
        yield c


def test_health_and_dashboard(client):
    assert client.get("/health").json()["status"] == "ok"
    page = client.get("/")
    assert page.status_code == 200 and "StreamSentinel" in page.text


def test_windows_and_anomalies(client, state):
    state.ingest(_score(start=0))
    state.ingest(_score(start=10_000, anomaly=True, label="volume_surge"))
    state.ingest(_score(symbol="ETHUSDT", start=0))

    assert client.get("/api/symbols").json() == ["BTCUSDT", "ETHUSDT"]
    assert len(client.get("/api/windows", params={"symbol": "btcusdt"}).json()) == 2
    anomalies = client.get("/api/anomalies").json()
    assert [a["anomaly_type"] for a in anomalies] == ["volume_surge"]
    assert client.get("/api/anomalies", params={"symbol": "ETHUSDT"}).json() == []


def test_stats_compute_live_precision_recall(client, state):
    state.ingest(_score(start=0, anomaly=True, label="pump"))  # hit
    state.ingest(_score(start=10_000, anomaly=True))  # false alarm
    state.ingest(_score(start=20_000, label="whale_trade"))  # missed
    state.ingest(_score(start=30_000))  # true negative
    state.ingest(_score(start=40_000, anomaly=True, source="binance"))  # live data: no labels

    stats = client.get("/api/stats").json()
    gt = stats["ground_truth"]
    assert (gt["tp"], gt["fp"], gt["fn"]) == (1, 1, 1)
    assert gt["precision"] == pytest.approx(0.5) and gt["recall"] == pytest.approx(0.5)
    assert stats["anomalies"] == 3 and stats["windows_scored"] == 5
    assert stats["latency_p50_s"] == pytest.approx(2.0)
    assert stats["model"] == {"mode": "zscore", "version": "3"}


def test_prometheus_metrics(client, state):
    state.ingest(_score(start=0, anomaly=True))
    body = client.get("/metrics").text
    assert 'streamsentinel_windows_scored_total{symbol="BTCUSDT"} 1.0' in body
    assert 'streamsentinel_anomalies_total{symbol="BTCUSDT",type="volume_surge"} 1.0' in body
    assert "streamsentinel_detection_latency_seconds_bucket" in body


def test_websocket_pushes_new_scores(client, state):
    with client.websocket_connect("/ws/live") as ws:
        state.ingest(_score(start=50_000, anomaly=True))
        message = ws.receive_json()
    assert message["type"] == "score"
    assert message["data"]["window_start"] == 50_000


def test_anomalies_survive_restart(tmp_path):
    db = str(tmp_path / "a.db")
    LiveState(AnomalyStore(db)).ingest(_score(start=0, anomaly=True))
    restarted = LiveState(AnomalyStore(db))
    assert len(restarted.anomalies(10)) == 1
