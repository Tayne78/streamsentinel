"""End-to-end without the Kafka transport: simulator -> Spark -> detector -> API.

Every hop uses the same JSON the services exchange over Kafka, so this catches schema drift
between the Spark job, the detector and the API.
"""

import json
import os
import shutil
import sys

import pytest

pytest.importorskip("pyspark")
if shutil.which("java") is None and not os.getenv("JAVA_HOME"):
    pytest.skip("no Java runtime", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from streamsentinel.api.main import create_app  # noqa: E402
from streamsentinel.api.state import AnomalyStore, LiveState  # noqa: E402
from streamsentinel.detector import DetectorConfig, StreamingDetector  # noqa: E402
from streamsentinel.kafka_io import encode  # noqa: E402
from streamsentinel.simulator import MarketSimulator  # noqa: E402
from streamsentinel.streaming.spark_job import build_features, parse_trades  # noqa: E402

pytestmark = pytest.mark.spark


@pytest.fixture(scope="module")
def spark():
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[1]")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def test_full_pipeline(spark):
    sim = MarketSimulator(seed=21, anomaly_rate=0.0)
    start = 1_700_000_000_000
    normal = sim.generate(60, start)
    crash = sim.generate_window(start + 60 * 10_000, force="flash_crash")
    trades = [r for df in (normal, crash) for r in df.astype(object).to_dict("records")]

    # producer -> Kafka payloads
    payloads = [
        (encode({**t, "label": t["label"] if isinstance(t["label"], str) else None}).decode(),)
        for t in trades
    ]

    # Spark job -> features topic payloads
    features_df = build_features(parse_trades(spark.createDataFrame(payloads, ["value"])), 10, 3)
    features = [
        json.loads(row.value)
        for row in features_df.selectExpr("to_json(struct(*)) AS value").collect()
    ]
    features.sort(key=lambda f: (f["window_start"], f["symbol"]))
    assert len(features) == 61 * 3

    # detector -> scores topic payloads
    detector = StreamingDetector(DetectorConfig(z_threshold=6.0))
    scores = [json.loads(encode(detector.process(f).to_dict())) for f in features]
    last = [s for s in scores if s["window_start"] == start + 600_000]
    assert all(s["label"] == "flash_crash" for s in last)
    assert sum(s["is_anomaly"] for s in last) >= 2
    assert {s["anomaly_type"] for s in last if s["is_anomaly"]} <= {
        "price_crash",
        "sell_pressure",
        "volatility_burst",
        "activity_surge",
        "volume_surge",
    }

    # API
    state = LiveState(AnomalyStore(":memory:"))
    for s in scores:
        state.ingest(s)
    with TestClient(create_app(state=state, consume_kafka=False)) as client:
        stats = client.get("/api/stats").json()
        assert stats["windows_scored"] == len(scores)
        assert stats["ground_truth"]["tp"] >= 2
        assert len(client.get("/api/windows", params={"symbol": "SOLUSDT"}).json()) == 61
