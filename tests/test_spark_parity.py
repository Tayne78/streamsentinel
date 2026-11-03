"""Spark and pandas must compute identical features, otherwise models trained offline drift
from what they see in production."""

import json
import os
import shutil
import sys

import pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark")
if shutil.which("java") is None and not os.getenv("JAVA_HOME"):
    pytest.skip("no Java runtime", allow_module_level=True)

from streamsentinel.features import compute_window_features  # noqa: E402
from streamsentinel.streaming.spark_job import build_features, parse_trades  # noqa: E402

pytestmark = pytest.mark.spark


@pytest.fixture(scope="module")
def spark():
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[1]")
        .appName("parity-test")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def test_spark_matches_pandas(spark, sim_trades):
    trades = sim_trades.head(4000)
    # Feed Spark exactly what the producers put on Kafka: JSON strings in a ``value`` column.
    values = [
        (json.dumps({k: (None if pd.isna(v) else v) for k, v in row.items()}),)
        for row in trades.astype(object).to_dict("records")
    ]
    raw = spark.createDataFrame(values, ["value"])
    got = build_features(parse_trades(raw), window_seconds=10, watermark_seconds=3).toPandas()
    expected = compute_window_features(trades, 10)

    key = ["symbol", "window_start"]
    got = got.sort_values(key).reset_index(drop=True)
    expected = expected.sort_values(key).reset_index(drop=True)
    assert len(got) == len(expected) > 0

    pd.testing.assert_frame_equal(
        got[expected.columns], expected, check_dtype=False, check_exact=False, rtol=1e-9
    )
