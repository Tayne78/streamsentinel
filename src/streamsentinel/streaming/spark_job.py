"""Spark Structured Streaming job: Kafka ``trades`` -> windowed features -> Kafka ``features``.

Event-time tumbling windows per symbol with a watermark, so late trades still land in the right
window and every window is emitted exactly once (append mode). The feature definitions mirror
``streamsentinel.features.compute_window_features``; ``tests/test_spark_parity.py`` checks both.

Run with:
    spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1 spark_job.py
"""

from __future__ import annotations

import logging
import os

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

log = logging.getLogger("streamsentinel.spark")

TRADE_SCHEMA = T.StructType(
    [
        T.StructField("symbol", T.StringType()),
        T.StructField("price", T.DoubleType()),
        T.StructField("qty", T.DoubleType()),
        T.StructField("ts", T.LongType()),
        T.StructField("is_buyer_maker", T.BooleanType()),
        T.StructField("trade_id", T.LongType()),
        T.StructField("source", T.StringType()),
        T.StructField("label", T.StringType()),
    ]
)


def parse_trades(raw: DataFrame) -> DataFrame:
    """Kafka records (binary value) -> typed trades with an event-time column."""
    return (
        raw.select(F.from_json(F.col("value").cast("string"), TRADE_SCHEMA).alias("t"))
        .select("t.*")
        .where(F.col("symbol").isNotNull() & (F.col("qty") > 0) & (F.col("price") > 0))
        .withColumn("event_time", F.timestamp_millis(F.col("ts")))
    )


def build_features(trades: DataFrame, window_seconds: int, watermark_seconds: int) -> DataFrame:
    """Aggregate trades into one feature row per (symbol, tumbling window)."""
    order = F.struct("ts", "trade_id")  # deterministic open/close when timestamps tie
    agg = (
        trades.withWatermark("event_time", f"{watermark_seconds} seconds")
        .groupBy(F.window("event_time", f"{window_seconds} seconds"), F.col("symbol"))
        .agg(
            F.count(F.lit(1)).alias("n_trades"),
            F.sum("qty").alias("volume"),
            F.sum(F.col("price") * F.col("qty")).alias("notional"),
            F.min_by("price", order).alias("open"),
            F.max("price").alias("high"),
            F.min("price").alias("low"),
            F.max_by("price", order).alias("close"),
            F.sum(F.when(~F.col("is_buyer_maker"), F.col("qty")).otherwise(F.lit(0.0))).alias(
                "taker_buy_volume"
            ),
            F.max("source").alias("source"),
            F.max("label").alias("label"),
        )
    )
    return agg.select(
        "symbol",
        F.unix_millis(F.col("window.start")).alias("window_start"),
        F.unix_millis(F.col("window.end")).alias("window_end"),
        "n_trades",
        "volume",
        "notional",
        (F.col("notional") / F.col("volume")).alias("vwap"),
        "open",
        "high",
        "low",
        "close",
        "taker_buy_volume",
        (F.col("taker_buy_volume") / F.col("volume")).alias("taker_buy_ratio"),
        ((F.col("high") - F.col("low")) / (F.col("notional") / F.col("volume"))).alias("range_pct"),
        "source",
        "label",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    bootstrap = os.getenv("KAFKA_BOOTSTRAP", "localhost:9094")
    trades_topic = os.getenv("TRADES_TOPIC", "trades")
    features_topic = os.getenv("FEATURES_TOPIC", "features")
    window_seconds = int(os.getenv("WINDOW_SECONDS", "10"))
    watermark_seconds = int(os.getenv("WATERMARK_SECONDS", "3"))
    checkpoint = os.getenv("CHECKPOINT_DIR", "/tmp/streamsentinel-checkpoints/features")

    spark = (
        SparkSession.builder.appName("streamsentinel-features")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "3")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", bootstrap)
        .option("subscribe", trades_topic)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )
    features = build_features(parse_trades(raw), window_seconds, watermark_seconds)

    query = (
        features.select(
            F.col("symbol").alias("key"), F.to_json(F.struct(*features.columns)).alias("value")
        )
        .writeStream.format("kafka")
        .option("kafka.bootstrap.servers", bootstrap)
        .option("topic", features_topic)
        .option("checkpointLocation", checkpoint)
        .outputMode("append")
        .trigger(processingTime="2 seconds")
        .start()
    )
    log.info(
        "Streaming %s -> %s (window=%ss, watermark=%ss)",
        trades_topic,
        features_topic,
        window_seconds,
        watermark_seconds,
    )
    query.awaitTermination()


if __name__ == "__main__":
    main()
